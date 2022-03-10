/*
    lib.rs : cufft_rust. A small, safe Rust wrapper around CUDA FFT.

    Copyright (C) 2021 Pim van den Berg

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#![warn(clippy::all)]

mod error;
pub use error::*;

pub mod bindings {
    #![allow(dead_code)]
    #![allow(non_upper_case_globals)]
    #![allow(non_camel_case_types)]
    #![allow(non_snake_case)]
    #![allow(deref_nullptr)]

    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}
use bindings::*;

use drop_guard::guard;
use num_complex::{Complex32, Complex64};
use num_traits::Zero;
use std::cell::Cell;
use std::thread_local;

pub fn frequencies(dt: f64, n: usize) -> impl Iterator<Item = f64> {
    // Sampled frequencies : k/(N dt)
    let freq_normalisation = 1.0 / (dt * n as f64);
    (0..n).map(move |i| i as f64 * freq_normalisation)
}

pub fn gpu_count() -> usize {
    thread_local! {
        // Cache to reduce calls to driver
        static COUNT: Cell<Option<usize>> = Cell::new(None);
    }

    COUNT.with(|cell| match cell.get() {
        Some(count) => count,
        None => unsafe {
            let mut count = 0i32;
            let _ = CudaError::from_raw(cudaGetDeviceCount(&mut count));
            let count = count as usize;
            cell.set(Some(count));
            count
        },
    })
}

pub fn gpu_name() -> Result<String> {
    unsafe {
        if gpu_count() == 0 {
            return Err(CudaError::NoDevice);
        }

        let mut properties = std::mem::MaybeUninit::<cudaDeviceProp>::uninit();
        CudaError::from_raw(cudaGetDeviceProperties(properties.as_mut_ptr(), 0))?;
        let properties = properties.assume_init();

        let name_len = properties.name.iter().position(|&x| x == 0).unwrap() + 1;
        let name = std::ffi::CStr::from_bytes_with_nul(std::slice::from_raw_parts(
            properties.name.as_ptr() as *const u8,
            name_len,
        ))
        .unwrap();
        Ok(name.to_str().unwrap().to_owned())
    }
}

pub fn gpu_memory() -> Result<u64> {
    unsafe {
        if gpu_count() == 0 {
            return Err(CudaError::NoDevice);
        }

        let mut properties = std::mem::MaybeUninit::<cudaDeviceProp>::uninit();
        CudaError::from_raw(cudaGetDeviceProperties(properties.as_mut_ptr(), 0))?;
        let properties = properties.assume_init();

        Ok(properties.totalGlobalMem)
    }
}

fn can_host_register() -> bool {
    unsafe {
        let mut can_host_register = 0i32;
        CudaError::from_raw(cudaDeviceGetAttribute(
            &mut can_host_register,
            cudaDeviceAttr_cudaDevAttrHostRegisterSupported,
            0,
        ))
        .unwrap();
        can_host_register != 0
    }
}

fn can_host_register_readonly() -> bool {
    unsafe {
        let mut can_host_register_readonly = 0i32;
        CudaError::from_raw(cudaDeviceGetAttribute(
            &mut can_host_register_readonly,
            cudaDeviceAttr_cudaDevAttrHostRegisterReadOnlySupported,
            0,
        ))
        .unwrap();
        can_host_register_readonly != 0
    }
}

/*
    f32 helpers
*/

pub fn fft32(data: &[f32], add_zeroes: usize, n_out: Option<usize>) -> Result<Vec<Complex32>> {
    fft32_batch(&[data], add_zeroes, n_out).map(|mut batch| batch.pop().unwrap())
}

pub fn fft32_norm(data: &[f32], add_zeroes: usize, n_out: Option<usize>) -> Result<Vec<f32>> {
    fft32_norm_batch(&[data], add_zeroes, n_out).map(|mut batch| batch.pop().unwrap())
}

pub fn fft32_norm_batch(
    batch: &[&[f32]],
    add_zeroes: usize,
    n_out: Option<usize>,
) -> Result<Vec<Vec<f32>>> {
    match fft32_batch(batch, add_zeroes, n_out) {
        Ok(batch) => Ok(batch
            .into_iter()
            .map(|fft| fft.into_iter().map(|z| z.norm()).collect())
            .collect()),
        Err(e) => Err(e),
    }
}

pub fn fft32_batch(
    batch: &[&[f32]],
    add_zeroes: usize,
    n_out: Option<usize>,
) -> Result<Vec<Vec<Complex32>>> {
    unsafe { fft_batch::<FFT32>(batch, add_zeroes, n_out) }
}

/*
    f64 helpers
*/

pub fn fft64(data: &[f64], add_zeroes: usize, n_out: Option<usize>) -> Result<Vec<Complex64>> {
    fft64_batch(&[data], add_zeroes, n_out).map(|mut batch| batch.pop().unwrap())
}

pub fn fft64_norm(data: &[f64], add_zeroes: usize, n_out: Option<usize>) -> Result<Vec<f64>> {
    fft64_norm_batch(&[data], add_zeroes, n_out).map(|mut batch| batch.pop().unwrap())
}

pub fn fft64_norm_batch(
    batch: &[&[f64]],
    add_zeroes: usize,
    n_out: Option<usize>,
) -> Result<Vec<Vec<f64>>> {
    match fft64_batch(batch, add_zeroes, n_out) {
        Ok(batch) => Ok(batch
            .into_iter()
            .map(|fft| fft.into_iter().map(|z| z.norm()).collect())
            .collect()),
        Err(e) => Err(e),
    }
}

pub fn fft64_batch(
    batch: &[&[f64]],
    add_zeroes: usize,
    n_out: Option<usize>,
) -> Result<Vec<Vec<Complex64>>> {
    unsafe { fft_batch::<FFT64>(batch, add_zeroes, n_out) }
}

/*
    Generic implementation
*/

unsafe fn fft_batch<MODE: FFTMode>(
    batch: &[&[MODE::Float]],
    add_zeroes: usize,
    n_out: Option<usize>,
) -> Result<Vec<Vec<MODE::Complex>>> {
    debug_assert_eq!(
        std::mem::size_of::<MODE::Float>(),
        std::mem::size_of::<MODE::CudaIn>()
    );
    debug_assert_eq!(
        std::mem::size_of::<MODE::Complex>(),
        std::mem::size_of::<MODE::CudaOut>()
    );

    if gpu_count() == 0 {
        return Err(CudaError::NoDevice);
    }

    // Amount of datasets in the batch
    let n_batch = batch.len();
    // Amount of points per dataset
    let n = batch[0].len();
    // Amount of complex values per FFT of a dataset
    let n_complex = (n + add_zeroes) / 2 + 1;

    // Check if n_out has a reasonable value
    if let Some(n_out) = n_out {
        if n_out > n_complex {
            return Err(CudaError::InvalidValue);
        }
    }

    // Deal with empty datasets
    if n_batch == 0 {
        return Ok(vec![]);
    }
    if n == 0 {
        return Ok(vec![vec![]; n_batch]);
    }

    // Check data length uniformity
    for data in batch {
        if data.len() != n {
            return Err(CudaError::InvalidValue);
        }
    }

    // Byte size of a dataset
    let bytes_single_cpu = std::mem::size_of::<MODE::CudaIn>() * n;
    let bytes_single_gpu = std::mem::size_of::<MODE::CudaIn>() * (n + add_zeroes);

    // Byte size of the batch
    //let bytes_batch_cpu = bytes_single_cpu * n_batch;
    let bytes_batch_gpu = bytes_single_gpu * n_batch;

    // Byte size of the FFT of a dataset
    let bytes_single_complex_gpu = std::mem::size_of::<MODE::CudaOut>() * n_complex;
    // Byte size of the FFT batch
    let bytes_batch_complex_gpu = bytes_single_complex_gpu * n_batch;

    // Prepare plan
    let mut plan: cufftHandle = 0;
    CudaError::from_raw(cufftPlan1d(
        &mut plan,
        (n + add_zeroes) as i32,
        MODE::FFT_TYPE,
        n_batch as i32,
    ))?;
    let plan = guard(plan, |plan| {
        let _ = cufftDestroy(plan);
    });

    // Allocate space on GPU
    let mut gpu_data_in: *mut MODE::CudaIn = std::ptr::null_mut();
    CudaError::from_raw(cudaMalloc(
        &mut gpu_data_in as *mut _ as *mut _,
        bytes_batch_gpu as size_t,
    ))?;
    let gpu_data_in = guard(gpu_data_in, |gpu_data_in| {
        let _ = cudaFree(gpu_data_in as *mut _);
    });

    let mut gpu_data_out: *mut MODE::CudaOut = std::ptr::null_mut();
    CudaError::from_raw(cudaMalloc(
        &mut gpu_data_out as *mut _ as *mut _,
        bytes_batch_complex_gpu as size_t,
    ))?;
    let gpu_data_out = guard(gpu_data_out, |gpu_data_out| {
        let _ = cudaFree(gpu_data_out as *mut _);
    });

    // Initialize GPU memory
    for (i, data) in batch.iter().enumerate() {
        // Page lock memory for DMA
        if can_host_register() {
            CudaError::from_raw(cudaHostRegister(
                data.as_ptr() as *mut _,
                bytes_single_cpu as size_t,
                if can_host_register_readonly() {
                    cudaHostRegisterReadOnly
                } else {
                    cudaHostRegisterDefault
                },
            ))?;
        }

        let _guard = guard(data.as_ptr(), |ptr| {
            // Undo page lock
            if can_host_register() {
                let _ = cudaHostUnregister(ptr as *mut _);
            }
        });

        // Copy data to GPU
        CudaError::from_raw(cudaMemcpy(
            (*gpu_data_in).add((n + add_zeroes) * i) as *mut _,
            data.as_ptr() as *const _,
            bytes_single_cpu as size_t,
            cudaMemcpyKind_cudaMemcpyHostToDevice,
        ))?;

        // We already allocated memory for zero padding, but the values are still undefined.
        // Therefore we must set the remaining values to zero.
        if add_zeroes > 0 {
            CudaError::from_raw(cudaMemset(
                (*gpu_data_in).add((n + add_zeroes) * i + n) as *mut _,
                0,
                (bytes_single_gpu - bytes_single_cpu) as size_t,
            ))?;
        }
    }

    // Execute FFT
    // Unnormalized
    // Recommendations: https://docs.nvidia.com/cuda/cufft/index.html#accuracy-and-performance
    CudaError::from_raw(MODE::exec(*plan, *gpu_data_in, *gpu_data_out))?;
    CudaError::from_raw(cudaDeviceSynchronize())?;

    // Retrieve results
    // NB. vec![Vec::with_capacity()] does NOT keep the capacity between the cloned values!
    let mut buf = (0..n_batch)
        .map(|_| Vec::<MODE::Complex>::with_capacity(n_out.unwrap_or(n_complex)))
        .collect::<Vec<_>>();

    for (i, out) in buf.iter_mut().enumerate() {
        let out_bytes = out.capacity() * std::mem::size_of::<MODE::Complex>();

        // Page lock memory for DMA
        if can_host_register() {
            CudaError::from_raw(cudaHostRegister(
                out.as_mut_ptr() as *mut _,
                out_bytes as size_t,
                cudaHostRegisterDefault,
            ))?;
        }

        let _guard = guard(out.as_mut_ptr(), |ptr| {
            // Undo page lock
            if can_host_register() {
                let _ = cudaHostUnregister(ptr as *mut _);
            }
        });

        CudaError::from_raw(cudaMemcpy(
            out.as_mut_ptr() as *mut _,
            (*gpu_data_out).add(i * n_complex) as *const _ as *const _,
            out_bytes as size_t,
            cudaMemcpyKind_cudaMemcpyDeviceToHost,
        ))?;

        // Safety: cudaMemcpy will initialize all allocated values
        out.set_len(out.capacity());
    }

    Ok(buf)
}

unsafe trait FFTMode {
    type Float: Copy + Clone;
    type Complex: Copy + Clone + Zero;

    // These must be transmutable to Float/Complex
    type CudaIn: Copy + Clone;
    type CudaOut: Copy + Clone;

    const FFT_TYPE: cufftType_t;

    unsafe fn exec(
        plan: cufftHandle,
        idata: *mut Self::CudaIn,
        odata: *mut Self::CudaOut,
    ) -> cufftResult_t;
}

struct FFT32;
unsafe impl FFTMode for FFT32 {
    type Float = f32;
    type Complex = Complex32;

    type CudaIn = cufftReal;
    type CudaOut = cufftComplex;

    const FFT_TYPE: cufftType_t = cufftType_t_CUFFT_R2C;

    unsafe fn exec(
        plan: cufftHandle,
        idata: *mut Self::CudaIn,
        odata: *mut Self::CudaOut,
    ) -> cufftResult_t {
        cufftExecR2C(plan, idata, odata)
    }
}

struct FFT64;
unsafe impl FFTMode for FFT64 {
    type Float = f64;
    type Complex = Complex64;

    type CudaIn = cufftDoubleReal;
    type CudaOut = cufftDoubleComplex;

    const FFT_TYPE: cufftType_t = cufftType_t_CUFFT_D2Z;

    unsafe fn exec(
        plan: cufftHandle,
        idata: *mut Self::CudaIn,
        odata: *mut Self::CudaOut,
    ) -> cufftResult_t {
        cufftExecD2Z(plan, idata, odata)
    }
}

#[test]
fn test_cuda_device() {
    dbg!(gpu_count());
    dbg!(gpu_name().unwrap());
    dbg!(can_host_register());
    dbg!(can_host_register_readonly());
    let megabytes = gpu_memory().unwrap() / 10u64.pow(6);
    dbg!(megabytes);
}

#[test]
fn test_fft32() {
    // Generate test data
    let y = (0..2u64.pow(14)) // 16384
        .map(|x| x as f32 / 100.0 * std::f32::consts::TAU)
        .map(|x| x.cos())
        .collect::<Vec<_>>();
    let fft = fft32_norm(y.as_slice(), 0, None).unwrap();

    // f = k/T so 1=k/(16384 / 100) -> k=164
    assert!(fft[164] > fft[163]);
    assert!(fft[164] > fft[165]);
}

#[test]
fn test_fft32_batch() {
    // Generate test data
    let y = (0..2u64.pow(10))
        .map(|x| x as f32 / 2.0f32.powi(8) * std::f32::consts::TAU)
        .map(|x| x.cos())
        .collect::<Vec<_>>();
    let batch = fft32_norm_batch(&[y.as_slice(); 100], 0, None).unwrap();

    let fft_0 = &batch[0];
    for fft in batch.iter() {
        approx::assert_abs_diff_eq!(fft_0.as_slice(), fft.as_slice());
    }
}

#[test]
fn test_fft64() {
    // Generate test data
    let y = (0..2u64.pow(14)) // 16384
        .map(|x| x as f64 / 100.0 * std::f64::consts::TAU)
        .map(|x| x.cos())
        .collect::<Vec<_>>();
    let fft = fft64_norm(y.as_slice(), 0, None).unwrap();

    // f = k/T so 1=k/(16384 / 100) -> k=164
    assert!(fft[164] > fft[163]);
    assert!(fft[164] > fft[165]);
}

#[test]
fn test_fft64_batch() {
    // Generate test data
    let y = (0..2u64.pow(10))
        .map(|x| x as f64 / 2.0f64.powi(8) * std::f64::consts::TAU)
        .map(|x| x.cos())
        .collect::<Vec<_>>();
    let batch = fft64_norm_batch(&[y.as_slice(); 100], 0, None).unwrap();

    let fft_0 = &batch[0];
    for fft in batch.iter() {
        approx::assert_abs_diff_eq!(fft_0.as_slice(), fft.as_slice());
    }
}

#[test]
fn test_fft64_zeropad() {
    // Generate test data
    let y = (0..2u64.pow(14)) // 16384
        .map(|x| x as f64 / 100.0 * std::f64::consts::TAU)
        .map(|x| x.cos())
        .collect::<Vec<_>>();

    // Pad to 2^20: add 1032192
    let fft = fft64_norm(y.as_slice(), 1032192, None).unwrap();

    // Expected length: n/2 + 1 = 2^19 + 1 = 524289
    assert_eq!(fft.len(), 524289);

    // f = k/T so 1=k/(1048576 / 100) -> k=10485.8
    assert!(fft[10486] > fft[10485]);
    assert!(fft[10486] > fft[10487]);
}

#[test]
fn test_fft64_truncate() {
    // Generate test data
    let y = (0..2u64.pow(14)) // 16384
        .map(|x| x as f64 / 100.0 * std::f64::consts::TAU)
        .map(|x| x.cos())
        .collect::<Vec<_>>();

    let fft = fft64_norm(y.as_slice(), 0, Some(10)).unwrap();

    assert_eq!(fft.len(), 10);
}
