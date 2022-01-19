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

    include!("../bindings.rs");
}
use bindings::*;

use drop_guard::guard;
use num_complex::{Complex32, Complex64};
use num_traits::Zero;

pub fn gpu_count() -> usize {
    unsafe {
        let mut count = 0i32;
        cudaGetDeviceCount(&mut count);
        count as usize
    }
}

pub fn first_gpu_name() -> Result<String> {
    unsafe {
        check_device()?;

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

fn check_device() -> Result<()> {
    if gpu_count() == 0 {
        Err(CudaError::NoDevice)
    } else {
        Ok(())
    }
}

/*
    f32 helpers
*/

pub fn fft32_norm(data: &[f32]) -> Result<Vec<f32>> {
    fft32_norm_batch(&[data]).map(|batch| batch.to_vec().pop().unwrap())
}

pub fn fft32(data: &[f32]) -> Result<Vec<Complex32>> {
    fft32_batch(&[data]).map(|batch| batch.to_vec().pop().unwrap())
}

pub fn fft32_norm_batch(batch: &[&[f32]]) -> Result<Vec<Vec<f32>>> {
    match fft32_batch(batch) {
        Ok(batch) => Ok(batch
            .to_vec()
            .into_iter()
            .map(|fft| fft.to_vec().into_iter().map(|z| z.norm()).collect())
            .collect()),
        Err(e) => Err(e),
    }
}

pub fn fft32_batch(batch: &[&[f32]]) -> Result<Vec<Vec<Complex32>>> {
    unsafe { fft_batch::<FFT32>(batch) }
}

/*
    f64 helpers
*/

pub fn fft64_norm(data: &[f64]) -> Result<Vec<f64>> {
    fft64_norm_batch(&[data]).map(|batch| batch.to_vec().pop().unwrap())
}

pub fn fft64(data: &[f64]) -> Result<Vec<Complex64>> {
    fft64_batch(&[data]).map(|batch| batch.to_vec().pop().unwrap())
}

pub fn fft64_norm_batch(batch: &[&[f64]]) -> Result<Vec<Vec<f64>>> {
    match fft64_batch(batch) {
        Ok(batch) => Ok(batch
            .to_vec()
            .into_iter()
            .map(|fft| fft.to_vec().into_iter().map(|z| z.norm()).collect())
            .collect()),
        Err(e) => Err(e),
    }
}

pub fn fft64_batch(batch: &[&[f64]]) -> Result<Vec<Vec<Complex64>>> {
    unsafe { fft_batch::<FFT64>(batch) }
}

/*
    Generic implementation
*/

unsafe fn fft_batch<MODE: FFTMode>(batch: &[&[MODE::Float]]) -> Result<Vec<Vec<MODE::Complex>>> {
    check_device()?;

    // Amount of datasets in the batch
    let n_batch = batch.len();
    // Amount of points per dataset
    let n = batch[0].len();
    // Amount of complex values per DFT of a dataset
    let n_dft = n / 2 + 1;

    // Deal with empty datasets
    if n_batch == 0 {
        return Ok(vec![]);
    }
    if n == 0 {
        return Ok(vec![vec![]; n_batch]);
    }

    // Check data length uniformity
    for &data in batch {
        if data.len() != n {
            return Err(CudaError::InvalidValue);
        }
    }

    // Byte size of a dataset
    let bytes_single = std::mem::size_of::<MODE::CudaIn>() * n;
    // Byte size of the batch
    let bytes_batch = bytes_single * n_batch;
    // Byte size of the DFT of a dataset
    let bytes_single_dft = std::mem::size_of::<MODE::CudaOut>() * n_dft;
    // Byte size of the DFT batch
    let bytes_batch_dft = bytes_single_dft * n_batch;

    // Prepare plan
    let mut plan: cufftHandle = 0;
    CudaError::from_raw(cufftPlan1d(
        &mut plan,
        n as i32,
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
        bytes_batch as size_t,
    ))?;
    let gpu_data_in = guard(gpu_data_in, |gpu_data_in| {
        let _ = cudaFree(gpu_data_in as *mut _);
    });

    let mut gpu_data_out: *mut MODE::CudaOut = std::ptr::null_mut();
    CudaError::from_raw(cudaMalloc(
        &mut gpu_data_out as *mut _ as *mut _,
        bytes_batch_dft as size_t,
    ))?;
    let gpu_data_out = guard(gpu_data_out, |gpu_data_out| {
        let _ = cudaFree(gpu_data_out as *mut _);
    });

    // Initialize GPU memory
    for (i, &data) in batch.iter().enumerate() {
        CudaError::from_raw(cudaMemcpy(
            (*gpu_data_in).offset((n * i) as isize) as *mut _,
            data.as_ptr() as *const _,
            bytes_single as size_t,
            cudaMemcpyKind_cudaMemcpyHostToDevice,
        ))?;
    }

    // Execute FFT
    // Unnormalized
    // Recommendations: https://docs.nvidia.com/cuda/cufft/index.html#accuracy-and-performance
    CudaError::from_raw(MODE::exec(*plan, *gpu_data_in, *gpu_data_out))?;
    CudaError::from_raw(cudaDeviceSynchronize())?;

    // Retrieve results
    // Safety: Complex32/64 is repr(C) and has the same layout as cufftComplex/cufftDoubleComplex
    let mut buf = vec![vec![MODE::Complex::zero(); n_dft]; n_batch];
    for (i, out) in buf.iter_mut().enumerate() {
        CudaError::from_raw(cudaMemcpy(
            out.as_mut_ptr() as *mut _,
            (*gpu_data_out).offset((i * n_dft) as isize) as *const _ as *const _,
            bytes_single_dft as size_t,
            cudaMemcpyKind_cudaMemcpyDeviceToHost,
        ))?;
    }

    Ok(buf)
}

unsafe trait FFTMode {
    type Float: Copy + Clone;
    type Complex: Copy + Clone + Zero;

    type CudaIn;
    type CudaOut;

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
    dbg!(first_gpu_name().unwrap());
}

#[test]
fn test_fft32() {
    // Generate test data
    let y = (0..2u64.pow(14)) // 16384
        .map(|x| x as f32 / 100.0 * std::f32::consts::TAU)
        .map(|x| x.cos())
        .collect::<Vec<_>>();
    let fft = fft32_norm(&y).unwrap();

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
    let batch = fft32_norm_batch(&[y.as_ref(); 100]).unwrap();

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
    let fft = fft64_norm(&y).unwrap();

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
    let batch = fft64_norm_batch(&[y.as_ref(); 100]).unwrap();

    let fft_0 = &batch[0];
    for fft in batch.iter() {
        approx::assert_abs_diff_eq!(fft_0.as_slice(), fft.as_slice());
    }
}
