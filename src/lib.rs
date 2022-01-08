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
use num_complex::Complex32;
use num_traits::Zero;

pub fn forward_transform_f32_norm(data: &[f32]) -> Result<Box<[f32]>> {
    forward_transform_f32(data).map(|fft| fft.to_vec().into_iter().map(|z| z.norm()).collect())
}

pub fn forward_transform_f32(data: &[f32]) -> Result<Box<[Complex32]>> {
    unsafe {
        if data.len() == 0 {
            return Ok(Box::new([]));
        }

        let mut plan: cufftHandle = 0;
        let mut gpu_data_in: *mut cufftReal = std::ptr::null_mut();
        let mut gpu_data_out: *mut cufftComplex = std::ptr::null_mut();

        let in_len = (std::mem::size_of::<cufftReal>() * data.len()) as u64;
        let out_len = (std::mem::size_of::<cufftComplex>() * (data.len() / 2 + 1)) as u64;

        // Prepare plan
        CudaError::from_raw(cufftPlan1d(
            &mut plan,
            data.len() as i32,
            cufftType_t_CUFFT_R2C,
            1,
        ))?;
        let plan = guard(plan, |plan| {
            let _ = cufftDestroy(plan);
        });

        // Allocate space on GPU
        CudaError::from_raw(cudaMalloc(&mut gpu_data_in as *mut _ as *mut _, in_len))?;
        let gpu_data_in = guard(gpu_data_in, |gpu_data_in| {
            let _ = cudaFree(gpu_data_in as *mut _);
        });
        CudaError::from_raw(cudaMalloc(&mut gpu_data_out as *mut _ as *mut _, out_len))?;
        let gpu_data_out = guard(gpu_data_out, |gpu_data_out| {
            let _ = cudaFree(gpu_data_out as *mut _);
        });

        // Initialize GPU memory
        CudaError::from_raw(cudaMemcpy(
            *gpu_data_in as *mut _,
            data.as_ptr() as *const _,
            std::mem::size_of_val(data) as u64,
            cudaMemcpyKind_cudaMemcpyHostToDevice,
        ))?;

        // Execute FFT
        // Unnormalized
        // Recommendations: https://docs.nvidia.com/cuda/cufft/index.html#accuracy-and-performance
        CudaError::from_raw(cufftExecR2C(*plan, *gpu_data_in, *gpu_data_out))?;
        CudaError::from_raw(cudaDeviceSynchronize())?;

        // Retrieve result
        // Safety: Complex32 is repr(C) and has the same layout as cufftComplex
        let mut buf = vec![Complex32::zero(); data.len() / 2 + 1].into_boxed_slice();
        debug_assert_eq!(std::mem::size_of_val(buf.as_ref()), out_len as usize);
        CudaError::from_raw(cudaMemcpy(
            buf.as_mut_ptr() as *mut _,
            *gpu_data_out as *const _ as *const _,
            out_len,
            cudaMemcpyKind_cudaMemcpyDeviceToHost,
        ))?;

        Ok(buf)
    }
}

#[test]
fn test_cuda_device() {
    unsafe {
        let mut count = 0i32;
        cudaGetDeviceCount(&mut count);
        dbg!(count);

        if count >= 1 {
            let mut properties = std::mem::MaybeUninit::<cudaDeviceProp>::uninit();
            let status = cudaGetDeviceProperties(properties.as_mut_ptr(), 0);
            assert!(CudaError::from_raw(status).is_ok());
            let properties = properties.assume_init();
            //dbg!(properties);
            let name_len = properties.name.iter().position(|&x| x == 0).unwrap() + 1;
            let name = std::ffi::CStr::from_bytes_with_nul(std::slice::from_raw_parts(
                properties.name.as_ptr() as *const u8,
                name_len,
            ))
            .unwrap();
            dbg!(name.to_str().unwrap());
        }
    }
}

#[test]
fn test_fft() {
    // Generate test data
    let y = (0..10_000)
        .map(|x| x as f32 / 100.0 * std::f32::consts::TAU)
        .map(|x| x.cos())
        .collect::<Vec<_>>();
    let fft = forward_transform_f32_norm(&y).unwrap();

    // f = k/T so 1=k/100 -> k=100
    assert!(fft[100] > fft[99]);
    assert!(fft[100] > fft[101]);

    // f = k/T
    // let x = (0..10_000).map(|x| x as f32 / 100.0).collect::<Vec<_>>();
    // let y = fft;

    // fn store<'a, 'b>(to: &str, x: &[f32], y: &[f32]) {
    //     use std::io::{BufWriter, Write};
    //     let mut out = BufWriter::with_capacity(2usize.pow(16), std::fs::File::create(to).unwrap());
    //     for (x, y) in x.iter().zip(y.iter()) {
    //         writeln!(&mut out, "{:.4},{:.4}", *x, *y).unwrap();
    //     }
    // }
    // store("test_fft.csv", &x, &y);
}
