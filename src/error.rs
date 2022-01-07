/*
    error.rs

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

use crate::bindings::*;
use std::error::Error;
use std::fmt;
use std::os::raw::c_int;

pub type Result<T> = std::result::Result<T, CudaError>;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum CudaError {
    Success,
    InvalidValue,
    MemoryAllocation,
    Initialization,
    InsufficientDriver,
    NoDevice,
    InvalidResourceHandle,
    Other(c_int),
}

#[allow(non_upper_case_globals)]
impl From<c_int> for CudaError {
    fn from(i: c_int) -> Self {
        match i {
            cudaError_cudaSuccess => CudaError::Success,
            cudaError_cudaErrorInvalidValue => CudaError::InvalidValue,
            cudaError_cudaErrorMemoryAllocation => CudaError::MemoryAllocation,
            cudaError_cudaErrorInitializationError => CudaError::Initialization,
            cudaError_cudaErrorInsufficientDriver => CudaError::InsufficientDriver,
            cudaError_cudaErrorNoDevice => CudaError::NoDevice,
            cudaError_cudaErrorInvalidResourceHandle => CudaError::InvalidResourceHandle,
            other => CudaError::Other(other),
        }
    }
}
impl Error for CudaError {}

impl fmt::Display for CudaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}
