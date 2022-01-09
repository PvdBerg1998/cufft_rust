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

pub type Result<T> = std::result::Result<T, CudaError>;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum CudaError {
    InvalidValue,
    MemoryAllocation,
    Initialization,
    InsufficientDriver,
    NoDevice,
    InvalidResourceHandle,
    Other(cudaError),
}

impl Into<cudaError> for CudaError {
    fn into(self) -> cudaError {
        match self {
            Self::InvalidValue => cudaError_cudaErrorInvalidValue,
            Self::MemoryAllocation => cudaError_cudaErrorMemoryAllocation,
            Self::Initialization => cudaError_cudaErrorInitializationError,
            Self::InsufficientDriver => cudaError_cudaErrorInsufficientDriver,
            Self::NoDevice => cudaError_cudaErrorNoDevice,
            Self::InvalidResourceHandle => cudaError_cudaErrorInvalidResourceHandle,
            Self::Other(x) => x,
        }
    }
}

#[allow(non_upper_case_globals)]
impl CudaError {
    pub(crate) fn from_raw(raw: cudaError) -> Result<()> {
        match raw {
            cudaError_cudaSuccess => Ok(()),
            cudaError_cudaErrorInvalidValue => Err(CudaError::InvalidValue),
            cudaError_cudaErrorMemoryAllocation => Err(CudaError::MemoryAllocation),
            cudaError_cudaErrorInitializationError => Err(CudaError::Initialization),
            cudaError_cudaErrorInsufficientDriver => Err(CudaError::InsufficientDriver),
            cudaError_cudaErrorNoDevice => Err(CudaError::NoDevice),
            cudaError_cudaErrorInvalidResourceHandle => Err(CudaError::InvalidResourceHandle),
            other => Err(CudaError::Other(other)),
        }
    }
}
impl Error for CudaError {}

impl fmt::Display for CudaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}
