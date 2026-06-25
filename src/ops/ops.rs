use std::fmt::Debug;
use std::ops::{Add, Mul};
use std::rc::Rc;

use crate::buffer::Buffer;
use crate::dtype::DType;
use crate::helpers::{calc_strides, get_coords};
use crate::storage::Storage;
use pyo3::exceptions::{PyNotImplementedError, PyValueError};
use pyo3::{PyRef, PyResult, pyfunction};

fn generic_add_tensors<T>(numel: usize, tensor_a: PyRef<Buffer>, tensor_b: PyRef<Buffer>) -> Buffer
where
    T: Add<Output = T> + Copy + Debug,
{
    let nbytes = numel * std::mem::size_of::<T>();
    let mut output = Storage::allocate(nbytes);
    unsafe {
        let output_slice = std::slice::from_raw_parts_mut(output.as_mut_ptr() as *mut T, numel);
        let itemsize = std::mem::size_of::<T>() as isize;
        let out_strides = calc_strides(&tensor_a.shape, std::mem::size_of::<T>() as isize);
        for linear_index in 0..numel {
            let coords = get_coords(&tensor_a.shape, linear_index);
            let mut a_offset = 0;
            let mut b_offset = 0;
            let mut out_offset = 0;
            for i in 0..coords.len() {
                a_offset += (tensor_a.strides[i] * coords[i]) as usize;
                b_offset += (tensor_b.strides[i] * coords[i]) as usize;
                out_offset += out_strides[i] * coords[i];
            }
            let out_idx = (out_offset / itemsize) as usize;
            output_slice[out_idx] = *(tensor_a
                .data
                .as_ptr()
                .add((tensor_a.offset + a_offset) as usize)
                as *const T)
                + *(tensor_b
                    .data
                    .as_ptr()
                    .add((tensor_b.offset + b_offset) as usize) as *const T);
        }
        Buffer {
            data: Rc::new(output),
            shape: tensor_a.shape.to_owned(),
            strides: out_strides,
            dtype: tensor_a.dtype.to_owned(),
            offset: 0,
        }
    }
}

#[inline(never)]
#[pyfunction]
pub fn add_tensors(a: PyRef<Buffer>, b: PyRef<Buffer>) -> PyResult<Buffer> {
    if a.shape != b.shape {
        return Err(PyValueError::new_err("add requires identical shapes"));
    }

    if a.dtype != b.dtype {
        return Err(PyValueError::new_err("add requires identical dtypes"));
    }

    let numel = a.shape.iter().map(|x| *x as usize).product::<usize>();

    match a.dtype {
        DType::Int8 => Ok(generic_add_tensors::<i8>(numel, a, b)),
        DType::Int16 => Ok(generic_add_tensors::<i16>(numel, a, b)),
        DType::Int32 => Ok(generic_add_tensors::<i32>(numel, a, b)),
        DType::Int64 => Ok(generic_add_tensors::<i64>(numel, a, b)),
        DType::Float32 => Ok(generic_add_tensors::<f32>(numel, a, b)),
        DType::Float64 => Ok(generic_add_tensors::<f64>(numel, a, b)),
        _ => Err(PyNotImplementedError::new_err(
            "add is currently implemented only for int32 buffers",
        )),
    }
}

pub fn generic_mul_tensors<T>(a: PyRef<Buffer>, b: PyRef<Buffer>) -> Buffer
where
    T: Mul<Output = T> + Copy,
{
    let numel = a.shape.iter().map(|x| *x as usize).product::<usize>();
    let nbytes = numel * std::mem::size_of::<T>();
    let mut output = Storage::allocate(nbytes);
    unsafe {
        let output_slice = std::slice::from_raw_parts_mut(output.as_mut_ptr() as *mut T, numel);
        let itemsize = std::mem::size_of::<T>() as isize;
        let out_strides = calc_strides(&a.shape, std::mem::size_of::<T>() as isize);
        for linear_index in 0..numel {
            let coords = get_coords(&a.shape, linear_index);
            let mut a_offset = 0;
            let mut b_offset = 0;
            let mut out_offset = 0;
            for i in 0..coords.len() {
                a_offset += (a.strides[i] * coords[i]) as usize;
                b_offset += (b.strides[i] * coords[i]) as usize;
                out_offset += out_strides[i] * coords[i];
            }
            let out_idx = (out_offset / itemsize) as usize;
            output_slice[out_idx] = *(a.data.as_ptr().add((a.offset + a_offset) as usize)
                as *const T)
                * *(b.data.as_ptr().add((b.offset + b_offset) as usize) as *const T);
        }
        Buffer {
            data: Rc::new(output),
            shape: a.shape.to_owned(),
            strides: out_strides,
            dtype: a.dtype.to_owned(),
            offset: 0,
        }
    }
}

#[pyfunction]
pub fn mul_tensors(a: PyRef<Buffer>, b: PyRef<Buffer>) -> PyResult<Buffer> {
    if a.shape != b.shape {
        return Err(PyValueError::new_err("mul requires identical shapes"));
    }

    if a.dtype != b.dtype {
        return Err(PyValueError::new_err("mul requires identical dtypes"));
    }

    match a.dtype {
        DType::Int8 => Ok(generic_mul_tensors::<i8>(a, b)),
        DType::Int16 => Ok(generic_mul_tensors::<i16>(a, b)),
        DType::Int32 => Ok(generic_mul_tensors::<i32>(a, b)),
        DType::Int64 => Ok(generic_mul_tensors::<i64>(a, b)),
        DType::Float32 => Ok(generic_mul_tensors::<f32>(a, b)),
        DType::Float64 => Ok(generic_mul_tensors::<f64>(a, b)),
        _ => Err(PyNotImplementedError::new_err(
            "add is currently implemented only for int32 buffers",
        )),
    }
}
