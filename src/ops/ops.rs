use std::fmt::Debug;
use std::ops::{Add, Mul};
use std::rc::Rc;

use crate::buffer::Buffer;
use crate::dtype::DType;
use crate::helpers::{calc_strides, get_coords};
use crate::storage::Storage;
use pyo3::exceptions::{PyNotImplementedError, PyValueError};
use pyo3::types::PyAnyMethods;
use pyo3::{Bound, PyAny, PyRef, PyResult, pyfunction};

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
            "mul is not implemented for this dtype",
        )),
    }
}

fn generic_reduce_sum<T>(buf: PyRef<Buffer>, axis: usize) -> Buffer
where
    T: Add<Output = T> + Copy + Default,
{
    let shape = &buf.shape;
    let out_shape: Vec<isize> = shape
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != axis)
        .map(|(_, &d)| d)
        .collect();
    let out_numel = if out_shape.is_empty() {
        1
    } else {
        out_shape.iter().map(|x| *x as usize).product()
    };
    let reduce_len = shape[axis] as usize;
    let itemsize = std::mem::size_of::<T>() as isize;
    let out_strides = calc_strides(&out_shape, itemsize);
    let nbytes = out_numel * std::mem::size_of::<T>();
    let mut output = Storage::allocate(nbytes);
    unsafe {
        let output_slice = std::slice::from_raw_parts_mut(output.as_mut_ptr() as *mut T, out_numel);
        for out_idx in 0..out_numel {
            let out_coords = if out_shape.is_empty() {
                Vec::new()
            } else {
                get_coords(&out_shape, out_idx)
            };
            let mut acc = T::default();
            for r in 0..reduce_len {
                let mut full_coords = Vec::with_capacity(shape.len());
                let mut oi = 0;
                for i in 0..shape.len() {
                    if i == axis {
                        full_coords.push(r as isize);
                    } else {
                        full_coords.push(out_coords[oi]);
                        oi += 1;
                    }
                }
                let mut byte_off: usize = 0;
                for i in 0..full_coords.len() {
                    byte_off += (buf.strides[i] * full_coords[i]) as usize;
                }
                acc = acc + *(buf.data.as_ptr().add(buf.offset + byte_off) as *const T);
            }
            output_slice[out_idx] = acc;
        }
        Buffer {
            data: Rc::new(output),
            shape: out_shape,
            strides: out_strides,
            dtype: buf.dtype.to_owned(),
            offset: 0,
        }
    }
}

#[pyfunction]
pub fn reduce_sum(a: PyRef<Buffer>, axis: isize) -> PyResult<Buffer> {
    let ndim = a.shape.len() as isize;
    if ndim == 0 {
        return Err(PyValueError::new_err("cannot reduce a 0-d tensor"));
    }
    let ax = if axis < 0 { axis + ndim } else { axis };
    if ax < 0 || ax >= ndim {
        return Err(PyValueError::new_err(format!(
            "axis {axis} is out of bounds for array of dimension {ndim}"
        )));
    }
    let axis_u = ax as usize;
    match a.dtype {
        DType::Int8 => Ok(generic_reduce_sum::<i8>(a, axis_u)),
        DType::Int16 => Ok(generic_reduce_sum::<i16>(a, axis_u)),
        DType::Int32 => Ok(generic_reduce_sum::<i32>(a, axis_u)),
        DType::Int64 => Ok(generic_reduce_sum::<i64>(a, axis_u)),
        DType::Float32 => Ok(generic_reduce_sum::<f32>(a, axis_u)),
        DType::Float64 => Ok(generic_reduce_sum::<f64>(a, axis_u)),
        _ => Err(PyNotImplementedError::new_err(
            "reduce_sum is not implemented for this dtype",
        )),
    }
}

/// Typed kernel: `T` must be known at compile time (no runtime "any" scalar here).
fn max_generic<T>(buffer: PyRef<Buffer>, value: T) -> Buffer
where
    T: Copy + PartialOrd,
{
    let numel = if buffer.shape.is_empty() {
        1
    } else {
        buffer.shape.iter().map(|x| *x as usize).product::<usize>()
    };
    let itemsize = std::mem::size_of::<T>() as isize;
    let out_strides = calc_strides(&buffer.shape, itemsize);
    let nbytes = numel * std::mem::size_of::<T>();
    let mut output = Storage::allocate(nbytes);
    unsafe {
        let output_slice = std::slice::from_raw_parts_mut(output.as_mut_ptr() as *mut T, numel);
        for linear_index in 0..numel {
            let coords = get_coords(&buffer.shape, linear_index);
            let mut byte_off: usize = 0;
            for i in 0..coords.len() {
                byte_off += (buffer.strides[i] * coords[i]) as usize;
            }
            let x = *(buffer.data.as_ptr().add(buffer.offset + byte_off) as *const T);
            // PartialOrd: NaN comparisons are false → NaN becomes `value`
            output_slice[linear_index] = if x > value { x } else { value };
        }
        Buffer {
            data: Rc::new(output),
            shape: buffer.shape.to_owned(),
            strides: out_strides,
            dtype: buffer.dtype.to_owned(),
            offset: 0,
        }
    }
}

/// Dynamic boundary: accept any Python number, coerce via f64, then cast to buffer dtype.
fn py_scalar_as_f64(value: &Bound<'_, PyAny>) -> PyResult<f64> {
    if let Ok(v) = value.extract::<f64>() {
        return Ok(v);
    }
    if let Ok(v) = value.extract::<i64>() {
        return Ok(v as f64);
    }
    if let Ok(v) = value.extract::<bool>() {
        return Ok(if v { 1.0 } else { 0.0 });
    }
    Err(PyValueError::new_err(
        "max value must be a number (int, float, or bool)",
    ))
}

/// `max(buffer, value)` — `value` is any Python scalar (int/float/bool).
#[pyfunction]
pub fn max(buffer: PyRef<Buffer>, value: &Bound<'_, PyAny>) -> PyResult<Buffer> {
    let v = py_scalar_as_f64(value)?;
    match buffer.dtype {
        DType::Int8 => Ok(max_generic::<i8>(buffer, v as i8)),
        DType::Int16 => Ok(max_generic::<i16>(buffer, v as i16)),
        DType::Int32 => Ok(max_generic::<i32>(buffer, v as i32)),
        DType::Int64 => Ok(max_generic::<i64>(buffer, v as i64)),
        DType::Float32 => Ok(max_generic::<f32>(buffer, v as f32)),
        DType::Float64 => Ok(max_generic::<f64>(buffer, v)),
        _ => Err(PyNotImplementedError::new_err(
            "max is not implemented for this dtype",
        )),
    }
}
