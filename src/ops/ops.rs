use core::f64;

use crate::buffer::Buffer;
use crate::dtype::DType;
use crate::helpers::{calc_strides, get_coords};
use crate::storage::Storage;
use pyo3::exceptions::{PyNotImplementedError, PyValueError};
use pyo3::{PyRef, PyResult, pyfunction};

#[pyfunction]
pub fn add_tensors(a: PyRef<Buffer>, b: PyRef<Buffer>) -> PyResult<Buffer> {
    if a.shape != b.shape {
        return Err(PyValueError::new_err("add requires identical shapes"));
    }

    if a.dtype != b.dtype {
        return Err(PyValueError::new_err("add requires identical dtypes"));
    }

    let numel = a.shape.iter().map(|n| *n as usize).product::<usize>();

    match a.dtype {
        DType::Int32 => {
            let nbytes = numel * std::mem::size_of::<i32>();
            if a.data.len() != nbytes || b.data.len() != nbytes {
                return Err(PyValueError::new_err(
                    "buffer byte length does not match shape and dtype",
                ));
            }

            let mut output = Storage::allocate(nbytes);

            unsafe {
                let a_slice = std::slice::from_raw_parts(a.data.as_ptr() as *const i32, numel);
                let b_slice = std::slice::from_raw_parts(b.data.as_ptr() as *const i32, numel);
                let out_slice =
                    std::slice::from_raw_parts_mut(output.as_mut_ptr() as *mut i32, numel);

                let itemsize = std::mem::size_of::<i32>() as isize;
                let out_strides = calc_strides(&a.shape, DType::get_byte_size(&DType::Int32));

                for linear_index in 0..numel {
                    let coords = get_coords(&a.shape, linear_index);

                    let mut a_sum = 0;
                    let mut b_sum = 0;
                    let mut out_sum = 0;
                    for i in 0..coords.len() {
                        a_sum += a.strides[i] * coords[i];
                        b_sum += b.strides[i] * coords[i];
                        out_sum += out_strides[i] * coords[i];
                    }
                    let a_idx = (a_sum / itemsize) as usize;
                    let b_idx = (b_sum / itemsize) as usize;
                    let out_idx = (out_sum / itemsize) as usize;
                    out_slice[out_idx] = a_slice[a_idx] + b_slice[b_idx];
                }

                Ok(Buffer {
                    data: output,
                    shape: a.shape.to_owned(),
                    strides: out_strides,
                    dtype: a.dtype.to_owned(),
                })
            }
        }
        DType::Float64 => {
            let nbytes = numel * std::mem::size_of::<f64>();
            if a.data.len() != nbytes || b.data.len() != nbytes {
                return Err(PyValueError::new_err(
                    "buffer byte length does not match shape and dtype",
                ));
            }

            let mut output = Storage::allocate(nbytes);

            unsafe {
                let a_slice = std::slice::from_raw_parts(a.data.as_ptr() as *const f64, numel);
                let b_slice = std::slice::from_raw_parts(b.data.as_ptr() as *const f64, numel);
                let out_slice =
                    std::slice::from_raw_parts_mut(output.as_mut_ptr() as *mut f64, numel);

                let itemsize = std::mem::size_of::<f64>() as isize;
                let out_strides = calc_strides(&a.shape, DType::get_byte_size(&DType::Float64));

                for linear_index in 0..numel {
                    let coords = get_coords(&a.shape, linear_index);

                    let mut a_sum = 0;
                    let mut b_sum = 0;
                    let mut out_sum = 0;
                    for i in 0..coords.len() {
                        a_sum += a.strides[i] * coords[i];
                        b_sum += b.strides[i] * coords[i];
                        out_sum += out_strides[i] * coords[i];
                    }
                    let a_idx = (a_sum / itemsize) as usize;
                    let b_idx = (b_sum / itemsize) as usize;
                    let out_idx = (out_sum / itemsize) as usize;
                    out_slice[out_idx] = a_slice[a_idx] + b_slice[b_idx];
                }

                Ok(Buffer {
                    data: output,
                    shape: a.shape.to_owned(),
                    strides: out_strides,
                    dtype: a.dtype.to_owned(),
                })
            }
        }
        _ => Err(PyNotImplementedError::new_err(
            "add is currently implemented only for int32 buffers",
        )),
    }
}
