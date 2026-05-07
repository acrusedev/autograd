use std::fmt::Debug;
use std::ops::Add;
use std::process::Output;

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
    println!("nbytes = {}", nbytes);
    let mut output = Storage::allocate(nbytes);
    unsafe {
        let tensor_a_slice = std::slice::from_raw_parts(tensor_a.data.as_ptr() as *const T, numel);
        let tensor_b_slice = std::slice::from_raw_parts(tensor_b.data.as_ptr() as *const T, numel);
        let output_slice = std::slice::from_raw_parts_mut(output.as_mut_ptr() as *mut T, numel);

        println!("tensor_a slice {:?}", tensor_a_slice);

        let itemsize = std::mem::size_of::<T>() as isize;
        println!("itemsize = {}", itemsize);
        let out_strides = calc_strides(&tensor_a.shape, std::mem::size_of::<T>() as isize);
        println!("out_strides = {:?}", out_strides);

        for linear_index in 0..numel {
            let coords = get_coords(&tensor_a.shape, linear_index);

            let mut a_offset = 0;
            let mut b_offset = 0;
            let mut out_offset = 0;
            assert!(
                coords.len() == tensor_a.strides.len() && coords.len() == tensor_b.strides.len(),
            );
            for i in 0..coords.len() {
                a_offset += tensor_a.strides[i] * coords[i];
                b_offset += tensor_b.strides[i] * coords[i];
                out_offset += out_strides[i] * coords[i];
            }
            println!(
                "a_sum={}, b_sum={}, out_sum={}",
                a_offset, b_offset, out_offset
            );
            let a_idx = (a_offset / itemsize) as usize;
            let b_idx = (b_offset / itemsize) as usize;
            let out_idx = (out_offset / itemsize) as usize;
            output_slice[out_idx] = tensor_a_slice[a_idx] + tensor_b_slice[b_idx];
        }

        Buffer {
            data: output,
            shape: tensor_a.shape.to_owned(),
            strides: out_strides,
            dtype: tensor_a.dtype.to_owned(),
        }
    }
}

#[pyfunction]
pub fn add_tensors(a: PyRef<Buffer>, b: PyRef<Buffer>) -> PyResult<Buffer> {
    if a.shape != b.shape {
        return Err(PyValueError::new_err("add requires identical shapes"));
    }

    if a.dtype != b.dtype {
        return Err(PyValueError::new_err("add requires identical dtypes"));
    }

    println!("{:?} {:?}", a.dtype, b.dtype);

    let numel = a.shape.iter().map(|n| *n as usize).product::<usize>();

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
