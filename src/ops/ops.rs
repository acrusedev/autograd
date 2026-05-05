use crate::buffer::Buffer;
use crate::dtype::DType;
use crate::storage::Storage;
use crate::helpers::calc_strides;
use pyo3::exceptions::{PyNotImplementedError, PyValueError};
use pyo3::{PyRef, PyResult, pyfunction};

#[pyfunction]
// add_tensors does not take strides into consideration
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

            fn get_coords(shape: &[isize], index: isize) -> (isize, isize) {
                let cols = shape[1];
                return (index / cols, index % cols);
            }

            unsafe {
                let a_slice = std::slice::from_raw_parts(a.data.as_ptr() as *const i32, numel);
                let b_slice = std::slice::from_raw_parts(b.data.as_ptr() as *const i32, numel);
                let out_slice =
                    std::slice::from_raw_parts_mut(output.as_mut_ptr() as *mut i32, numel);

                for linear_index in 0..numel - 1 {
                    let coords = get_coords(&a.shape, linear_index as isize);
                    let a_offset = a.strides[0] * coords.0 + a.strides[1] * coords.1;
                    let b_offset = b.strides[0] * coords.0 + b.strides[1] * coords.1;
                    let out_strides = calc_strides(&a.shape, &a.dtype);
                    let out_offset = nbytes * linear_index;
                }
            }

            Ok(Buffer {
                data: output,
                shape: a.shape.clone(),
                strides: a.strides.clone(),
                dtype: a.dtype.clone(),
            })
        }
        _ => Err(PyNotImplementedError::new_err(
            "add is currently implemented only for int32 buffers",
        )),
    }
}
