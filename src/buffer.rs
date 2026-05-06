use crate::dtype::DType;
use crate::helpers::{calc_strides, get_coords};
use crate::storage::Storage;
use pyo3::ffi::Py_buffer;
use pyo3::prelude::*;
use pyo3::types::PyType;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use std::fmt::{Display, Write};
use std::os::raw::{c_int, c_void};

#[gen_stub_pyclass]
#[pyclass(unsendable)]
#[derive(Debug, Clone)]
#[repr(C)]
pub struct Buffer {
    pub data: Storage,
    pub shape: Vec<isize>,
    pub strides: Vec<isize>,
    pub dtype: DType,
}

#[gen_stub_pymethods]
#[pymethods]
impl Buffer {
    #[new]
    fn new(
        pydata: &Bound<'_, PyAny>,
        shape: Vec<isize>,
        strides: Vec<isize>,
        format: &str,
    ) -> PyResult<Self> {
        let dtype = DType::from_str(format);
        let bytes = pydata.extract::<&[u8]>()?;
        let storage = Storage::from_slice(bytes);

        Ok(Buffer {
            data: storage,
            shape,
            strides,
            dtype,
        })
    }

    #[classmethod]
    fn from_bytes(
        _cls: &Bound<'_, PyType>,
        bytes: Vec<u8>,
        shape: Vec<isize>,
        strides: Vec<isize>,
        dtype: &str,
    ) -> Buffer {
        Buffer {
            data: Storage::from_slice(bytes.as_slice()),
            shape,
            strides,
            dtype: DType::from_str(dtype),
        }
    }

    #[classmethod]
    fn cast_buffer(_cls: &Bound<'_, PyType>, buffer: PyRef<Buffer>, new_dtype: &str) -> Buffer {
        let numel = buffer.shape.iter().map(|n| *n as usize).product::<usize>();
        match buffer.dtype {
            DType::Int32 => unsafe {
                let slice = std::slice::from_raw_parts(buffer.data.as_ptr() as *const i32, numel);
                let itemsize_i32 = DType::get_byte_size(&DType::Int32);
                match DType::from_str(new_dtype) {
                    DType::Float64 => {
                        let itemsize_f64 = DType::get_byte_size(&DType::Float64);
                        let strides_f64 = calc_strides(&buffer.shape, itemsize_f64);
                        let nbytes = numel * std::mem::size_of::<f64>();
                        let mut new_storage = Storage::allocate(nbytes);
                        let output = std::slice::from_raw_parts_mut(
                            new_storage.as_mut_ptr() as *mut f64,
                            numel,
                        );
                        for linear_index in 0..numel {
                            let coords = get_coords(&buffer.shape, linear_index);

                            let mut buffer_sum = 0;
                            let mut output_sum = 0;

                            for i in 0..coords.len() {
                                buffer_sum += buffer.strides[i] * coords[i];
                                output_sum += strides_f64[i] * coords[i];
                            }

                            let buffer_idx = (buffer_sum / itemsize_i32) as usize;
                            let output_idx = (output_sum / itemsize_f64) as usize;
                            let item = slice[buffer_idx] as f64;
                            output[output_idx] = item;
                        }
                        Buffer {
                            data: new_storage,
                            shape: buffer.shape.clone(),
                            strides: strides_f64,
                            dtype: DType::from_str(new_dtype),
                        }
                    }
                    _ => panic!("not implemented yet"),
                }
            },
            _ => panic!("not implemented yet"),
        }
    }
}

pub fn numpy_a<T>(tensor: PyRef<Buffer>, num_cols: usize) -> String
where
    T: Clone + Display,
{
    unsafe {
        let numel = tensor.shape.iter().map(|x| *x as usize).product();
        let tensor_slice = std::slice::from_raw_parts(tensor.data.as_ptr() as *const T, numel);
        let v = tensor_slice.to_vec();
        let mut s = String::from("<Tensor [");
        for (index, element) in v.iter().enumerate() {
            if (index + 1) != numel {
                _ = write!(&mut s, "{}, ", element.to_string());
            } else {
                _ = write!(&mut s, "{}", element.to_string());
            }
            if (index + 1) % num_cols == 0 {
                s.push_str("\n\t");
            }
        }
        s.push_str(&format!("]>, dtype={}", tensor.dtype));
        s
    }
}

#[pyfunction]
pub fn numpy(tensor: PyRef<Buffer>) -> String {
    let num_cols = 20;
    match tensor.dtype {
        DType::Int8 => numpy_a::<i8>(tensor, num_cols),
        DType::Int16 => numpy_a::<i16>(tensor, num_cols),
        DType::Int32 => numpy_a::<i32>(tensor, num_cols),
        DType::Int64 => numpy_a::<i64>(tensor, num_cols),
        DType::Float32 => numpy_a::<f32>(tensor, num_cols),
        DType::Float64 => numpy_a::<f64>(tensor, num_cols),
        _ => return "Not implemented".to_owned(),
    }
}

#[pymethods]
impl Buffer {
    // https://docs.python.org/3/c-api/typeobj.html#c.PyBufferProcs.bf_getbuffer
    // from https://docs.python.org/3/c-api/buffer.html
    unsafe fn __getbuffer__(
        slf: PyRefMut<'_, Self>,
        view: *mut Py_buffer,
        _flags: c_int,
    ) -> PyResult<()> {
        unsafe {
            (*view).buf = slf.data.as_ptr() as *mut c_void;
            (*view).len = slf.data.len() as isize;
            (*view).itemsize = slf.dtype.get_bit_size();
            (*view).readonly = 0; // modifiable on
            (*view).ndim = slf.shape.len() as c_int;
            (*view).shape = slf.shape.as_ptr() as *mut isize;
            (*view).strides = slf.strides.as_ptr() as *mut isize;
            (*view).suboffsets = std::ptr::null_mut();
            (*view).internal = std::ptr::null_mut() as *mut c_void;
        }
        Ok(())
    }
}
