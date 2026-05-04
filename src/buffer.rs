use crate::dtype::DType;
use crate::storage::Storage;
use core::fmt;
use pyo3::ffi::Py_buffer;
use pyo3::prelude::*;
use pyo3::types::PyType;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use std::fmt::Write;
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

    fn __repr__(&self) -> String {
        let ptr = self.data.as_ptr();
        let data_len = self.shape.iter().product::<isize>() as usize;
        let data_str = unsafe {
            match self.dtype {
                DType::Float64 => {
                    let slice = std::slice::from_raw_parts(ptr as *const f64, data_len);
                    format!("{:?}", &slice[..slice.len().min(5)])
                }
                DType::Float32 => {
                    let slice = std::slice::from_raw_parts(ptr as *const f32, data_len);
                    format!("{:?}", &slice[..slice.len().min(5)])
                }
                DType::Uint8 => {
                    let slice = std::slice::from_raw_parts(ptr as *const u8, data_len);
                    format!("{:?}", &slice[..slice.len().min(5)])
                }
                DType::Int8 => {
                    let slice = std::slice::from_raw_parts(ptr as *const i8, data_len);
                    format!("{:?}", &slice[..slice.len().min(5)])
                }

                DType::Bool => {
                    let slice = std::slice::from_raw_parts(ptr as *const bool, data_len);
                    format!("{:?}", &slice[..slice.len().min(5)])
                }
                DType::Int16 => {
                    let slice = std::slice::from_raw_parts(ptr as *const i16, data_len);
                    format!("{:?}", &slice[..slice.len().min(5)])
                }
                DType::Int32 => {
                    let slice = std::slice::from_raw_parts(ptr as *const i32, data_len);
                    format!("{:?}", &slice[..slice.len().min(5)])
                }
                DType::Int64 => {
                    let slice = std::slice::from_raw_parts(ptr as *const i64, data_len);
                    format!("{:?}", &slice[..slice.len().min(5)])
                } // TODO
                // DType::Bfloat16 => {
                //     let slice = std::slice::from_raw_parts(ptr as *const i8, data_len);
                //     format!("{:?}", &slice[..slice.len().min(5)])
                // }

                // DType::Float16 => {
                //     let slice = std::slice::from_raw_parts(ptr as *const i8, data_len);
                //     format!("{:?}", &slice[..slice.len().min(5)])
                // }
                _ => {
                    format!("this data type is not yet supported")
                }
            }
        };
        data_str
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
}

#[pyfunction]
pub fn numpy(tensor: PyRef<Buffer>) -> String {
    let num_cols = 20;
    match tensor.dtype {
        DType::Int32 => unsafe {
            let numel = tensor.shape.iter().map(|x| *x as usize).product();
            let tensor_slice =
                std::slice::from_raw_parts(tensor.data.as_ptr() as *const i32, numel);
            let v = tensor_slice.to_vec();
            let mut s = String::from("<Tensor [");
            for (index, element) in v.iter().enumerate() {
                if index + 1 != numel {
                    write!(&mut s, "{}, ", element.to_string());
                } else {
                    write!(&mut s, "{}", element.to_string());
                }
                if index + 1 % num_cols == 0 {
                    s.push_str("\n");
                }
            }
            s.push_str(&format!("]>, dtype={}", tensor.dtype));
            s
        },
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
