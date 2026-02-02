use pyo3::ffi::Py_buffer;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use pyo3::{PyRefMut, PyResult};
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use std::cell::UnsafeCell;
use std::os::raw::{c_char, c_int, c_void};

pub enum BufferDataType {
    U8(Vec<u8>),
    I8(Vec<i8>),
    F32(Vec<f32>),
}

#[gen_stub_pyclass]
#[pyclass(unsendable)]
pub struct Buffer {
    // from https://docs.python.org/3/c-api/buffer.html
    data: UnsafeCell<BufferDataType>,
    item_size: usize,      // item size in bytes of a single element
    strides: Vec<isize>, // an array of length ndim giving the number of bytes to skip to get to the new element in each dimension
    shape: Vec<isize>, // an array of item_size of length ndim indicating the shape of the memory as an n-dimensional array
    format: &'static [u8], // "B\0" or "f\0"
}

#[gen_stub_pymethods]
#[pymethods]
impl Buffer {
    #[new]
    fn new(
        py_data: &Bound<'_, PyAny>,
        shape: Vec<isize>,
        strides: Vec<isize>,
        format: &str,
    ) -> PyResult<Self> {
        let (data, item_size, fmt) = match format {
            "f" => {
                let vec: Vec<f32> = py_data.extract()?;
                (BufferDataType::F32(vec), 4, b"f\0".as_slice())
            }
            "B" => {
                if let Ok(bytes) = py_data.cast::<PyBytes>() {
                    let vec = bytes.as_bytes().to_vec();
                    (BufferDataType::U8(vec), 1, b"B\0".as_slice())
                } else {
                    let vec: Vec<u8> = py_data.extract()?;
                    (BufferDataType::U8(vec), 1, b"B\0".as_slice())
                }
            }
            "b" => {
                if let Ok(_) = py_data.cast::<PyBytes>() {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "unsupported data format, bytes cant be of dtype i8",
                    ));
                } else {
                    let vec: Vec<i8> = py_data.extract()?;
                    (BufferDataType::I8(vec), 1, b"b\0".as_slice())
                }
            }
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "unsupported data format, expected f32, i8 or u8",
                ));
            }
        };

        Ok(Buffer {
            data: UnsafeCell::new(data),
            shape,
            strides,
            item_size,
            format: fmt,
        })
    }

    fn __repr__(&self) -> String {
        let data_to_print = unsafe {
            match &*self.data.get() {
                BufferDataType::I8(v) => format!("{:?}", &v[..v.len().min(10)]),
                BufferDataType::U8(v) => format!("{:?}", &v[..v.len().min(10)]),
                BufferDataType::F32(v) => format!("{:?}", &v[..v.len().min(10)]),
            }
        };
        format!(
            "PyBuffer(shape={:?}, strides={:?}, format={:?}, data={}...)",
            self.shape,
            self.strides,
            std::str::from_utf8(self.format).unwrap_or("?"),
            data_to_print
        )
    }

    fn reshape(&mut self, new_shape: Vec<isize>, new_strides: Vec<isize>) {
        self.shape = new_shape;
        self.strides = new_strides;
    }
}

#[pymethods]
impl Buffer {
    unsafe fn __getbuffer__(
        slf: PyRefMut<'_, Self>,
        view: *mut Py_buffer,
        flags: c_int,
    ) -> PyResult<()> {
        let (data_pointer, data_length) = unsafe {
            match &*slf.data.get() {
                BufferDataType::U8(v) => (v.as_ptr() as *mut c_void, v.len()),
                BufferDataType::I8(v) => (v.as_ptr() as *mut c_void, v.len()),
                BufferDataType::F32(v) => (v.as_ptr() as *mut c_void, v.len()),
            }
        };

        unsafe {
            (*view).buf = data_pointer;
            (*view).len = (data_length * slf.item_size) as isize;
            (*view).itemsize = slf.item_size as isize;
            (*view).readonly = 0;
            (*view).ndim = slf.shape.len() as c_int;
            (*view).format = slf.format.as_ptr() as *mut c_char;
            (*view).shape = slf.shape.as_ptr() as *mut isize;
            (*view).strides = slf.strides.as_ptr() as *mut isize;
            (*view).suboffsets = std::ptr::null_mut();
            (*view).internal = std::ptr::null_mut();
        }

        Ok(())
    }
}
