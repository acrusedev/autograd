use pyo3::ffi::Py_buffer;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use std::os::raw::{c_int, c_void};

use crate::dtype::DType;
use crate::storage::Storage;

#[gen_stub_pyclass]
#[pyclass(unsendable)]
pub struct Buffer {
    data: Storage,
    shape: Vec<isize>,
    strides: Vec<isize>,
    dtype: DType,
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
                DType::float64 => {
                    let slice = std::slice::from_raw_parts(ptr as *const f64, data_len);
                    format!("{:?}", &slice[..slice.len().min(5)])
                }
                DType::float32 => {
                    let slice = std::slice::from_raw_parts(ptr as *const f32, data_len);
                    format!("{:?}", &slice[..slice.len().min(5)])
                }
                DType::unsigned8 => {
                    let slice = std::slice::from_raw_parts(ptr as *const u8, data_len);
                    format!("{:?}", &slice[..slice.len().min(5)])
                }
                DType::signed8 => {
                    let slice = std::slice::from_raw_parts(ptr as *const i8, data_len);
                    format!("{:?}", &slice[..slice.len().min(5)])
                }
            }
        };
        format!(
            "PyBuffer(shape={:?}, strides={:?}, format={:?}, data={}...)",
            self.shape, self.strides, self.dtype, data_str
        )
    }

    fn reshape(&mut self, new_shape: Vec<isize>, new_strides: Vec<isize>) {
        self.shape = new_shape;
        self.strides = new_strides;
    }
}

#[pymethods]
impl Buffer {
    // https://docs.python.org/3/c-api/typeobj.html#c.PyBufferProcs.bf_getbuffer
    // from https://docs.python.org/3/c-api/buffer.html
    unsafe fn __getbuffer__(
        slf: PyRefMut<'_, Self>,
        view: *mut Py_buffer,
        flags: c_int,
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
