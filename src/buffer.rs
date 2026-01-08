use pyo3::prelude::*;
use std::cell::UnsafeCell;

pub enum BufferDataType {
    U8(Vec<i8>),
    F32(Vec<f32>),
}

#[pyclass(unsendable)]
pub struct PyBuffer {
    // from https://docs.python.org/3/c-api/buffer.html
    data: UnsafeCell<BufferDataType>,
    // readonly: i8,          // 0 for readonly, 1 for mutable
    item_size: usize, // item size in bytes of a single element
    // ndim: usize,           // number of dimensions the memory represents as an n-dimensional array
    strides: Vec<isize>, // an array of length ndim giving the number of bytes to skip to get to the new element in each dimension
    shape: Vec<isize>, // an array of item_size of length ndim indicating the shape of the memory as an n-dimensional array
    format: &'static [u8], // "B\0" lub "f\0"
}

#[pymethods]
impl PyBuffer {
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
            "b" => {
                let vec: Vec<i8> = py_data.extract()?;
                (BufferDataType::U8(vec), 1, b"b\0".as_slice())
            }
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "unsupported format",
                ));
            }
        };

        Ok(PyBuffer {
            data: UnsafeCell::new(data),
            shape,
            strides,
            item_size,
            format: fmt,
        })
    }
}
