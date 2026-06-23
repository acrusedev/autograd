use crate::buffer::Buffer;
use pyo3::{PyRef, pyfunction};

#[pyfunction]
pub fn select_buffer_element(buffer: PyRef<Buffer>, key: usize) -> Buffer {
    let element_offset = buffer.offset + buffer.strides[0] as usize * key;

    Buffer {
        data: buffer.data.clone(),
        shape: vec![],
        strides: vec![],
        dtype: buffer.dtype.clone(),
        offset: element_offset,
    }
}

#[pyfunction]
pub fn slice_buffer(buffer: PyRef<Buffer>) -> Buffer {}
