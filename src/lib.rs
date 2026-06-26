use buffer::{Buffer, numpy};
use ops::ops::{add_tensors, mul_tensors};
// use ops::select::slice_buffer;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use view::View;

pub mod buffer;
pub mod dtype;
pub mod helpers;
pub mod ops;
pub mod storage;
pub mod view;

#[pymodule]
fn autograd_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Buffer>()?;
    m.add_class::<View>()?;
    m.add_function(wrap_pyfunction!(add_tensors, m)?)?;
    m.add_function(wrap_pyfunction!(mul_tensors, m)?)?;
    m.add_function(wrap_pyfunction!(numpy, m)?)?;
    Ok(())
}
