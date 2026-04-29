use buffer::Buffer;
use ops::ops::add;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3_stub_gen::define_stub_info_gatherer;

pub mod buffer;
pub mod dtype;
pub mod ops;
pub mod storage;

#[pymodule]
fn autograd_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Buffer>()?;
    m.add_function(wrap_pyfunction!(add, m)?)?;
    define_stub_info_gatherer!(stub_info);
    Ok(())
}
