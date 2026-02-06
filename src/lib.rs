use buffer::Buffer;
use pyo3::prelude::*;
use pyo3_stub_gen::define_stub_info_gatherer;

pub mod buffer;
pub mod dtype;
pub mod storage;

#[pymodule]
fn autograd_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Buffer>()?;
    define_stub_info_gatherer!(stub_info);
    Ok(())
}
