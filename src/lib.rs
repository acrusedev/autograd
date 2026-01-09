use buffer::Buffer;
use pyo3::prelude::*;
use pyo3_stub_gen::define_stub_info_gatherer;

pub mod buffer;

/// A Python module implemented in Rust.
#[pymodule]
fn autograd_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Buffer>()?;
    Ok(())
}

// Generowanie stubów
define_stub_info_gatherer!(stub_info);
