use pyo3::{Bound, PyAny, PyResult, pyclass, pymethods};
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

#[gen_stub_pyclass]
#[pyclass(unsendable)]
#[derive(Debug, Clone)]
#[repr(C)]
pub struct View {
    #[pyo3(get)]
    pub shape: Vec<isize>,
    #[pyo3(get)]
    pub strides: Vec<isize>,
    #[pyo3(get)]
    pub offset: isize,
}

#[gen_stub_pymethods]
#[pymethods]
impl View {
    #[new]
    fn new(shape: Vec<isize>, strides: Vec<isize>, offset: isize) -> Self {
        View {
            shape,
            strides,
            offset,
        }
    }
}
