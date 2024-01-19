use numpy::{Ix2, PyReadonlyArray};
use peroxymanova_core::permanova as core_permanova;
use pyo3::prelude::*;
// use polars;

#[pyfunction]
fn permanova(sqdistances: PyReadonlyArray<f64, Ix2>, labels: Vec<usize>) -> (f64, f64) {
    return core_permanova(&sqdistances.as_array(), labels);
}

#[pymodule]
#[pyo3(name = "_oxide")]
fn module(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(permanova, m)?)?;
    Ok(())
}
