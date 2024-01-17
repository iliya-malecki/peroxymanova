use numpy::{PyReadonlyArray, Ix2};
use pyo3::prelude::*;
use peroxymanova_core:: permanova as core_permanova;
// use polars;

#[pyfunction]
fn permanova(sqdistances:PyReadonlyArray<f64,Ix2>, labels: Vec<usize>) -> (f64, f64) {
    return core_permanova(sqdistances, labels)
}

#[pymodule]
#[pyo3(name="_oxide")]
fn module(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(permanova, m)?)?;
    Ok(())
}
