use itertools::Itertools;
use ndarray::prelude::*;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use numpy::{IntoPyArray, Ix2, PyArray1, PyReadonlyArray};
use paste;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::seq::SliceRandom;
use rayon::prelude::*;
use std::collections::HashMap;
use std::hash::Hash;

fn get_ss_t(sqdistances: &ArrayView2<f64>) -> f64 {
    let mut sum = 0f64;
    for i in 0..sqdistances.shape()[0] {
        for j in 0..sqdistances.shape()[1] {
            if i != j {
                sum += sqdistances[[i, j]];
            }
        }
    }
    return sum / (sqdistances.shape()[0] as f64) / 2.;
}

fn get_ss_w(sqdistances: &ArrayView2<f64>, labels: &Vec<usize>, bincount: &Vec<i64>) -> f64 {
    let mut sums = vec![0f64; bincount.len()];
    for i in 0..sqdistances.shape()[0] {
        for j in 0..sqdistances.shape()[0] {
            if labels[i] == labels[j] && i != j {
                sums[labels[i]] += sqdistances[[i, j]];
            }
        }
    }

    let mut sum = 0f64;
    for (a, b) in std::iter::zip(sums, bincount) {
        sum += a / *b as f64;
    }

    sum / 2.
}

fn get_f(ss_t: f64, ss_w: f64, a: u64, n: u64) -> f64 {
    let ss_a = ss_t - ss_w;
    (ss_a / (a - 1) as f64) / (ss_w / (n - a) as f64)
}

// Vec<usize> is used since we have to consume it due to the shuffling, no use asking for views
pub fn _permanova(
    sqdistances: &ArrayView2<f64>,
    labels: Vec<usize>,
    permutations: usize,
) -> (f64, f64) {
    let max_label = *(labels.iter().max().unwrap());
    let bincount: Vec<i64> = (0..=max_label)
        .map(|el| labels.iter().filter(|&x| *x == el).count() as i64)
        .collect();
    let ss_t = get_ss_t(&sqdistances);
    let ss_w = get_ss_w(&sqdistances, &labels, &bincount);
    let f = get_f(ss_t, ss_w, bincount.len() as u64, labels.len() as u64);

    let other_fs: Vec<_> = (0..permutations)
        .into_par_iter()
        .map_with(labels, |labels, _| {
            labels.shuffle(&mut rand::thread_rng());
            get_f(
                ss_t,
                get_ss_w(&sqdistances, &labels, &bincount),
                bincount.len() as u64,
                labels.len() as u64,
            )
        })
        .collect();

    return (
        f,
        other_fs.iter().filter(|&other_f| *other_f >= f).count() as f64 / other_fs.len() as f64,
    );
}

pub fn generate_data(size: usize, category_count: usize) -> (Array2<f64>, Vec<usize>) {
    let mut distances = Array2::random((size, size), Uniform::new(0., 1.));
    for i in 0..size {
        for j in 0..i {
            let total = distances[[i, j]] + distances[[j, i]];
            distances[[i, j]] = total;
            distances[[j, i]] = total;
        }
    }
    for i in 0..size {
        distances[[i, i]] = 0.;
    }
    let labels = Array1::random(size, Uniform::new(0, category_count));

    (distances, labels.to_vec())
}

#[pyfunction]
pub fn permanova(
    sqdistances: PyReadonlyArray<f64, Ix2>,
    labels: Vec<usize>,
    permutations: Option<usize>,
) -> PyResult<(f64, f64)> {
    if labels.len() <= 2 {
        return Err(PyValueError::new_err(
            "`labels.len()` cant be <=2, permanova cant run on 2 points",
        ));
    }
    if labels.iter().max().unwrap() == &0usize {
        return Err(PyValueError::new_err(
            "`lables` cant have only one category",
        ));
    }
    let shape = sqdistances.shape();
    if labels.len() != shape[0] {
        return Err(PyValueError::new_err(
            "The length of `labels` must equal the size of the square `sqdistances` matrix",
        ));
    }
    if shape[0] != shape[1] {
        return Err(PyValueError::new_err(
            "The `sqdistances` matrix must be square",
        ));
    }
    if !(labels.iter().min().unwrap() == &0usize
        && labels.iter().max().unwrap() == &(labels.iter().unique().collect::<Vec<_>>().len() - 1))
    {
        return Err(PyValueError::new_err("`labels` must be ordinal-encoded"));
    }
    return Ok(_permanova(
        &sqdistances.as_array(),
        labels,
        permutations.unwrap_or(1000),
    ));
}

pub fn ordinal_encoding<T: Eq + Hash + Clone>(labels: Vec<T>) -> Vec<usize> {
    let mut last = 0usize;
    let mut dic = HashMap::<T, usize>::new();

    labels
        .iter()
        .map(|key| match dic.get(key) {
            None => {
                dic.insert(key.clone(), last);
                last += 1;
                last - 1
            }
            Some(x) => *x,
        })
        .collect::<Vec<_>>()
}

/// implement ordinal_encoding for a concrete type, giving it a `$dtype` name in the module.
/// The secret handshake is that the name must be a valid numpy `dtype.name`.
macro_rules! concrete_ordinal_encoding {
    ($typename: ident, $dtype: literal, $pymodule: ident) => {
        paste::item! {
            #[pyfunction]
            fn [<ordinal_encoding_$dtype>]<'py>(py: Python<'py>, labels: Vec<$typename>) -> &'py PyArray1<usize> {
                ordinal_encoding(labels).into_pyarray(py)
            }
            $pymodule.add_function(wrap_pyfunction!([<ordinal_encoding_$dtype>], $pymodule)?)?;
        }
    };
}

#[pymodule]
fn _oxide(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(permanova, m)?)?;
    concrete_ordinal_encoding!(String, "str", m);
    concrete_ordinal_encoding!(i64, "int64", m);
    concrete_ordinal_encoding!(i32, "int32", m);
    concrete_ordinal_encoding!(i16, "int16", m);
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::{_permanova, ordinal_encoding};
    use ndarray::array;

    #[test]
    fn _permanova_statistic_valid() {
        let sqdistances = array![
            [0., 4., 16., 36., 64.],
            [4., 0., 4., 16., 36.],
            [16., 4., 0., 4., 16.],
            [36., 16., 4., 0., 4.],
            [64., 36., 16., 4., 0.]
        ];
        let labels = vec![0, 1, 2, 0, 1];

        let (statistic, _pvalue) = _permanova(&sqdistances.view(), labels, 10);
        assert_eq!(statistic, 0.1111111111111111);
    }
    #[test]
    fn _permanova_pvalue_approx_valid() {
        let sqdistances = array![
            [0., 4., 16., 36., 64.],
            [4., 0., 4., 16., 36.],
            [16., 4., 0., 4., 16.],
            [36., 16., 4., 0., 4.],
            [64., 36., 16., 4., 0.]
        ];
        let labels = vec![0, 1, 2, 0, 1];

        let (_statistic, pvalue) = _permanova(&sqdistances.view(), labels, 10000);
        assert!((0.92..0.95).contains(&pvalue));
    }

    #[test]
    fn ordinal_encoding_valid() {
        let labels = vec!["aa", "bb", "aa", "cc"];
        assert_eq!(ordinal_encoding(labels), vec![0, 1, 0, 2]);
    }
}
