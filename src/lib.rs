use numpy::{Ix2, PyReadonlyArray};
use pyo3::prelude::*;
// use polars;
use ndarray::prelude::*;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use rand::seq::SliceRandom;

// use polars;

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

pub fn _permanova(sqdistances: &ArrayView2<f64>, mut labels: Vec<usize>) -> (f64, f64) {
    let max_label = *(labels.iter().max().unwrap());
    let bincount: Vec<i64> = (0..=max_label)
        .into_iter()
        .map(|el| labels.iter().filter(|&x| *x == el).count() as i64)
        .collect();
    let ss_t = get_ss_t(&sqdistances);
    let ss_w = get_ss_w(&sqdistances, &labels, &bincount);
    let f = get_f(ss_t, ss_w, bincount.len() as u64, labels.len() as u64);

    let mut other_fs: Vec<f64> = Vec::new();
    let mut rng = rand::thread_rng();
    for _ in 0..1000 {
        labels.shuffle(&mut rng);
        other_fs.push(get_f(
            ss_t,
            get_ss_w(&sqdistances, &labels, &bincount),
            bincount.len() as u64,
            labels.len() as u64,
        ));
    }

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
            distances[[i, j]] = total;
        }
    }
    for i in 0..size {
        distances[[i, i]] = 0.;
    }
    let labels = Array1::random(size, Uniform::new(0, category_count));

    (distances, labels.to_vec())
}


#[pyfunction]
pub fn permanova(sqdistances: PyReadonlyArray<f64, Ix2>, labels: Vec<usize>) -> (f64, f64) {
    return _permanova(&sqdistances.as_array(), labels);
}

#[pymodule]
fn _oxide(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(permanova, m)?)?;
    Ok(())
}
