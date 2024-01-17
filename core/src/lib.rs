use rand::seq::SliceRandom;
use numpy::{PyReadonlyArray, Ix2};
// use polars;


pub fn get_ss_t(sqdistances:&PyReadonlyArray<f64,Ix2>) -> f64 {
    let mut sum = 0f64;
    for i in 0..sqdistances.shape()[0] {
        for j in 0..sqdistances.shape()[1] {
            if i != j {
                sum += sqdistances.get([i,j]).unwrap();
            }
        }
    }
    return sum/(sqdistances.shape()[0] as f64)/2.
}




pub fn get_ss_w(
    sqdistances:&PyReadonlyArray<f64,Ix2>,
    labels:&Vec<usize>,
    bincount:&Vec<i64>) -> f64 {
        let mut sums = vec![0f64; bincount.len()];
        for i in 0..sqdistances.shape()[0] {
            for j in 0..sqdistances.shape()[0] {
                if labels[i] == labels[j] && i != j {
                    sums[labels[i]] += sqdistances.get([i,j]).unwrap();
                }
            }
        }

        let mut sum = 0f64;
        for (a, b) in std::iter::zip(sums, bincount) {
            sum += a / *b as f64;
        };

        sum / 2.
}


pub fn get_f(ss_t: f64, ss_w: f64, a: i64, n:i64) -> f64 {
    let ss_a = ss_t - ss_w;
    (ss_a/(a-1) as f64)/(ss_w/(n-a) as f64)
}

pub fn permanova(
    sqdistances:PyReadonlyArray<f64,Ix2>,
    mut labels: Vec<usize>,
) -> (f64, f64) {
    let max_label = *(labels.iter().max().unwrap());
    let bincount: Vec<i64> = (0..=max_label)
        .into_iter()
        .map(
            |el|labels.iter().filter(|&x| *x==el).count() as i64
        )
        .collect();
    let ss_t = get_ss_t(&sqdistances);
    let ss_w = get_ss_w(&sqdistances, &labels, &bincount);
    let f = get_f(ss_t, ss_w, bincount.len() as i64, labels.len() as i64);

    let mut other_fs: Vec<f64> = Vec::new();
    let mut rng = rand::thread_rng();
    for _ in 0..1000 {
        labels.shuffle(&mut rng);
        other_fs.push(
            get_f(
                ss_t,
                get_ss_w(&sqdistances, &labels, &bincount),
                bincount.len() as i64,
                labels.len() as i64
            )
        );

    }

    return (
        f,
        other_fs
            .iter()
            .filter(|&other_f| *other_f >= f)
            .count() as f64
        / other_fs.len() as f64
    )
}
