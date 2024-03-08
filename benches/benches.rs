use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
// use ndarray::prelude::*;
use peroxymanova::{_permanova, generate_data};

fn criterion_benchmark(c: &mut Criterion) {
    let (data, labels) = black_box(generate_data(1000, 7));
    c.bench_function("standard permanova", |b| {
        b.iter_batched(
            || labels.clone(),
            |labels| _permanova(&data.view(), labels, 1000),
            criterion::BatchSize::SmallInput,
        )
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default().measurement_time(Duration::from_secs(100));
    targets = criterion_benchmark
}
criterion_main!(benches);
