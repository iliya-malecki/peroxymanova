use std::{hash::Hash, time::Duration};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use peroxymanova::{_permanova, generate_data, ordinal_encoding};
use rand::seq::SliceRandom;

fn benchmark_permanova(c: &mut Criterion) {
    let (data, labels) = black_box(generate_data(1000, 7));
    c.bench_function("standard permanova", |b| {
        b.iter_batched(
            || labels.clone(),
            |labels| _permanova(&data.view(), labels, 1000),
            criterion::BatchSize::LargeInput,
        )
    });
}

fn benchmark_ordinal_encoding<T: Clone + Eq + Hash>(possibilities: Vec<T>, c: &mut Criterion) {
    let labels: Vec<_> = possibilities
        .choose_multiple(&mut rand::thread_rng(), 100)
        .cloned()
        .collect();
    c.bench_function(
        &format!("ordinal encoding for type {}", std::any::type_name::<T>()),
        |b| {
            b.iter_batched(
                || labels.clone(),
                |data| ordinal_encoding(data),
                criterion::BatchSize::LargeInput,
            )
        },
    );
}

pub fn ordinal_encoding_benchmark_group() {
    let mut c = Criterion::default().configure_from_args();
    benchmark_ordinal_encoding(vec![1, 2, 3], &mut c);
    benchmark_ordinal_encoding(vec!["hello", "world", "this is a string"], &mut c);
    benchmark_ordinal_encoding(
        vec![
            "hello".to_string(),
            "world".to_string(),
            "this is a string".to_string(),
        ],
        &mut c,
    );
}

criterion_group! {
    name = permanova_group;
    config = Criterion::default().measurement_time(Duration::from_secs(100));
    targets = benchmark_permanova
}

criterion_main!(permanova_group, ordinal_encoding_benchmark_group);
