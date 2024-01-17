use criterion::{black_box, criterion_group, criterion_main, Criterion};

use peroxymanova_core::permanova;

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("fib 20", |b| b.iter(|| 2 + 2));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
