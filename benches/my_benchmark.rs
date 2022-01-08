use criterion::{criterion_group, criterion_main, Criterion};
use cufft_rust::*;

pub fn criterion_benchmark(c: &mut Criterion) {
    // Prepare data
    let y = (0..2u64.pow(20))
        .map(|x| x as f32 / 2.0f32.powi(18) * std::f32::consts::TAU)
        .map(|x| x.cos())
        .collect::<Vec<_>>();
    c.bench_function("fft32 2^20", |b| {
        b.iter_with_large_drop(|| fft32(y.as_ref()))
    });
    c.bench_function("fft32 2^20 batch 100", |b| {
        b.iter_with_large_drop(|| fft32_batch(&[y.as_ref(); 100]))
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
