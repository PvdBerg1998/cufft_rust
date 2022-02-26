use criterion::{black_box, criterion_group, criterion_main, Criterion};
use cufft_rust::*;
use std::time::{Duration, Instant};

pub fn criterion_benchmark(c: &mut Criterion) {
    // Prepare data
    let y = (0..2u64.pow(20))
        .map(|x| x as f32 / 2.0f32.powi(18) * std::f32::consts::TAU)
        .map(|x| x.cos())
        .collect::<Vec<_>>();

    c.bench_function("fft32 2^20", |b| {
        b.iter_custom(|iters| {
            let mut dt = Duration::default();
            for _i in 0..iters {
                let data = y.as_slice();
                let start = Instant::now();
                let res = black_box(fft32(data));
                dt += start.elapsed();
                drop(res);
            }
            dt
        })
    });

    c.bench_function("fft32 2^20 batch 100", |b| {
        b.iter_custom(|iters| {
            let mut dt = Duration::default();
            for _i in 0..iters {
                let data = &[y.as_slice(); 100];
                let start = Instant::now();
                let res = black_box(fft32_batch(data));
                dt += start.elapsed();
                drop(res);
            }
            dt
        })
    });

    // Prepare data
    let y = (0..2u64.pow(20))
        .map(|x| x as f64 / 2.0f64.powi(18) * std::f64::consts::TAU)
        .map(|x| x.cos())
        .collect::<Vec<_>>();

    c.bench_function("fft64 2^20", |b| {
        b.iter_custom(|iters| {
            let mut dt = Duration::default();
            for _i in 0..iters {
                let data = y.as_slice();
                let start = Instant::now();
                let res = black_box(fft64(data));
                dt += start.elapsed();
                drop(res);
            }
            dt
        })
    });

    c.bench_function("fft64 2^20 batch 100", |b| {
        b.iter_custom(|iters| {
            let mut dt = Duration::default();
            for _i in 0..iters {
                let data = &[y.as_slice(); 100];
                let start = Instant::now();
                let res = black_box(fft64_batch(data));
                dt += start.elapsed();
                drop(res);
            }
            dt
        })
    });

    c.bench_function("fft64 to 32 2^20", |b| {
        b.iter_custom(|iters| {
            let mut dt = Duration::default();
            for _i in 0..iters {
                let data = y.clone();
                let start = Instant::now();
                let data = data.into_iter().map(|x| x as f32).collect::<Vec<_>>();
                let res = black_box(fft32(data.as_slice()));
                dt += start.elapsed();
                drop(data);
                drop(res);
            }
            dt
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
