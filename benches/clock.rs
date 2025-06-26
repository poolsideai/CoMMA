// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use criterion::{criterion_group, criterion_main, Criterion};
use nccl_profiler::clock::{CachedClock, TscClock};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("read tsc", |b| {
        b.iter(|| {
            let _ = TscClock::read_tsc();
        })
    });

    let mut clock = TscClock::default();
    c.bench_function("read clock", |b| {
        b.iter(|| {
            let _ = clock.now();
        })
    });

    c.bench_function("std clock", |b| {
        b.iter(|| {
            let _ = std::time::Instant::now();
        })
    });

    c.bench_function("multithread clock", |b| {
        let clock = CachedClock::new(Instant::now());
        let stop = AtomicBool::new(false);
        let barrier = std::sync::Barrier::new(2);
        std::thread::scope(|s| {
            s.spawn(|| {
                barrier.wait();
                while !stop.load(Ordering::Relaxed) {
                    std::thread::sleep(Duration::from_micros(5));
                    clock.update_cache(Instant::now());
                }
            });
            barrier.wait();
            b.iter(|| clock.recent());
            stop.store(true, Ordering::Relaxed);
        });
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
