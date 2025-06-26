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

// 1D and 2D histogram implementation based on the algorithm in
// https://observablehq.com/@iopsystems/h2histogram

use serde_json::json;

use std::sync::atomic::{AtomicPtr, Ordering};
use std::sync::Arc;

const HISTOGRAM_ERR_EXP: u8 = 5;
const HISTOGRAM_MAX_EXP: u8 = 60;

// algorithm: https://observablehq.com/@iopsystems/h2histogram
#[derive(Debug, Clone)]
struct HistogramConfig {
    max: u64,
    grouping_exp: u8,
    max_exp: u8,
    exact_buckets: usize,
    group_buckets: usize,
    _buckets_per_group: usize,
}

impl HistogramConfig {
    pub fn new(p: u8, n: u8) -> Self {
        let max = if n >= 64 { u64::MAX } else { 2u64.pow(n as _) };
        let buckets_per_group = 2usize.pow(p as _);
        let exact_buckets: usize = 2usize.pow((p + 1) as _);
        let group_buckets: usize = (n - p - 1) as usize * buckets_per_group;
        Self {
            max,
            grouping_exp: p,
            max_exp: n,
            exact_buckets,
            group_buckets,
            _buckets_per_group: buckets_per_group,
        }
    }

    pub fn num_buckets(&self) -> usize {
        self.exact_buckets + self.group_buckets + 1
    }

    /// return the bucket index for a given value
    pub fn index_of_val(&self, val: u64) -> usize {
        if val < self.exact_buckets as u64 {
            return val as _;
        }

        if val > self.max {
            return self.num_buckets() - 1;
        }

        let h = u64::BITS - 1 - val.leading_zeros();
        let w: u32 = h - self.grouping_exp as u32;
        (w + 1) as usize * 2usize.pow(self.grouping_exp as _) + ((val - 2u64.pow(h)) >> w) as usize
    }

    /// return the value range that the given bucket contains
    pub fn index_range(&self, idx: usize) -> std::ops::RangeInclusive<u64> {
        if idx < self.exact_buckets {
            return (idx as u64)..=(idx as u64);
        }
        let g = idx as u64 >> self.grouping_exp;
        let h = idx as u64 - g * (1 << self.grouping_exp);
        let width = 1 << (g - 1);
        let lower = (1 << (self.grouping_exp as u64 + g - 1)) + (1 << (g - 1)) * h;
        let upper = if idx == self.num_buckets() - 1 {
            u64::MAX
        } else {
            lower + width - 1
        };
        lower..=upper
    }
}

pub struct Histogram1D {
    inner: histogram::AtomicHistogram,
}

impl Histogram1D {
    pub fn new(p: u8, n: u8) -> Self {
        Self {
            inner: histogram::AtomicHistogram::new(p, n).unwrap(),
        }
    }

    pub fn load(&self) -> histogram::Histogram {
        self.inner.load()
    }

    #[cfg(test)]
    pub fn drain(&self) -> histogram::Histogram {
        self.inner.drain()
    }

    pub fn add(&self, value: u64, count: u64) {
        self.inner.add(value, count).unwrap();
    }

    pub fn to_json(&self) -> serde_json::Value {
        let snapshot = self.load();
        let mut entries = Vec::new();
        for bucket in snapshot.into_iter() {
            if bucket.count() > 0 {
                entries.push(json!({
                    "bucket": {
                        "start": bucket.range().start(),
                        "end": bucket.range().end(),
                    },
                    "count": bucket.count(),
                }));
            }
        }
        json!(entries)
    }
}

impl std::default::Default for Histogram1D {
    fn default() -> Self {
        Self::new(HISTOGRAM_ERR_EXP, HISTOGRAM_MAX_EXP)
    }
}

impl std::fmt::Debug for Histogram1D {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        f.debug_struct("Histogram1D")
            .field("inner", &"<histogram>")
            .finish()
    }
}

// histogram that follows https://observablehq.com/@iopsystems/h2histogram
#[derive(Debug)]
pub struct Histogram2D {
    config: HistogramConfig,
    outer_buckets: Vec<AtomicPtr<Histogram1D>>,
}

impl Histogram2D {
    pub fn new(p: u8, n: u8) -> Self {
        let config = HistogramConfig::new(p, n);
        let buckets = (0..config.num_buckets())
            .map(|_| AtomicPtr::new(std::ptr::null_mut()))
            .collect();
        Self {
            config,
            outer_buckets: buckets,
        }
    }

    fn get_or_create(&self, val: u64) -> *const Histogram1D {
        let idx = self.config.index_of_val(val);
        let atomic_ptr = &self.outer_buckets[idx];
        loop {
            let ptr = atomic_ptr.load(Ordering::Acquire);
            if ptr.is_null() {
                let inner = Arc::new(Histogram1D::new(
                    self.config.grouping_exp,
                    self.config.max_exp,
                ));
                let raw = Arc::into_raw(inner);
                match atomic_ptr.compare_exchange(
                    std::ptr::null_mut(),
                    raw as _,
                    Ordering::AcqRel,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => {
                        return raw;
                    }
                    Err(_) => {
                        let _ = unsafe { Arc::from_raw(raw) };
                        continue;
                    }
                }
            }
            break ptr;
        }
    }

    pub fn get(&self, val: u64) -> &Histogram1D {
        let ptr = self.get_or_create(val);
        unsafe { &*ptr }
    }

    pub fn _get_arc(&self, val: u64) -> Arc<Histogram1D> {
        let ptr = self.get_or_create(val);
        let arc = unsafe { Arc::from_raw(ptr) };
        let ret = arc.clone();
        let _ = Arc::into_raw(arc);
        ret
    }

    pub fn add(&self, value: (u64, u64), count: u64) {
        self.get(value.0).add(value.1, count);
    }

    pub fn buckets(
        &self,
    ) -> impl std::iter::Iterator<Item = (std::ops::RangeInclusive<u64>, &Histogram1D)> {
        self.outer_buckets
            .iter()
            .enumerate()
            .flat_map(|(idx, ptr)| {
                let ptr = ptr.load(Ordering::Acquire);
                if !ptr.is_null() {
                    Some((self.config.index_range(idx), unsafe { &*ptr }))
                } else {
                    None
                }
            })
    }

    pub fn to_json(&self) -> serde_json::Value {
        let mut entries = Vec::new();
        for (range, inner) in self.buckets() {
            entries.push(json!({
                "bucket": {
                    "start": range.start(),
                    "end": range.end(),
                },
                "inner": inner.to_json(),
            }));
        }
        json!(entries)
    }
}

impl std::default::Default for Histogram2D {
    fn default() -> Self {
        Self::new(HISTOGRAM_ERR_EXP, HISTOGRAM_MAX_EXP)
    }
}

impl std::ops::Drop for Histogram2D {
    fn drop(&mut self) {
        for ptr in self.outer_buckets.iter() {
            let ptr = ptr.load(Ordering::Acquire);
            if !ptr.is_null() {
                let _val = unsafe { Arc::from_raw(ptr) };
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use rand::prelude::*;

    #[test]
    fn bucket_correct() {
        let p = 5;
        let n = 32;
        let config = HistogramConfig::new(p, n);
        for i in 0..config.num_buckets() {
            let r = config.index_range(i);
            assert_eq!(config.index_of_val(*r.start()), i);
            assert_eq!(config.index_of_val(*r.end()), i);
        }
    }

    #[test]
    fn histogram_1d_lossless() {
        const N: usize = 32768;
        const N_SAMPLE: usize = 10;
        let p = 5;
        let n = 32;

        let histogram = Histogram1D::new(p, n);
        let barrier = std::sync::Barrier::new(2);

        std::thread::scope(|s| {
            s.spawn(|| {
                let mut rng = rand::rng();
                let cnt_between_sample = N / N_SAMPLE;
                for cnt in 0..N {
                    histogram.add(rng.random::<u32>() as u64, 1);
                    if (cnt + 1) % cnt_between_sample == 0 {
                        barrier.wait();
                    }
                }
                barrier.wait();
            });

            s.spawn(|| {
                let mut total_cnt = 0;
                for _ in 0..N_SAMPLE {
                    barrier.wait();
                    let snapshot = histogram.drain();
                    for bucket in snapshot.into_iter() {
                        total_cnt += bucket.count();
                    }
                }
                barrier.wait();
                let snapshot = histogram.load();
                for bucket in snapshot.into_iter() {
                    total_cnt += bucket.count();
                }
                assert_eq!(total_cnt as usize, N);
            });
        });
    }

    #[test]
    fn histogram_2d_lossless() {
        const N: usize = 32768;
        const N_SAMPLE: usize = 10;
        let p = 5;
        let n = 32;

        let histogram = Histogram2D::new(p, n);
        let barrier = std::sync::Barrier::new(2);

        std::thread::scope(|s| {
            s.spawn(|| {
                let mut rng = rand::rng();
                let cnt_between_sample = N / N_SAMPLE;
                for cnt in 0..N {
                    let outer = rng.random::<u32>() as u64;
                    let inner_histogram = histogram.get(outer);
                    inner_histogram.add(rng.random::<u32>() as u64, 1);
                    if (cnt + 1) % cnt_between_sample == 0 {
                        barrier.wait();
                    }
                }
                barrier.wait();
            });

            s.spawn(|| {
                let mut total_cnt = 0;
                for _ in 0..N_SAMPLE {
                    barrier.wait();
                    for (_, inner) in histogram.buckets() {
                        let snapshot = inner.drain();
                        for bucket in snapshot.into_iter() {
                            total_cnt += bucket.count();
                        }
                    }
                }
                barrier.wait();
                for (_, inner) in histogram.buckets() {
                    let snapshot = inner.load();
                    for bucket in snapshot.into_iter() {
                        total_cnt += bucket.count();
                    }
                }
                assert_eq!(total_cnt as usize, N);
            });
        });
    }
}
