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

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

const SAMPLE_INTERVAL: u64 = 100000000;

#[derive(Debug)]
pub struct TscClock {
    last_instant: Instant,
    last_sample_tsc: u64,
    last_ns_per_tick: f64,
}

impl std::default::Default for TscClock {
    fn default() -> Self {
        let t0 = Instant::now();
        let tsc0 = Self::read_tsc();
        std::thread::sleep(Duration::from_millis(10));
        let t1 = Instant::now();
        let tsc1 = Self::read_tsc();
        let dt_ns = (t1 - t0).as_nanos() as u64;
        let d_tsc = tsc1 - tsc0;
        let ns_per_tick = dt_ns as f64 / d_tsc as f64;

        Self {
            last_instant: t1,
            last_sample_tsc: tsc1,
            last_ns_per_tick: ns_per_tick,
        }
    }
}

impl TscClock {
    pub fn now(&mut self) -> Instant {
        let tsc = Self::read_tsc();
        let d_tsc = tsc - self.last_sample_tsc;
        if d_tsc < SAMPLE_INTERVAL {
            self.last_instant + Duration::from_nanos((d_tsc as f64 * self.last_ns_per_tick) as _)
        } else {
            let t = Instant::now();
            let tsc = Self::read_tsc();
            let d_tsc = tsc - self.last_sample_tsc;
            let dt = (t - self.last_instant).as_nanos() as u64;
            self.last_sample_tsc = tsc;
            self.last_ns_per_tick = dt as f64 / d_tsc as f64;
            self.last_instant = t;
            t
        }
    }

    #[inline(always)]
    #[cfg(target_arch = "x86")]
    pub fn read_tsc() -> u64 {
        unsafe { core::arch::x86::_rdtsc() }
    }

    #[inline(always)]
    #[cfg(target_arch = "x86_64")]
    pub fn read_tsc() -> u64 {
        unsafe { core::arch::x86_64::_rdtsc() }
    }

    #[inline(always)]
    #[cfg(target_arch = "aarch64")]
    pub fn read_tsc() -> u64 {
        let value: u64;
        unsafe {
            core::arch::asm!("mrs {}, CNTPCT_EL0", out(reg) value);
        }
        value
    }
}

#[derive(Debug)]
pub struct CachedClock {
    start_time: Instant,
    cached_ns: AtomicU64,
}

impl CachedClock {
    pub fn new(start_time: Instant) -> Self {
        Self {
            start_time,
            cached_ns: AtomicU64::new(start_time.elapsed().as_nanos() as _),
        }
    }

    pub fn update_cache(&self, time: Instant) {
        let dt = time - self.start_time;
        self.cached_ns.store(dt.as_nanos() as _, Ordering::Relaxed);
    }

    pub fn recent_ns(&self) -> u64 {
        self.cached_ns.load(Ordering::Relaxed)
    }

    pub fn recent(&self) -> Instant {
        self.start_time + Duration::from_nanos(self.recent_ns())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tsc_clock_accuracy() {
        let mut clock = TscClock::default();
        let dur = Duration::from_micros(50);
        let t = Instant::now();
        std::thread::sleep(dur);
        let c = clock.now();
        let dur = t.elapsed();
        let dt = c - t;
        let diff_ns = dt.abs_diff(dur).as_nanos();
        assert!(
            (diff_ns as f64) / (dt.as_nanos() as f64) < 0.2,
            "{:?} vs {:?}",
            dt,
            dur
        );

        let mut clock = TscClock::default();
        let dur = Duration::from_millis(200);
        let t = Instant::now();
        std::thread::sleep(dur);
        let c = clock.now();
        let dur = t.elapsed();
        let dt = c - t;
        let diff_ns = dt.abs_diff(dur).as_nanos();
        assert!(
            (diff_ns as f64) / (dt.as_nanos() as f64) < 0.2,
            "{:?} vs {:?}",
            dt,
            dur
        );
    }
}
