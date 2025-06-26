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
use crate::daemon::*;
use crate::profiler::Profiler;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};

#[derive(Debug)]
pub struct XProfDaemon {
    worker: Option<JoinHandle<()>>,
    stop_signal: Arc<AtomicBool>,
}

impl Daemon for XProfDaemon {
    fn new(profiler: &'static Profiler) -> Self {
        let stop_signal = Arc::new(AtomicBool::new(false));
        let stop_signal_clone = stop_signal.clone();
        let worker = thread::spawn(move || {
            let _ = main_loop(profiler, stop_signal_clone);
        });
        Self {
            worker: Some(worker),
            stop_signal: stop_signal,
        }
    }
}

impl std::ops::Drop for XProfDaemon {
    fn drop(&mut self) {
        self.stop_signal.store(true, Ordering::Release);
        if let Some(worker) = self.worker.take() {
            worker.join().unwrap();
        }
    }
}
struct XProf {}
impl Export for XProf {
    fn export(&self, _ctx: &mut PollingContext, _maybe_retry_ms: Option<u64>) {}
}

fn main_loop(profiler: &'static Profiler, stop: Arc<AtomicBool>) -> std::io::Result<()> {
    let xprof = XProf {};
    let stop_clone = stop.clone();
    let mut memory_poller = PollingContext::new(profiler, stop_clone);
    polling_loop(&mut memory_poller, xprof);
    Ok(())
}
