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

use crossbeam::utils::CachePadded;

use std::cell::UnsafeCell;
use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

#[derive(Debug)]
pub struct Spsc<T> {
    entries: Vec<UnsafeCell<MaybeUninit<T>>>,

    rd_idx: CachePadded<AtomicU32>,

    // wr_idx is what the sender use as the "write head"
    // wr_idx_pub is updated periodically by the receiver
    // receiver should poll on wr_idx_pub to avoid frequent access to
    // sender's cache line
    wr_idx: CachePadded<AtomicU32>,
    wr_idx_pub: CachePadded<AtomicU32>,
}

unsafe impl<T> Sync for Spsc<T> {}

impl<T> Spsc<T> {
    fn new(sz: usize) -> Self {
        assert!(sz > 0);
        assert_eq!(sz & (sz - 1), 0);
        Self {
            entries: (0..sz)
                .map(|_| UnsafeCell::new(MaybeUninit::uninit()))
                .collect(),
            rd_idx: CachePadded::new(AtomicU32::new(0)),
            wr_idx: CachePadded::new(AtomicU32::new(0)),
            wr_idx_pub: CachePadded::new(AtomicU32::new(0)),
        }
    }

    /// # Safety
    ///
    /// index must be in bound
    #[inline(always)]
    unsafe fn get_entry(&self, idx: u32) -> *mut MaybeUninit<T> {
        self.entries.get_unchecked(idx as usize).get()
    }
}

#[derive(Debug)]
pub struct Sender<T> {
    channel: Arc<Spsc<T>>,
    fifo_len_mask: u32,
    rd_idx_cache: u32,
    wr_idx_cache: u32,
    num_pending_post: usize,
    batch_sz: usize,
}

impl<T> Sender<T> {
    fn new(spsc: Arc<Spsc<T>>) -> Self {
        let sz_mask = spsc.entries.len() - 1;
        Self {
            channel: spsc,
            fifo_len_mask: sz_mask as u32,
            rd_idx_cache: 0,
            wr_idx_cache: 0,
            num_pending_post: 0,
            batch_sz: 1,
        }
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    pub fn prefetch_next(&self) {
        let next_idx = (self.wr_idx_cache + 1) & self.fifo_len_mask;
        unsafe {
            use core::arch::x86_64::{_mm_prefetch, _MM_HINT_ET0};
            let ptr = (*self.channel.get_entry(next_idx)).as_mut_ptr();
            _mm_prefetch(ptr.cast(), _MM_HINT_ET0);
        }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "x86")))]
    pub fn prefetch_next(&self) {}

    pub fn send(&mut self, obj: T, use_barrier: bool) -> Result<(), T> {
        let channel = &self.channel;
        let idx = self.wr_idx_cache;
        let next_idx = (idx + 1) & self.fifo_len_mask;

        if next_idx == self.rd_idx_cache {
            self.rd_idx_cache = self.channel.rd_idx.load(Ordering::Relaxed);
        }

        if next_idx != self.rd_idx_cache {
            unsafe {
                std::ptr::write((*self.channel.get_entry(idx)).as_mut_ptr(), obj);
            }
            self.wr_idx_cache = next_idx;
            if use_barrier {
                channel.wr_idx.store(next_idx, Ordering::Release);
            }
            self.num_pending_post += 1;
            if self.num_pending_post >= self.batch_sz {
                self.flush();
            }
            Ok(())
        } else {
            Err(obj)
        }
    }

    pub fn set_batch(&mut self, batch_sz: usize) {
        self.batch_sz = std::cmp::min(batch_sz, self.channel.entries.len() - 1);
        if self.num_pending_post >= self.batch_sz {
            self.flush();
        }
    }

    #[inline(always)]
    pub fn flush(&mut self) {
        let channel = &self.channel;
        channel.wr_idx.store(self.wr_idx_cache, Ordering::Relaxed);
        channel
            .wr_idx_pub
            .store(self.wr_idx_cache, Ordering::Release);
        self.num_pending_post = 0;
    }
}

impl<T> std::ops::Drop for Sender<T> {
    fn drop(&mut self) {
        self.flush();
    }
}

#[derive(Debug)]
pub struct Receiver<T> {
    channel: Arc<Spsc<T>>,
    wr_idx_cache: u32,
}

impl<T> Receiver<T> {
    fn new(spsc: Arc<Spsc<T>>) -> Self {
        Self {
            channel: spsc,
            wr_idx_cache: 0,
        }
    }

    pub fn recv(&mut self) -> Option<T> {
        let idx = self.channel.rd_idx.load(Ordering::Relaxed);
        if idx == self.wr_idx_cache {
            self.wr_idx_cache = self.channel.wr_idx_pub.load(Ordering::Acquire);
        }
        if idx != self.wr_idx_cache {
            let mut next_idx = idx + 1;
            if next_idx as usize == self.channel.entries.len() {
                next_idx = 0;
            }
            let obj = unsafe { (*self.channel.get_entry(idx)).assume_init_read() };
            self.channel.rd_idx.store(next_idx, Ordering::Release);
            Some(obj)
        } else {
            None
        }
    }

    pub fn try_fetch_from_sender(&mut self) {
        let channel = &self.channel;
        let wr_idx_pub = channel.wr_idx_pub.load(Ordering::Relaxed);
        let wr_idx = channel.wr_idx.load(Ordering::Relaxed);
        let _ = channel.wr_idx_pub.compare_exchange(
            wr_idx_pub,
            wr_idx,
            Ordering::AcqRel,
            Ordering::Relaxed,
        );
    }
}

pub fn channel<T>(sz: usize) -> (Sender<T>, Receiver<T>) {
    let spsc = Arc::new(Spsc::new(sz));
    let sender = Sender::new(spsc.clone());
    let receiver = Receiver::new(spsc);
    (sender, receiver)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_channel_test() {
        const N: usize = 65539;

        let (mut tx, mut rx) = channel::<usize>(16);
        tx.set_batch(16);

        std::thread::scope(move |s| {
            s.spawn(move || {
                for i in 0..N {
                    let mut v = i;
                    while let Err(r) = tx.send(v, true) {
                        v = r;
                    }
                }
                tx.flush();
            });
            s.spawn(move || {
                for i in 0..N {
                    let v = loop {
                        if let Some(v) = rx.recv() {
                            break v;
                        }
                    };
                    assert_eq!(i, v);
                }
            });
        });
    }
}
