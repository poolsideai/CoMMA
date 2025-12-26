//! Phase-aware telemetry aggregation for NCCL operations
//!
//! # Design
//!
//! - Phase field is an opaque u64 value - interpretation is up to the user
//! - PhaseScope tracks all NCCL events with the same phase tag
//! - Finalized scopes include start/end timestamps and aggregated metrics
//! - TelemetryPool provides pull API for training code to export metrics
//!
//! # Phase Tag
//!
//! The phase field is an opaque u64 value. CoMMA does not interpret its meaning -
//! users can encode any information they need (e.g., step number, phase type, job ID, etc.).
//!
//! # Usage
//!
//! ```rust,no_run
//! use nccl_profiler::phase_scope::PhaseScopeTracker;
//!
//! let tracker = PhaseScopeTracker::new();
//!
//! // User defines their own phase encoding (example: step in upper 32 bits)
//! let phase_tag = ((42u64) << 32) | 1;  // e.g., step=42, phase=1
//! tracker.begin_phase(phase_tag).unwrap();
//!
//! // ... NCCL operations happen ...
//!
//! // End phase via FFI: ncclProfilerEndPhase() calls mark_phase_closed()
//! // Daemon exports via finalize_and_remove_phase() when ready
//! ```

use crate::event;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::SystemTime;

// Per-phase metrics accumulation

/// Per-phase accumulated metrics
///
/// NcclOp-based accounting only, no EventStep accounting
/// Reference counting for correct PhaseScope attribution
/// - start_time_ns: Immutable start timestamp
/// - end_time_ns: Phase state (0 = OPEN, non-zero = CLOSED)
/// - pending_coll: Reference counter (incremented on NcclOp creation, decremented on account_ncclop)
//  - all fields are u64
#[derive(Debug)]
pub struct PhaseMetrics {
    /// Phase start timestamp (nanoseconds since UNIX epoch, immutable)
    pub start_time_ns: u64,

    /// Phase end timestamp (nanoseconds since UNIX epoch)
    /// State marker: 0 = OPEN (phase active), non-zero = CLOSED (endPhase called)
    pub end_time_ns: AtomicU64,

    /// Pending NcclOp reference counter
    /// Incremented: when NcclOp is created with this phase tag
    /// Decremented: when account_ncclop() completes for this phase
    /// Ready to export when: pending_coll == 0 AND end_time_ns > 0
    pub pending_coll: AtomicU64,

    /// Total end-to-end latency (network + FIFO wait) in nanoseconds
    pub e2e_latency_sum_ns: AtomicU64,

    /// Network-only latency (excludes FIFO wait) in nanoseconds
    pub net_latency_sum_ns: AtomicU64,

    /// Total bytes transferred
    pub bytes_transferred: AtomicU64,

    /// Number of low-level transfer operations (chunks)
    pub transfer_op_count: AtomicU64,

    /// Number of NCCL collective operations
    pub nccl_op_count: AtomicU64,
}

impl PhaseMetrics {
    pub fn new(start_time_ns: u64) -> Self {
        Self {
            start_time_ns,
            end_time_ns: AtomicU64::new(0), // OPEN state
            pending_coll: AtomicU64::new(0),
            e2e_latency_sum_ns: AtomicU64::new(0),
            net_latency_sum_ns: AtomicU64::new(0),
            bytes_transferred: AtomicU64::new(0),
            transfer_op_count: AtomicU64::new(0),
            nccl_op_count: AtomicU64::new(0),
        }
    }

    /// Reset all metrics and return snapshot
    pub fn reset(&self) -> PhaseMetricsSnapshot {
        PhaseMetricsSnapshot {
            e2e_latency_sum_ns: self.e2e_latency_sum_ns.swap(0, Ordering::AcqRel),
            net_latency_sum_ns: self.net_latency_sum_ns.swap(0, Ordering::AcqRel),
            bytes_transferred: self.bytes_transferred.swap(0, Ordering::AcqRel),
            transfer_op_count: self.transfer_op_count.swap(0, Ordering::AcqRel),
            nccl_op_count: self.nccl_op_count.swap(0, Ordering::AcqRel),
        }
    }

    /// Read current metrics without resetting
    pub fn read(&self) -> PhaseMetricsSnapshot {
        PhaseMetricsSnapshot {
            e2e_latency_sum_ns: self.e2e_latency_sum_ns.load(Ordering::Acquire),
            net_latency_sum_ns: self.net_latency_sum_ns.load(Ordering::Acquire),
            bytes_transferred: self.bytes_transferred.load(Ordering::Acquire),
            transfer_op_count: self.transfer_op_count.load(Ordering::Acquire),
            nccl_op_count: self.nccl_op_count.load(Ordering::Acquire),
        }
    }
}

/// Snapshot of phase metrics at a point in time
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct PhaseMetricsSnapshot {
    pub e2e_latency_sum_ns: u64,
    pub net_latency_sum_ns: u64,
    pub bytes_transferred: u64,
    pub transfer_op_count: u64,
    pub nccl_op_count: u64,
}

impl PhaseMetricsSnapshot {
    pub fn avg_e2e_latency_ns(&self) -> u64 {
        if self.nccl_op_count == 0 {
            0
        } else {
            self.e2e_latency_sum_ns / self.nccl_op_count
        }
    }

    pub fn avg_net_latency_ns(&self) -> u64 {
        if self.nccl_op_count == 0 {
            0
        } else {
            self.net_latency_sum_ns / self.nccl_op_count
        }
    }

    pub fn avg_chunk_size(&self) -> u64 {
        if self.nccl_op_count == 0 {
            0
        } else {
            self.bytes_transferred / self.nccl_op_count
        }
    }
}

// PhaseScope: Finalized phase with timestamps

/// Finalized PhaseScope with start/end timestamps and aggregated metrics
///
/// This structure represents a completed phase with all NCCL operations aggregated.
/// It includes timestamps to measure phase duration.
///
/// # C FFI Compatibility
///
/// This structure is #[repr(C)] for shared memory export via TelemetryPool.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct PhaseScope {
    /// Opaque phase tag (interpretation is up to the user)
    pub phase: u64,

    /// Phase start timestamp (nanoseconds since epoch)
    pub start_time_ns: u64,

    /// Phase end timestamp (nanoseconds since epoch)
    pub end_time_ns: u64,

    /// Total end-to-end latency sum (nanoseconds)
    pub e2e_latency_sum_ns: u64,

    /// Network-only latency sum (nanoseconds)
    pub net_latency_sum_ns: u64,

    /// Total bytes transferred
    pub bytes_transferred: u64,

    /// Number of low-level transfer operations (chunks)
    pub transfer_op_count: u64,

    /// Number of NCCL collective operations
    pub nccl_op_count: u64,
}

impl PhaseScope {
    /// Get phase duration in nanoseconds
    pub fn duration_ns(&self) -> u64 {
        self.end_time_ns.saturating_sub(self.start_time_ns)
    }

    /// Get phase duration in milliseconds
    pub fn duration_ms(&self) -> f64 {
        self.duration_ns() as f64 / 1_000_000.0
    }

    /// Get average latency per operation (microseconds)
    pub fn avg_latency_us(&self) -> f64 {
        if self.nccl_op_count == 0 {
            0.0
        } else {
            (self.e2e_latency_sum_ns as f64 / self.nccl_op_count as f64) / 1000.0
        }
    }

    /// Get bandwidth in GB/s
    pub fn bandwidth_gbps(&self) -> f64 {
        let duration_s = self.duration_ns() as f64 / 1_000_000_000.0;
        if duration_s > 0.0 {
            (self.bytes_transferred as f64 / duration_s) / (1024.0_f64.powi(3))
        } else {
            0.0
        }
    }
}

// PhaseScope tracker with timestamp capture

/// Global phase scope tracker with timestamp capture
///
/// This tracker captures metrics for the currently active phase and provides
/// PhaseScope snapshots when phases end. The phase field is treated as opaque u64.
#[derive(Debug)]
pub struct PhaseScopeTracker {
    /// Currently active phase (opaque u64 value)
    /// 0 = no active phase
    current_phase: AtomicU64,

    /// Start timestamp for active phase (nanoseconds since epoch)
    /// Only valid when current_phase != 0
    start_time_ns: AtomicU64,

    /// Metrics accumulation for all active phases (key = phase tag)
    pub phase_metrics: RwLock<HashMap<u64, Arc<PhaseMetrics>>>,

    /// NCCL rank for debug logging (None if rank not yet known)
    rank: Option<i32>,
}

impl PhaseScopeTracker {
    pub fn new() -> Self {
        Self {
            current_phase: AtomicU64::new(0),
            start_time_ns: AtomicU64::new(0),
            phase_metrics: RwLock::new(HashMap::new()),
            rank: None,
        }
    }

    pub fn new_with_rank(rank: Option<i32>) -> Self {
        Self {
            current_phase: AtomicU64::new(0),
            start_time_ns: AtomicU64::new(0),
            phase_metrics: RwLock::new(HashMap::new()),
            rank,
        }
    }

    /// Lookup metrics for a phase without creating (returns None if not found)
    fn try_get_metrics_for_phase(&self, phase: u64) -> Option<Arc<PhaseMetrics>> {
        let map = self.phase_metrics.read().unwrap();
        map.get(&phase).cloned()
    }

    /// Begin a new phase
    ///
    /// # Arguments
    /// * `phase` - Opaque phase tag, no assumption about encoding
    ///
    /// # Returns
    /// - Ok(()) on success
    /// - Err if phase is 0 (reserved for "no active phase")
    pub fn begin_phase(&self, phase: u64) -> Result<(), &'static str> {
        if phase == 0 {
            return Err("Phase tag 0 is reserved (no active phase)");
        }

        // Capture start timestamp (nanoseconds since UNIX epoch)
        let start_ns = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;

        // Create metrics entry for this phase under exclusive lock
        {
            let mut map = self.phase_metrics.write().unwrap();
            if map.contains_key(&phase) {
                if let Some(r) = self.rank {
                    eprintln!("[WARN R{}] Phase 0x{:x} collision: reused before export", r, phase);
                }
            }
            map.insert(phase, Arc::new(PhaseMetrics::new(start_ns)));
        }

        // Store with Release ordering to ensure all previous writes are visible
        self.current_phase.store(phase, Ordering::Release);
        self.start_time_ns.store(start_ns, Ordering::Release);

        Ok(())
    }

    /// Get current active encoded phase (0 if none)
    pub fn current_phase(&self) -> u64 {
        self.current_phase.load(Ordering::Acquire)
    }

    /// Get start timestamp of current phase (0 if none)
    pub fn start_time_ns(&self) -> u64 {
        self.start_time_ns.load(Ordering::Acquire)
    }

    /// Increment pending NcclOp counter for a phase
    ///
    /// Called when NcclOp is created with this phase tag.
    /// Must be balanced by account_ncclop() decrement!
    pub fn increment_pending(&self, phase: u64) {
        if let Some(metrics) = self.try_get_metrics_for_phase(phase) {
            metrics.pending_coll.fetch_add(1, Ordering::Relaxed);
        }
        // If phase not found, it was already exported - skip silently
    }

    /// Called by ncclProfilerEndPhase() to transition phase from OPEN to CLOSED.
    /// Returns the current pending_coll value for debug logging (0 if phase not found).
    pub fn mark_phase_closed(&self, phase: u64, end_time_ns: u64) -> u64 {
        match self.try_get_metrics_for_phase(phase) {
            Some(metrics) => {
                metrics.end_time_ns.store(end_time_ns, Ordering::Release);
                metrics.pending_coll.load(Ordering::Acquire)
            }
            None => 0, // Phase already exported by timeout
        }
    }

    /// Account for a completed NcclOp by aggregating all child ProxyOp and EventStep metrics
    ///
    /// This implements hierarchical accounting where NcclOp phase tag determines which
    /// PhaseMetrics to update. All ProxyOp and EventStep metrics are summed locally,
    /// then applied as a single batched atomic update.
    ///
    pub fn account_ncclop(&self, op: &event::NcclOp) {
        let phase = op.phase();

        // Skip if no phase tracking
        if phase == 0 {
            return;
        }

        // Local accumulators (no atomics in loop)
        let mut total_e2e_latency_ns = 0u64;
        let mut total_net_latency_ns = 0u64;
        let mut total_bytes = 0u64;
        let mut total_steps = 0u64;

        // Sum all ProxyOp metrics
        if let Some(proxyops) = op.proxyops.as_ref() {
            for proxyop in proxyops {
                if let Some(steps) = proxyop.steps.as_ref() {
                    for step in steps {
                        let e2e_ns = step.dur_ns as u64 + step.fifo_wait_dur_ns.unwrap_or(0) as u64;
                        let net_ns = step.dur_ns as u64;

                        total_e2e_latency_ns += e2e_ns;
                        total_net_latency_ns += net_ns;
                        total_bytes += step.size as u64;
                        total_steps += 1;
                    }
                }
            }
        }

        // Lookup metrics for this phase (don't create if already exported)
        let metrics = match self.try_get_metrics_for_phase(phase) {
            Some(m) => m,
            None => {
                // Phase was already exported by timeout - drop late operation
                return;
            }
        };

        metrics.e2e_latency_sum_ns.fetch_add(total_e2e_latency_ns, Ordering::Relaxed);
        metrics.net_latency_sum_ns.fetch_add(total_net_latency_ns, Ordering::Relaxed);
        metrics.bytes_transferred.fetch_add(total_bytes, Ordering::Relaxed);
        metrics.transfer_op_count.fetch_add(total_steps, Ordering::Relaxed);
        metrics.nccl_op_count.fetch_add(1, Ordering::Relaxed);

        // This balances the increment from NcclOp creation, see increment_pending() users
        let prev = metrics.pending_coll.fetch_sub(1, Ordering::Release);

        // Sanity check: detect underflow
        if prev == 0 {
            if let Some(r) = self.rank {
                eprintln!("[WARN R{}] Phase 0x{:x}: pending_coll underflow in account_ncclop!", r, phase);
            }
        }
    }

    /// Finalize phase and remove from HashMap
    ///
    /// Called by daemon when phase is ready to export:
    /// - pending_coll == 0 (all NcclOps accounted)
    /// - end_time_ns > 0 (phase CLOSED)
    ///
    /// Returns None if phase not found or still OPEN
    pub fn finalize_and_remove_phase(&self, phase: u64) -> Option<PhaseScope> {
        let mut map = self.phase_metrics.write().unwrap();

        let metrics = map.remove(&phase)?;
        let end_time_ns = metrics.end_time_ns.load(Ordering::Acquire);

        // Sanity check: should be CLOSED
        if end_time_ns == 0 {
            eprintln!("[ERROR] Attempted to finalize OPEN phase 0x{:x}", phase);
            return None;
        }

        // Snapshot final metrics
        Some(PhaseScope {
            phase,
            start_time_ns: metrics.start_time_ns,
            end_time_ns,
            e2e_latency_sum_ns: metrics.e2e_latency_sum_ns.load(Ordering::Relaxed),
            net_latency_sum_ns: metrics.net_latency_sum_ns.load(Ordering::Relaxed),
            bytes_transferred: metrics.bytes_transferred.load(Ordering::Relaxed),
            transfer_op_count: metrics.transfer_op_count.load(Ordering::Relaxed),
            nccl_op_count: metrics.nccl_op_count.load(Ordering::Relaxed),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phase_scope_duration() {
        let scope = PhaseScope {
            phase: 100, // Opaque phase tag
            start_time_ns: 1_000_000_000, // 1 second
            end_time_ns: 1_500_000_000,   // 1.5 seconds
            e2e_latency_sum_ns: 0,
            net_latency_sum_ns: 0,
            bytes_transferred: 0,
            transfer_op_count: 0,
            nccl_op_count: 0,
        };

        assert_eq!(scope.duration_ns(), 500_000_000); // 500ms
        assert_eq!(scope.duration_ms(), 500.0);
    }

    #[test]
    fn test_tracker_begin_phase() {
        let tracker = PhaseScopeTracker::new();

        assert_eq!(tracker.current_phase(), 0);

        let phase = 100; // Opaque phase tag
        tracker.begin_phase(phase).unwrap();
        assert_eq!(tracker.current_phase(), phase);

        // Phase 0 is reserved
        assert!(tracker.begin_phase(0).is_err());
    }

    #[test]
    fn test_scope_bandwidth_calculation() {
        let scope = PhaseScope {
            phase: 100,
            start_time_ns: 0,
            end_time_ns: 1_000_000_000,         // 1 second
            e2e_latency_sum_ns: 0,
            net_latency_sum_ns: 0,
            bytes_transferred: 1024 * 1024 * 1024, // 1 GiB
            transfer_op_count: 1,
            nccl_op_count: 1,
        };

        let bw = scope.bandwidth_gbps();
        assert!((bw - 1.0).abs() < 0.01); // ~1 GB/s
    }
}
