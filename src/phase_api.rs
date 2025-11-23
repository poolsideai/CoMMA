//! Phase notification API for training code
//!
//! This module provides C FFI functions for training code to signal phase transitions
//! to the CoMMA profiler. These APIs enable phase-aware telemetry aggregation.
//!
//! # API Functions
//!
//! - `ncclProfilerBeginPhase(encoded_phase)` - Signal start of training phase
//! - `ncclProfilerEndPhase(encoded_phase)` - Signal end of training phase
//!
//! # Usage Example (C)
//!
//! ```c
//! // Training code signals phase transitions with opaque phase tags
//! // User can encode any information (e.g., step number, phase type)
//! uint64_t phase_tag = (step << 32) | phase_type;
//! ncclProfilerBeginPhase(phase_tag);
//! // ... forward pass NCCL operations ...
//! ncclProfilerEndPhase(phase_tag);
//! ```
//!
//! # Python Usage
//!
//! ```python
//! from nccl_telemetry import encode_phase  # Helper function (optional)
//!
//! # User can encode phase tags however they want
//! phase_tag = encode_phase(step=42, phase=1)  # One suggested encoding
//! ncclProfilerBeginPhase(phase_tag)
//! # ... forward pass ...
//! ncclProfilerEndPhase(phase_tag)
//! ```

use crate::profiler_shim::ncclResult_t;
use std::sync::OnceLock;

/// Global profiler reference for phase API
static PROFILER: OnceLock<&'static crate::profiler::Profiler> = OnceLock::new();

/// Initialize phase API with profiler reference
///
/// This is called internally by profiler init handlers
pub fn init_phase_api(profiler: &'static crate::profiler::Profiler) {
    let _ = PROFILER.set(profiler);
}

/// Begin a training phase
///
/// # Arguments
/// - `encoded_phase` - Opaque phase tag (interpretation is up to the user)
///
/// # Returns
/// - ncclSuccess on success
/// - ncclInternalError if phase tracking not enabled or invalid phase (0 reserved)
///
/// # Safety
/// This function is safe to call from any thread
#[no_mangle]
pub extern "C" fn ncclProfilerBeginPhase(encoded_phase: u64) -> ncclResult_t {
    let profiler = match PROFILER.get() {
        Some(p) => p,
        None => {
            eprintln!("ncclProfilerBeginPhase: Profiler not initialized");
            return crate::profiler_shim::ncclResult_t_ncclInternalError;
        }
    };

    if !profiler.config.enable_phase_scope {
        eprintln!("ncclProfilerBeginPhase: Phase tracking not enabled (set NCCL_PROFILER_ENABLE_PHASE_SCOPE=true)");
        return crate::profiler_shim::ncclResult_t_ncclInternalError;
    }

    let tracker = match &profiler.phase_scope_tracker {
        Some(t) => t,
        None => {
            eprintln!("ncclProfilerBeginPhase: PhaseScope tracker not initialized");
            return crate::profiler_shim::ncclResult_t_ncclInternalError;
        }
    };

    match tracker.begin_phase(encoded_phase) {
        Ok(_) => crate::profiler_shim::ncclResult_t_ncclSuccess,
        Err(e) => {
            eprintln!("ncclProfilerBeginPhase: {}", e);
            crate::profiler_shim::ncclResult_t_ncclInvalidArgument
        }
    }
}

/// End a training phase by marking it CLOSED
///
/// # Arguments
/// - `encoded_phase` - Opaque phase tag (must match begin_phase call)
///
/// # Returns
/// - ncclSuccess on success
/// - ncclInternalError if phase tracking not enabled
///
/// # Safety
/// This function is safe to call from any thread
///
/// This function marks the phase as CLOSED by setting end_time_ns in PhaseMetrics.
/// The daemon will periodically scan for ready phases and export them when:
/// 1. pending_coll == 0 (all NcclOps accounted)
/// 2. OR timeout reached (safety mechanism)
#[no_mangle]
pub extern "C" fn ncclProfilerEndPhase(encoded_phase: u64) -> ncclResult_t {
    let profiler = match PROFILER.get() {
        Some(p) => p,
        None => {
            eprintln!("ncclProfilerEndPhase: Profiler not initialized");
            return crate::profiler_shim::ncclResult_t_ncclInternalError;
        }
    };

    if !profiler.config.enable_phase_scope {
        eprintln!("ncclProfilerEndPhase: Phase tracking not enabled");
        return crate::profiler_shim::ncclResult_t_ncclInternalError;
    }

    if let Some(ref tracker) = profiler.phase_scope_tracker {
        // Capture end timestamp
        let end_ns = std::time::SystemTime::now()
            .duration_since(std::time::SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;

        // Mark phase as CLOSED
        let pending = tracker.mark_phase_closed(encoded_phase, end_ns);

        if profiler.config.debug_phase_tracking {
            let rank_str = match profiler.rank {
                Some(r) => format!("R{}", r),
                None => "R?".to_string(),
            };
            eprintln!("[PHASE {}] end_phase(0x{:x}): Marked CLOSED (pending={})",
                      rank_str, encoded_phase, pending);
        }
    }

    crate::profiler_shim::ncclResult_t_ncclSuccess
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_api_not_initialized() {
        // Test that API returns error when profiler not initialized
        let phase_tag = 100; // Opaque phase tag

        // These should fail gracefully with error messages
        let result = ncclProfilerBeginPhase(phase_tag);
        assert_eq!(result, crate::profiler_shim::ncclResult_t_ncclInternalError);

        let result = ncclProfilerEndPhase(phase_tag);
        assert_eq!(result, crate::profiler_shim::ncclResult_t_ncclInternalError);
    }
}
