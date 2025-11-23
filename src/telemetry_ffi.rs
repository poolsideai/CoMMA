//! C FFI interface for TelemetryPool pull API
//!
//! This module provides C-compatible functions for Python (via ctypes) to:
//! - Open TelemetryPool from shared memory
//! - Pull PhaseScopes
//! - Close handle
//!
//! # Python Usage Example
//!
//! ```python
//! import ctypes
//!
//! lib = ctypes.CDLL("libnccl_profiler.so")
//!
//! # Open pool
//! handle = lib.nccl_telemetry_open(b"/nccl_telemetry_${PGID}_${RANK}")
//!
//! # Pull scope
//! scope = CPhaseScope()
//! if lib.nccl_telemetry_pull(handle, ctypes.byref(scope)) == 1:
//!     print(f"Step: {scope.step()}, Duration: {scope.duration_ms()}ms")
//!
//! # Close
//! lib.nccl_telemetry_close(handle)
//! ```

use crate::phase_scope::PhaseScope;
use crate::telemetry_pool::TelemetryPool;
use std::ffi::CStr;

/// C-compatible PhaseScope structure (same as PhaseScope, already #[repr(C)])
///
/// This is exported as-is since PhaseScope is already #[repr(C)]
pub type CPhaseScope = PhaseScope;

/// Opaque handle for TelemetryPool (C pointer)
pub type TelemetryPoolHandle = *mut TelemetryPool;

/// Open TelemetryPool from shared memory (consumer)
///
/// # Safety
/// - `name` must be a valid null-terminated C string
/// - Caller must call `nccl_telemetry_close()` to free resources
///
/// # Returns
/// - Non-null handle on success
/// - Null on failure
#[no_mangle]
pub unsafe extern "C" fn nccl_telemetry_open(name: *const libc::c_char) -> TelemetryPoolHandle {
    if name.is_null() {
        eprintln!("nccl_telemetry_open: null name");
        return std::ptr::null_mut();
    }

    let name_str = match CStr::from_ptr(name).to_str() {
        Ok(s) => s,
        Err(e) => {
            eprintln!("nccl_telemetry_open: invalid UTF-8 in name: {}", e);
            return std::ptr::null_mut();
        }
    };

    match TelemetryPool::open(name_str) {
        Ok(pool) => Box::into_raw(Box::new(pool)),
        Err(e) => {
            eprintln!("nccl_telemetry_open: {}", e);
            std::ptr::null_mut()
        }
    }
}

/// Pull one PhaseScope from TelemetryPool
///
/// # Safety
/// - `handle` must be a valid handle from `nccl_telemetry_open()`
/// - `scope` must be a valid pointer to CPhaseScope
///
/// # Returns
/// - 1 if PhaseScope was pulled (scope populated)
/// - 0 if ring buffer is empty (scope unchanged)
/// - -1 on error
#[no_mangle]
pub unsafe extern "C" fn nccl_telemetry_pull(
    handle: TelemetryPoolHandle,
    scope: *mut CPhaseScope,
) -> libc::c_int {
    if handle.is_null() || scope.is_null() {
        eprintln!("nccl_telemetry_pull: null handle or scope");
        return -1;
    }

    let pool = &*handle;

    match pool.pull() {
        Some(s) => {
            std::ptr::write(scope, s);
            1
        }
        None => 0,
    }
}

/// Get number of PhaseScopes available in buffer
///
/// # Safety
/// - `handle` must be a valid handle from `nccl_telemetry_open()`
///
/// # Returns
/// - Number of PhaseScopes available
/// - -1 on error
#[no_mangle]
pub unsafe extern "C" fn nccl_telemetry_len(handle: TelemetryPoolHandle) -> libc::c_int {
    if handle.is_null() {
        eprintln!("nccl_telemetry_len: null handle");
        return -1;
    }

    let pool = &*handle;
    pool.len() as libc::c_int
}

/// Close TelemetryPool handle and free resources
///
/// # Safety
/// - `handle` must be a valid handle from `nccl_telemetry_open()`
/// - Handle must not be used after calling this function
#[no_mangle]
pub unsafe extern "C" fn nccl_telemetry_close(handle: TelemetryPoolHandle) {
    if !handle.is_null() {
        let _ = Box::from_raw(handle);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;

    #[test]
    fn test_ffi_open_close() {
        let name = CString::new("/nccl_telemetry_ffi_test").unwrap();

        // Create pool first
        let pool = TelemetryPool::create(name.to_str().unwrap(), 16).unwrap();
        drop(pool);

        // Open via FFI
        let handle = unsafe { nccl_telemetry_open(name.as_ptr()) };
        assert!(!handle.is_null());

        // Close via FFI
        unsafe { nccl_telemetry_close(handle) };

        // Cleanup
        unsafe { libc::shm_unlink(name.as_ptr()) };
    }

    #[test]
    fn test_ffi_pull() {
        let name = CString::new("/nccl_telemetry_ffi_pull_test").unwrap();

        // Create and push scope
        let pool = TelemetryPool::create(name.to_str().unwrap(), 16).unwrap();
        let scope = PhaseScope {
            phase: 12345, // Opaque phase tag
            start_time_ns: 1000,
            end_time_ns: 2000,
            e2e_latency_sum_ns: 500,
            net_latency_sum_ns: 400,
            bytes_transferred: 2048,
            transfer_op_count: 10,
            nccl_op_count: 10,
        };
        pool.push(scope).unwrap();
        drop(pool);

        // Open via FFI
        let handle = unsafe { nccl_telemetry_open(name.as_ptr()) };
        assert!(!handle.is_null());

        // Pull via FFI
        let mut pulled = PhaseScope {
            phase: 0,
            start_time_ns: 0,
            end_time_ns: 0,
            e2e_latency_sum_ns: 0,
            net_latency_sum_ns: 0,
            bytes_transferred: 0,
            transfer_op_count: 0,
            nccl_op_count: 0,
        };
        let result = unsafe { nccl_telemetry_pull(handle, &mut pulled) };
        assert_eq!(result, 1);
        assert_eq!(pulled.phase, 12345);
        assert_eq!(pulled.bytes_transferred, 2048);

        // Pull again (empty)
        let result = unsafe { nccl_telemetry_pull(handle, &mut pulled) };
        assert_eq!(result, 0);

        // Close
        unsafe { nccl_telemetry_close(handle) };

        // Cleanup
        unsafe { libc::shm_unlink(name.as_ptr()) };
    }

    #[test]
    fn test_ffi_len() {
        let name = CString::new("/nccl_telemetry_ffi_len_test").unwrap();

        let pool = TelemetryPool::create(name.to_str().unwrap(), 16).unwrap();
        let scope = PhaseScope {
            phase: 100,
            start_time_ns: 0,
            end_time_ns: 0,
            e2e_latency_sum_ns: 0,
            net_latency_sum_ns: 0,
            bytes_transferred: 0,
            transfer_op_count: 0,
            nccl_op_count: 0,
        };
        pool.push(scope).unwrap();
        pool.push(scope).unwrap();
        drop(pool);

        let handle = unsafe { nccl_telemetry_open(name.as_ptr()) };
        assert!(!handle.is_null());

        let len = unsafe { nccl_telemetry_len(handle) };
        assert_eq!(len, 2);

        unsafe { nccl_telemetry_close(handle) };
        unsafe { libc::shm_unlink(name.as_ptr()) };
    }
}
