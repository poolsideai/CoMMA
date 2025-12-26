//! Shared memory TelemetryPool for pull-based PhaseScope export
//!
//! This module implements a lock-free SPSC (Single Producer Single Consumer) ring buffer
//! in shared memory for exporting PhaseScopes from CoMMA daemon to training code.
//!
//! # Architecture
//!
//! ```text
//! [CoMMA Daemon] --push--> [Shared Memory Ring Buffer] <--pull-- [Training Code]
//!   (Producer)                   (TelemetryPool)                    (Consumer)
//! ```
//!
//! - **Producer**: CoMMA daemon pushes completed PhaseScopes
//! - **Consumer**: Training code pulls PhaseScopes and exports to OTel/CloudWatch
//! - **Lock-free**: Atomic head/tail pointers for zero-contention access
//!
//! # Usage
//!
//! ## Producer (CoMMA daemon)
//!
//! ```rust,no_run
//! use nccl_profiler::telemetry_pool::TelemetryPool;
//! use nccl_profiler::phase_scope::PhaseScope;
//!
//! let pool = TelemetryPool::create("/nccl_telemetry_0", 1024).unwrap();
//!
//! let scope = PhaseScope {
//!     phase: 1,
//!     start_time_ns: 0,
//!     end_time_ns: 1000,
//!     e2e_latency_sum_ns: 0,
//!     net_latency_sum_ns: 0,
//!     bytes_transferred: 0,
//!     transfer_op_count: 0,
//!     nccl_op_count: 0,
//! };
//! pool.push(scope).unwrap();
//! ```
//!
//! ## Consumer (Training code via Python FFI)
//!
//! ```python
//! from nccl_telemetry import NCCLTelemetryClient
//!
//! client = NCCLTelemetryClient()
//! for scope in client.pull_all():
//!     print(f"Step {scope.step}, Phase {scope.phase.name}: {scope.duration_ms}ms")
//! ```

use crate::phase_scope::PhaseScope;
use std::sync::atomic::{AtomicU64, Ordering};

/// Magic number for TelemetryPool header validation
const TELEMETRY_POOL_MAGIC: u64 = 0x4E43434C54454C50; // "NCCLTLP"

/// Shared memory ring buffer for PhaseScope export
///
/// # Memory Layout
///
/// ```text
/// +-------------------+
/// | Header (64 bytes) |  <- Magic, capacity, head, tail
/// +-------------------+
/// | PhaseScope[0]     |  <- Ring buffer entries
/// | PhaseScope[1]     |
/// | ...               |
/// | PhaseScope[N-1]   |
/// +-------------------+
/// ```
///
/// # Lock-free SPSC Protocol
///
/// - **head**: Write position (producer increments)
/// - **tail**: Read position (consumer increments)
/// - **Empty**: head == tail
/// - **Full**: (head + 1) % capacity == tail
#[repr(C)]
struct TelemetryPoolHeader {
    /// Magic number for validation (TELEMETRY_POOL_MAGIC)
    magic: u64,

    /// Ring buffer capacity (power of 2 for fast modulo)
    capacity: u64,

    /// Write position (producer owns)
    head: AtomicU64,

    /// Read position (consumer owns)
    tail: AtomicU64,

    /// Padding to 64 bytes
    _padding: [u64; 4],
}

/// TelemetryPool handle for producer (CoMMA daemon)
#[derive(Debug)]
pub struct TelemetryPool {
    /// Shared memory file descriptor
    shm_fd: i32,

    /// Mapped memory region
    ptr: *mut u8,

    /// Total mapped size (header + ring buffer)
    size: usize,

    /// Header pointer
    header: *mut TelemetryPoolHeader,

    /// Ring buffer pointer
    ring: *mut PhaseScope,

    /// Ring buffer capacity (cached from header)
    capacity: u64,
}

unsafe impl Send for TelemetryPool {}
unsafe impl Sync for TelemetryPool {}

impl TelemetryPool {
    /// Create or open TelemetryPool in shared memory
    ///
    /// # Arguments
    /// * `name` - Shared memory name (e.g., "/nccl_telemetry_0")
    /// * `capacity` - Ring buffer capacity (power of 2 recommended)
    ///
    /// # Returns
    /// - Ok(TelemetryPool) on success
    /// - Err on shared memory failure
    pub fn create(name: &str, capacity: u64) -> Result<Self, String> {
        if capacity == 0 || capacity > (1 << 30) {
            return Err("capacity must be 0 < capacity <= 2^30".to_string());
        }

        // Calculate total size with PAGE_SIZE alignment
        let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize };
        let header_size = std::mem::size_of::<TelemetryPoolHeader>();
        let ring_size = capacity as usize * std::mem::size_of::<PhaseScope>();
        let total_size = (header_size + ring_size + page_size - 1) & !(page_size - 1);

        // Create shared memory (POSIX shm_open)
        let cname = std::ffi::CString::new(name).unwrap();
        let shm_fd = unsafe {
            libc::shm_open(
                cname.as_ptr(),
                libc::O_CREAT | libc::O_RDWR,
                0o600,
            )
        };

        if shm_fd < 0 {
            return Err(format!("shm_open failed: {}", std::io::Error::last_os_error()));
        }

        // Resize shared memory
        if unsafe { libc::ftruncate(shm_fd, total_size as i64) } < 0 {
            unsafe { libc::close(shm_fd) };
            return Err(format!("ftruncate failed: {}", std::io::Error::last_os_error()));
        }

        // Memory map
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                total_size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                shm_fd,
                0,
            )
        };

        if ptr == libc::MAP_FAILED {
            unsafe { libc::close(shm_fd) };
            return Err(format!("mmap failed: {}", std::io::Error::last_os_error()));
        }

        let header = ptr as *mut TelemetryPoolHeader;
        let ring = unsafe { ptr.add(header_size) } as *mut PhaseScope;

        // Initialize header (check if already initialized)
        unsafe {
            if (*header).magic != TELEMETRY_POOL_MAGIC {
                // First initialization
                std::ptr::write(
                    header,
                    TelemetryPoolHeader {
                        magic: TELEMETRY_POOL_MAGIC,
                        capacity,
                        head: AtomicU64::new(0),
                        tail: AtomicU64::new(0),
                        _padding: [0; 4],
                    },
                );
            }
        }

        Ok(Self {
            shm_fd,
            ptr: ptr as *mut u8,
            size: total_size,
            header,
            ring,
            capacity,
        })
    }

    /// Open existing TelemetryPool for consumer
    ///
    /// # Arguments
    /// * `name` - Shared memory name
    ///
    /// # Returns
    /// - Ok(TelemetryPool) on success
    /// - Err if shared memory doesn't exist or invalid
    pub fn open(name: &str) -> Result<Self, String> {
        let cname = std::ffi::CString::new(name).unwrap();
        let shm_fd = unsafe {
            libc::shm_open(cname.as_ptr(), libc::O_RDWR, 0o600)
        };

        if shm_fd < 0 {
            return Err(format!("shm_open failed: {}", std::io::Error::last_os_error()));
        }

        // Get shared memory size
        let mut stat: libc::stat = unsafe { std::mem::zeroed() };
        if unsafe { libc::fstat(shm_fd, &mut stat) } < 0 {
            unsafe { libc::close(shm_fd) };
            return Err(format!("fstat failed: {}", std::io::Error::last_os_error()));
        }

        let total_size = stat.st_size as usize;
        let header_size = std::mem::size_of::<TelemetryPoolHeader>();

        if total_size < header_size {
            unsafe { libc::close(shm_fd) };
            return Err("shared memory too small for header".to_string());
        }

        // Memory map
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                total_size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                shm_fd,
                0,
            )
        };

        if ptr == libc::MAP_FAILED {
            unsafe { libc::close(shm_fd) };
            return Err(format!("mmap failed: {}", std::io::Error::last_os_error()));
        }

        let header = ptr as *mut TelemetryPoolHeader;
        let ring = unsafe { ptr.add(header_size) } as *mut PhaseScope;

        // Validate header
        unsafe {
            if (*header).magic != TELEMETRY_POOL_MAGIC {
                libc::munmap(ptr, total_size);
                libc::close(shm_fd);
                return Err("invalid magic number in shared memory".to_string());
            }
        }

        let capacity = unsafe { (*header).capacity };

        Ok(Self {
            shm_fd,
            ptr: ptr as *mut u8,
            size: total_size,
            header,
            ring,
            capacity,
        })
    }

    /// Push PhaseScope to ring buffer (producer only)
    ///
    /// # Arguments
    /// * `scope` - PhaseScope to push
    ///
    /// # Returns
    /// - Ok(()) on success
    /// - Err if ring buffer is full
    pub fn push(&self, scope: PhaseScope) -> Result<(), &'static str> {
        unsafe {
            let head = (*self.header).head.load(Ordering::Acquire);
            let tail = (*self.header).tail.load(Ordering::Acquire);

            // Check if full: (head + 1) % capacity == tail
            let next_head = (head + 1) % self.capacity;
            if next_head == tail {
                return Err("ring buffer full");
            }

            // Write scope to ring[head]
            let entry = self.ring.add(head as usize);
            std::ptr::write(entry, scope);

            // Advance head with Release ordering
            (*self.header).head.store(next_head, Ordering::Release);
        }

        Ok(())
    }

    /// Pull PhaseScope from ring buffer (consumer only)
    ///
    /// # Returns
    /// - Some(PhaseScope) if available
    /// - None if ring buffer is empty
    pub fn pull(&self) -> Option<PhaseScope> {
        unsafe {
            let head = (*self.header).head.load(Ordering::Acquire);
            let tail = (*self.header).tail.load(Ordering::Acquire);

            // Check if empty: head == tail
            if head == tail {
                return None;
            }

            // Read scope from ring[tail]
            let entry = self.ring.add(tail as usize);
            let scope = std::ptr::read(entry);

            // Advance tail with Release ordering
            let next_tail = (tail + 1) % self.capacity;
            (*self.header).tail.store(next_tail, Ordering::Release);

            Some(scope)
        }
    }

    /// Pull all available PhaseScopes (consumer convenience)
    pub fn pull_all(&self) -> Vec<PhaseScope> {
        let mut scopes = Vec::new();
        while let Some(scope) = self.pull() {
            scopes.push(scope);
        }
        scopes
    }

    /// Get current number of PhaseScopes in buffer
    pub fn len(&self) -> usize {
        unsafe {
            let head = (*self.header).head.load(Ordering::Acquire);
            let tail = (*self.header).tail.load(Ordering::Acquire);

            if head >= tail {
                (head - tail) as usize
            } else {
                (self.capacity - tail + head) as usize
            }
        }
    }

    /// Check if ring buffer is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get ring buffer capacity
    pub fn capacity(&self) -> u64 {
        self.capacity
    }
}

impl Drop for TelemetryPool {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.ptr as *mut libc::c_void, self.size);
            libc::close(self.shm_fd);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_and_open() {
        let name = "/nccl_telemetry_test_create";
        let pool = TelemetryPool::create(name, 16).unwrap();
        assert_eq!(pool.capacity(), 16);
        assert!(pool.is_empty());

        // Open from another handle
        let pool2 = TelemetryPool::open(name).unwrap();
        assert_eq!(pool2.capacity(), 16);
        assert!(pool2.is_empty());

        // Cleanup
        drop(pool);
        drop(pool2);
        unsafe { libc::shm_unlink(std::ffi::CString::new(name).unwrap().as_ptr()) };
    }

    #[test]
    fn test_push_pull() {
        let name = "/nccl_telemetry_test_push_pull";
        let pool = TelemetryPool::create(name, 16).unwrap();

        let scope = PhaseScope {
            phase: 100, // Opaque phase tag
            start_time_ns: 1000,
            end_time_ns: 2000,
            e2e_latency_sum_ns: 500,
            net_latency_sum_ns: 400,
            bytes_transferred: 1024,
            transfer_op_count: 10,
            nccl_op_count: 10,
        };

        pool.push(scope).unwrap();
        assert_eq!(pool.len(), 1);

        let pulled = pool.pull().unwrap();
        assert_eq!(pulled.phase, scope.phase);
        assert_eq!(pulled.bytes_transferred, 1024);
        assert!(pool.is_empty());

        // Cleanup
        drop(pool);
        unsafe { libc::shm_unlink(std::ffi::CString::new(name).unwrap().as_ptr()) };
    }

    #[test]
    fn test_push_full() {
        let name = "/nccl_telemetry_test_push_full";
        let pool = TelemetryPool::create(name, 4).unwrap();

        let scope = PhaseScope {
            phase: 100,
            start_time_ns: 1000,
            end_time_ns: 2000,
            e2e_latency_sum_ns: 0,
            net_latency_sum_ns: 0,
            bytes_transferred: 0,
            transfer_op_count: 0,
            nccl_op_count: 0,
        };

        // Push 3 items (capacity - 1, since we lose one slot for full detection)
        pool.push(scope).unwrap();
        pool.push(scope).unwrap();
        pool.push(scope).unwrap();

        // Next push should fail (full)
        assert!(pool.push(scope).is_err());
        assert_eq!(pool.len(), 3);

        // Cleanup
        drop(pool);
        unsafe { libc::shm_unlink(std::ffi::CString::new(name).unwrap().as_ptr()) };
    }

    #[test]
    fn test_pull_all() {
        let name = "/nccl_telemetry_test_pull_all";
        let pool = TelemetryPool::create(name, 16).unwrap();

        for i in 0..5 {
            let scope = PhaseScope {
                phase: 100 + i, // Different phase tags
                start_time_ns: 1000,
                end_time_ns: 2000,
                e2e_latency_sum_ns: 0,
                net_latency_sum_ns: 0,
                bytes_transferred: 0,
                transfer_op_count: 0,
                nccl_op_count: 0,
            };
            pool.push(scope).unwrap();
        }

        assert_eq!(pool.len(), 5);

        let scopes = pool.pull_all();
        assert_eq!(scopes.len(), 5);
        assert!(pool.is_empty());

        for (i, scope) in scopes.iter().enumerate() {
            assert_eq!(scope.phase, 100 + i as u64);
        }

        // Cleanup
        drop(pool);
        unsafe { libc::shm_unlink(std::ffi::CString::new(name).unwrap().as_ptr()) };
    }

    #[test]
    fn test_cross_process() {
        let name = "/nccl_telemetry_test_cross_process";
        let pool = TelemetryPool::create(name, 16).unwrap();

        let scope = PhaseScope {
            phase: 12345, // Opaque phase tag
            start_time_ns: 1000,
            end_time_ns: 2000,
            e2e_latency_sum_ns: 500,
            net_latency_sum_ns: 400,
            bytes_transferred: 2048,
            transfer_op_count: 20,
            nccl_op_count: 20,
        };

        pool.push(scope).unwrap();
        drop(pool);

        // Open from "another process"
        let pool2 = TelemetryPool::open(name).unwrap();
        assert_eq!(pool2.len(), 1);

        let pulled = pool2.pull().unwrap();
        assert_eq!(pulled.phase, 12345);
        assert_eq!(pulled.bytes_transferred, 2048);

        // Cleanup
        drop(pool2);
        unsafe { libc::shm_unlink(std::ffi::CString::new(name).unwrap().as_ptr()) };
    }
}
