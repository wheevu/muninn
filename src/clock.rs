//! A monotonic clock that works on every target Muninn builds for.
//!
//! `std::time::Instant` traps on `wasm32-unknown-unknown`, which has no clock
//! to read, so the web build carries no wall-clock signal at all. The VM's
//! wall-clock budget is documented as best-effort for that reason, and fuel
//! (`HostPolicy::max_steps`) stays the deterministic bound. A host that needs
//! a hard deadline on a target without a clock enforces it outside the VM.

use std::time::{Duration, Instant};

pub(crate) struct Clock {
    started: Option<Instant>,
}

impl Clock {
    pub(crate) fn now() -> Self {
        #[cfg(not(target_arch = "wasm32"))]
        {
            Clock {
                started: Some(Instant::now()),
            }
        }
        #[cfg(target_arch = "wasm32")]
        {
            Clock { started: None }
        }
    }

    /// Time since this clock was read, or zero on a target with no clock.
    pub(crate) fn elapsed(&self) -> Duration {
        self.started
            .map_or(Duration::ZERO, |started| started.elapsed())
    }
}
