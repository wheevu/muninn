use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::bytecode::{BytecodeModule, Chunk, Constant, GlobalValueKind, OpCode, validate_module};
use crate::error::MuninnError;
use crate::jit::{JitBackend, TraceEngine, TraceKey, TraceOutcome, TraceStats};
use crate::native::{
    HostCaps, NativeFunctionKind, add_values, divide_values, invoke_native_with_limit,
    multiply_values, native_name, registered_natives, subtract_values,
};
use crate::runtime::{VmResult, vm_error};
use crate::span::Span;
use crate::value::Value;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReloadStatus {
    Idle,
    Pending,
    Ready,
}

pub struct Vm {
    module: BytecodeModule,
    globals: HashMap<String, Value>,
    policy: HostPolicy,
    steps_used: u64,
    started_at: Instant,
    ops_since_clock_check: u32,
    global_cache: HashMap<String, CachedGlobal>,
    global_cache_stats: GlobalCacheStats,
    globals_epoch: u64,
    stack: Vec<Value>,
    frames: Vec<CallFrame>,
    started: bool,
    pending_reload: Option<BytecodeModule>,
    preserve_existing_globals: bool,
    traces: Option<Box<dyn JitBackend>>,
    /// First fatal execution fault, if any. Faulting instructions leave the
    /// stack and frames mid-operation, so the VM cannot resume: later
    /// `run`/`step_instruction` calls repeat this error instead of
    /// cascading into misleading follow-on errors. A successful
    /// `apply_pending_reload` starts a fresh program and clears it.
    aborted: Option<MuninnError>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VmOptions {
    pub jit_enabled: bool,
    pub hot_loop_threshold: usize,
}

/// Deny-by-default host policy for embedding untrusted scripts.
///
/// Each `Vm` owns an isolated globals map, so running one untrusted script
/// per `Vm` is the isolation boundary: globals, fuel, and clocks are never
/// shared across VMs. `HostPolicy::default()` is permissive (legacy
/// behavior: all natives allowed, no limits); `HostPolicy::sandboxed()` is
/// the deny-by-default starting point where the host explicitly enables
/// what the guest may use.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostPolicy {
    /// Deterministic instruction budget. Exhaustion traps with a vm-phase
    /// error carrying the span of the current instruction. `None` = unlimited.
    pub max_steps: Option<u64>,
    /// Best-effort wall-clock preemption, checked every 1024 instructions
    /// to bound overhead. `None` = no limit.
    pub wall_clock_timeout_ms: Option<u64>,
    /// Cap on tensor element counts, enforced at native allocation
    /// boundaries. Defaults to the built-in 10M limit.
    pub max_tensor_elements: usize,
    /// Allowlist for native builtins. `None` = allow all (legacy);
    /// `Some(set)` = only kinds in the set may run. Disabled builtins
    /// trap with an actionable vm-phase diagnostic, never a panic.
    pub allowed_natives: Option<HashSet<NativeFunctionKind>>,
    /// When true, scripts cannot overwrite existing globals (host-seeded
    /// values) and `SetGlobal` always traps. Fresh `DefineGlobal` bindings
    /// are still allowed.
    pub readonly_globals: bool,
    /// Script arguments visible to `args_len`/`args_get`. Empty by
    /// default; the CLI fills this from arguments after the entry file.
    pub argv: Vec<String>,
    /// Env names visible to `env_get`/`env_has`. `None` allows every name
    /// (trusted CLI default); `Some(set)` allows only names in the set.
    /// Denied names trap with a policy diagnostic; allowed-but-missing
    /// names report `false` from `env_has` and trap from `env_get`.
    pub allowed_env: Option<HashSet<String>>,
    /// Filesystem root scoping `fs_read`/`fs_exists`/`fs_write`. `None`
    /// denies all filesystem access (fail closed, even under `default()`:
    /// the IO surface is new, so there is no legacy behavior to preserve).
    /// `Some(root)` allows relative paths that resolve under `root`.
    pub fs_root: Option<std::path::PathBuf>,
    /// Piped stdin content visible to `stdin_read`. Empty by default; the
    /// CLI fills it when stdin is not a terminal.
    pub stdin_data: String,
}

/// Built-in tensor element limit from `tensor.rs`; the host default.
pub const DEFAULT_MAX_TENSOR_ELEMENTS: usize = 10_000_000;

const CLOCK_CHECK_INTERVAL: u32 = 1024;

impl HostPolicy {
    /// Deny-by-default policy: no natives allowed, bounded fuel and tensor
    /// cap, no wall-clock limit. The host enables natives explicitly via
    /// [`HostPolicy::allow`] and tunes the numeric limits.
    pub fn sandboxed() -> Self {
        Self {
            max_steps: Some(100_000),
            wall_clock_timeout_ms: None,
            max_tensor_elements: DEFAULT_MAX_TENSOR_ELEMENTS,
            allowed_natives: Some(HashSet::new()),
            readonly_globals: false,
            argv: Vec::new(),
            allowed_env: Some(HashSet::new()),
            fs_root: None,
            stdin_data: String::new(),
        }
    }

    /// Explicitly enables one native builtin for the guest.
    pub fn allow(&mut self, kind: NativeFunctionKind) {
        match &mut self.allowed_natives {
            Some(set) => {
                set.insert(kind);
            }
            None => {
                let mut set = HashSet::new();
                set.insert(kind);
                self.allowed_natives = Some(set);
            }
        }
    }

    /// Allows every known native builtin (restores legacy behavior).
    pub fn allow_all_natives(&mut self) {
        self.allowed_natives = None;
    }

    pub fn is_native_allowed(&self, kind: NativeFunctionKind) -> bool {
        self.allowed_natives
            .as_ref()
            .is_none_or(|set| set.contains(&kind))
    }

    /// Explicitly allows one env name for `env_get`/`env_has`. Converts
    /// an allow-all policy into an allowlist containing just `name`.
    pub fn allow_env(&mut self, name: impl Into<String>) {
        match &mut self.allowed_env {
            Some(set) => {
                set.insert(name.into());
            }
            None => {
                let mut set = HashSet::new();
                set.insert(name.into());
                self.allowed_env = Some(set);
            }
        }
    }

    /// Allows every env name (restores the trusted-CLI default).
    pub fn allow_all_env(&mut self) {
        self.allowed_env = None;
    }

    /// Shrinks the ambient surface after startup: read config, then drop
    /// filesystem access so later script code cannot reach it.
    pub fn revoke_fs(&mut self) {
        self.fs_root = None;
    }

    /// Grants script arguments visible to `args_len`/`args_get`.
    pub fn set_argv(&mut self, argv: Vec<String>) {
        self.argv = argv;
    }

    /// Scopes filesystem builtins under `root`.
    pub fn set_fs_root(&mut self, root: std::path::PathBuf) {
        self.fs_root = Some(root);
    }

    /// Seeds piped stdin content visible to `stdin_read`.
    pub fn set_stdin(&mut self, data: String) {
        self.stdin_data = data;
    }

    /// Builds the per-call capability view for native dispatch. The VM
    /// stays the isolation boundary: one `Vm` per guest, budgets and
    /// capabilities never shared across VMs.
    fn host_caps(&self) -> HostCaps<'_> {
        HostCaps {
            argv: &self.argv,
            allowed_env: &self.allowed_env,
            fs_root: &self.fs_root,
            stdin_data: &self.stdin_data,
        }
    }
}

impl Default for HostPolicy {
    fn default() -> Self {
        Self {
            max_steps: None,
            wall_clock_timeout_ms: None,
            max_tensor_elements: DEFAULT_MAX_TENSOR_ELEMENTS,
            allowed_natives: None,
            readonly_globals: false,
            argv: Vec::new(),
            allowed_env: None,
            fs_root: None,
            stdin_data: String::new(),
        }
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct GlobalCacheStats {
    pub hits: usize,
    pub misses: usize,
    /// Number of cache eviction events: single-key evictions on global
    /// writes plus whole-cache clears on reload. Includes events where the
    /// evicted key (or the whole cache) was already empty.
    pub invalidations: usize,
}

impl Default for VmOptions {
    fn default() -> Self {
        Self {
            jit_enabled: false,
            hot_loop_threshold: 100,
        }
    }
}

#[derive(Debug, Clone)]
struct CachedGlobal {
    value: Value,
    epoch: u64,
}

#[derive(Debug, Clone, Copy)]
struct CallFrame {
    function_id: usize,
    ip: usize,
    stack_base: usize,
}

impl Vm {
    pub fn new(module: BytecodeModule) -> Self {
        Self::new_with_options(module, VmOptions::default())
    }

    pub fn new_with_options(module: BytecodeModule, options: VmOptions) -> Self {
        Self::new_with_policy_and_options(module, HostPolicy::default(), options)
    }

    /// Embeds a module under an explicit host policy. Use one `Vm` per
    /// untrusted script: globals and budgets are per-VM, never shared.
    pub fn new_with_policy(module: BytecodeModule, policy: HostPolicy) -> Self {
        Self::new_with_policy_and_options(module, policy, VmOptions::default())
    }

    pub fn new_with_policy_and_options(
        module: BytecodeModule,
        policy: HostPolicy,
        options: VmOptions,
    ) -> Self {
        let mut vm = Self {
            globals: HashMap::new(),
            policy,
            global_cache: HashMap::new(),
            global_cache_stats: GlobalCacheStats::default(),
            globals_epoch: 0,
            stack: Vec::new(),
            frames: Vec::new(),
            started: false,
            pending_reload: None,
            preserve_existing_globals: false,
            traces: options.jit_enabled.then(|| {
                Box::new(TraceEngine::new(options.hot_loop_threshold)) as Box<dyn JitBackend>
            }),
            aborted: None,
            module,
            steps_used: 0,
            started_at: Instant::now(),
            ops_since_clock_check: 0,
        };
        vm.install_natives();
        vm.reserve_runtime_capacity(
            vm.module.estimated_stack_capacity(),
            vm.module.estimated_frame_capacity(),
        );
        vm
    }

    pub fn jit_stats(&self) -> Option<TraceStats> {
        self.traces.as_ref().map(|traces| traces.stats())
    }

    pub fn global_cache_stats(&self) -> GlobalCacheStats {
        self.global_cache_stats
    }

    pub fn reserve_runtime_capacity(&mut self, stack_capacity: usize, frame_capacity: usize) {
        if self.stack.capacity() < stack_capacity {
            self.stack.reserve(stack_capacity - self.stack.capacity());
        }
        if self.frames.capacity() < frame_capacity {
            self.frames.reserve(frame_capacity - self.frames.capacity());
        }
    }

    pub fn run(&mut self) -> VmResult<Value> {
        if let Some(error) = &self.aborted {
            return Err(error.clone());
        }
        self.ensure_started(Span::default())?;
        loop {
            if self.poll_safe_point() == ReloadStatus::Ready {
                self.apply_pending_reload()?;
            }

            if let Some(value) = self.step_instruction()? {
                return Ok(value);
            }
        }
    }

    pub fn step_instruction(&mut self) -> VmResult<Option<Value>> {
        if let Some(error) = &self.aborted {
            return Err(error.clone());
        }
        self.ensure_started(Span::default())?;
        let result = self.execute_instruction();
        if let Err(error) = &result {
            self.aborted = Some(error.clone());
        }
        result
    }

    pub fn request_reload(&mut self, module: BytecodeModule) -> VmResult<()> {
        validate_module(&module).map_err(first_validation_error)?;
        self.invalidate_runtime_caches();
        self.pending_reload = Some(module);
        Ok(())
    }

    pub fn poll_safe_point(&self) -> ReloadStatus {
        // There is intentionally no execution budget here: a pending reload
        // staged against a non-terminating nested call waits indefinitely
        // (see the `top_level_loop_is_a_safe_point_but_nested_loop_is_not`
        // test). Callers own that scheduling decision.
        if self.pending_reload.is_none() {
            ReloadStatus::Idle
        } else if self.is_safe_point() {
            ReloadStatus::Ready
        } else {
            ReloadStatus::Pending
        }
    }

    pub fn apply_pending_reload(&mut self) -> VmResult<()> {
        if self.pending_reload.is_none() {
            return Ok(());
        }
        if !self.is_safe_point() {
            return Err(vm_error(
                "reload is only allowed at a safe point",
                Span::default(),
            ));
        }

        let Some(pending) = self.pending_reload.take() else {
            return Err(vm_error(
                "reload requested but no pending module",
                Span::default(),
            ));
        };
        self.validate_reload_compatibility(&pending)?;

        self.module = pending;
        self.invalidate_runtime_caches();
        self.started = false;
        self.aborted = None;
        self.preserve_existing_globals = true;
        self.reserve_runtime_capacity(
            self.module.estimated_stack_capacity(),
            self.module.estimated_frame_capacity(),
        );
        self.ensure_started(Span::default())
    }

    pub fn frame_depth(&self) -> usize {
        self.frames.len()
    }

    pub fn global(&self, name: &str) -> Option<&Value> {
        self.globals.get(name)
    }

    fn install_natives(&mut self) {
        for native in registered_natives() {
            self.globals
                .insert(native.name.to_string(), Value::Native(native.kind));
        }
    }

    fn ensure_started(&mut self, span: Span) -> VmResult<()> {
        if self.started {
            return Ok(());
        }
        self.stack.clear();
        self.frames.clear();
        self.steps_used = 0;
        self.started_at = Instant::now();
        self.ops_since_clock_check = 0;
        self.push_frame(self.module.entry_function, 0, span)?;
        self.started = true;
        Ok(())
    }

    /// Returns the host policy this VM executes under.
    pub fn policy(&self) -> &HostPolicy {
        &self.policy
    }

    /// Returns the deterministic instruction count consumed so far.
    pub fn steps_used(&self) -> u64 {
        self.steps_used
    }

    /// Enforces the host fuel and wall-clock budgets. Fuel exhaustion is
    /// deterministic: the same script with the same `max_steps` traps at
    /// the same instruction. Wall-clock preemption is best-effort and
    /// checked every [`CLOCK_CHECK_INTERVAL`] instructions to bound overhead.
    fn check_budget(&mut self, span: Span) -> VmResult<()> {
        self.steps_used += 1;
        if let Some(max_steps) = self.policy.max_steps
            && self.steps_used > max_steps
        {
            return Err(vm_error(
                format!(
                    "fuel exhausted after {} instructions (max_steps {})",
                    self.steps_used, max_steps
                ),
                span,
            ));
        }
        self.ops_since_clock_check += 1;
        if self.ops_since_clock_check >= CLOCK_CHECK_INTERVAL {
            self.ops_since_clock_check = 0;
            if let Some(timeout_ms) = self.policy.wall_clock_timeout_ms
                && self.started_at.elapsed() > Duration::from_millis(timeout_ms)
            {
                return Err(vm_error(
                    format!("wall-clock timeout after {} ms", timeout_ms),
                    span,
                ));
            }
        }
        Ok(())
    }

    fn execute_instruction(&mut self) -> VmResult<Option<Value>> {
        if self.frames.is_empty() {
            self.started = false;
            self.preserve_existing_globals = false;
            return Ok(Some(self.stack.pop().unwrap_or(Value::Nil)));
        }

        let frame_index = self.frames.len() - 1;
        let function_id = self.frames[frame_index].function_id;
        let ip = self.frames[frame_index].ip;

        if self.pending_reload.is_none()
            && let Some(traces) = &mut self.traces
            && let Some(outcome) = traces.run_if_ready(
                TraceKey {
                    function_id,
                    loop_header_ip: ip,
                },
                &self.module,
                &mut self.stack,
                self.frames[frame_index].stack_base,
            )?
        {
            match outcome {
                TraceOutcome::Continue => return Ok(None),
                TraceOutcome::ExitToInterpreter { ip } => {
                    self.frames[frame_index].ip = ip;
                    return Ok(None);
                }
            }
        }

        let span = self.current_chunk(function_id).span_at(ip);
        self.check_budget(span)?;
        let byte = *self
            .current_chunk(function_id)
            .code
            .get(ip)
            .ok_or_else(|| vm_error("instruction pointer out of range", span))?;
        let op = OpCode::from_byte(byte)
            .ok_or_else(|| vm_error(format!("invalid opcode {}", byte), span))?;
        self.frames[frame_index].ip += 1;

        match op {
            OpCode::Constant => {
                let index = self.read_u16(frame_index, span)? as usize;
                let constant = self
                    .current_chunk(function_id)
                    .constants
                    .get(index)
                    .ok_or_else(|| vm_error(format!("invalid constant index {}", index), span))?;
                self.stack.push(self.constant_to_value(constant));
            }
            OpCode::Nil => self.stack.push(Value::Nil),
            OpCode::True => self.stack.push(Value::Bool(true)),
            OpCode::False => self.stack.push(Value::Bool(false)),
            OpCode::Pop => {
                self.stack.pop();
            }
            OpCode::GetLocal => {
                let slot = self.read_u16(frame_index, span)? as usize;
                let stack_index = self.local_stack_index(frame_index, slot, span)?;
                self.stack.push(self.stack[stack_index].clone());
            }
            OpCode::SetLocal => {
                let slot = self.read_u16(frame_index, span)? as usize;
                let value = self.pop(span)?;
                let stack_index = self.local_stack_index(frame_index, slot, span)?;
                self.stack[stack_index] = value;
            }
            OpCode::DefineGlobal => {
                let name = self.read_name(frame_index, span)?;
                let value = self.pop(span)?;
                let preserve = self.preserve_existing_globals
                    && self.globals.contains_key(&name)
                    && self.module.global_kind(&name) != Some(GlobalValueKind::Function);
                if !preserve {
                    self.evict_global_key(&name);
                    self.globals.insert(name, value);
                }
            }
            OpCode::GetGlobal => {
                let name = self.read_name(frame_index, span)?;
                let value = self.get_global_cached(&name, span)?;
                self.stack.push(value);
            }
            OpCode::BuildRecord => {
                let fields = self.read_u16(frame_index, span)? as usize;
                if self.stack.len() < fields.saturating_mul(2) {
                    return Err(vm_error("stack underflow", span));
                }
                let mut record = std::collections::BTreeMap::new();
                for _ in 0..fields {
                    let name = self.pop(span)?;
                    let value = self.pop(span)?;
                    let Value::String(name) = name else {
                        return Err(vm_error("record field name must be a string", span));
                    };
                    record.insert(name.to_string(), value);
                }
                self.stack.push(Value::Record(record));
            }
            OpCode::GetField => {
                let name = self.read_name(frame_index, span)?;
                let base = self.pop(span)?;
                match base {
                    Value::Record(fields) => match fields.get(&name) {
                        Some(value) => self.stack.push(value.clone()),
                        None => {
                            return Err(vm_error(format!("record has no field '{}'", name), span));
                        }
                    },
                    other => {
                        return Err(vm_error(
                            format!(
                                "value of type {} has no fields (field '{}')",
                                other.kind_name(),
                                name
                            ),
                            span,
                        ));
                    }
                }
            }
            OpCode::SetGlobal => {
                let name = self.read_name(frame_index, span)?;
                let value = self.pop(span)?;
                if !self.globals.contains_key(&name) {
                    return Err(vm_error(format!("unknown global '{}'", name), span));
                }
                if self.policy.readonly_globals {
                    return Err(vm_error(
                        format!("global '{}' is readonly under this host policy", name),
                        span,
                    ));
                }
                self.evict_global_key(&name);
                self.globals.insert(name, value);
            }
            OpCode::Add => {
                let right = self.pop(span)?;
                let left = self.pop(span)?;
                self.stack.push(add_values(left, right, span)?);
            }
            OpCode::Subtract => {
                let right = self.pop(span)?;
                let left = self.pop(span)?;
                self.stack.push(subtract_values(left, right, span)?);
            }
            OpCode::Multiply => {
                let right = self.pop(span)?;
                let left = self.pop(span)?;
                self.stack.push(multiply_values(left, right, span)?);
            }
            OpCode::Divide => {
                let right = self.pop(span)?;
                let left = self.pop(span)?;
                self.stack.push(divide_values(left, right, span)?);
            }
            OpCode::Negate => {
                let value = self.pop(span)?;
                match value {
                    Value::Int(value) => {
                        let negated = value
                            .checked_neg()
                            .ok_or_else(|| vm_error("integer overflow in negation", span))?;
                        self.stack.push(Value::Int(negated));
                    }
                    Value::Float(value) => self.stack.push(Value::Float(-value)),
                    other => {
                        return Err(vm_error(
                            format!("cannot negate {}", other.stringify()),
                            span,
                        ));
                    }
                }
            }
            OpCode::Not => {
                let value = self.pop(span)?;
                match value {
                    Value::Bool(value) => self.stack.push(Value::Bool(!value)),
                    other => {
                        return Err(vm_error(
                            format!("cannot apply '!' to {}", other.stringify()),
                            span,
                        ));
                    }
                }
            }
            OpCode::Equal => {
                let right = self.pop(span)?;
                let left = self.pop(span)?;
                self.stack.push(Value::Bool(left.equals(&right)));
            }
            OpCode::Greater => self.ordering_compare(span, |ord| ord.is_gt())?,
            OpCode::Less => self.ordering_compare(span, |ord| ord.is_lt())?,
            OpCode::JumpIfFalse => {
                let jump = self.read_u16(frame_index, span)? as usize;
                let condition = self
                    .stack
                    .last()
                    .cloned()
                    .ok_or_else(|| vm_error("stack underflow", span))?;
                match condition {
                    Value::Bool(value) => {
                        if !value {
                            self.frames[frame_index].ip += jump;
                        }
                    }
                    other => {
                        return Err(vm_error(
                            format!("condition must be Bool, got {}", other.stringify()),
                            span,
                        ));
                    }
                }
            }
            OpCode::Jump => {
                let jump = self.read_u16(frame_index, span)? as usize;
                self.frames[frame_index].ip += jump;
            }
            OpCode::Loop => {
                let jump = self.read_u16(frame_index, span)? as usize;
                let loop_header_ip = self.frames[frame_index].ip.saturating_sub(jump);
                self.frames[frame_index].ip = loop_header_ip;
                if let Some(traces) = &mut self.traces {
                    traces.observe_loop(
                        &self.module,
                        TraceKey {
                            function_id,
                            loop_header_ip,
                        },
                    );
                }
            }
            OpCode::Call => {
                let arg_count = self.read_u8(frame_index, span)? as usize;
                self.call_value(arg_count, span)?;
            }
            OpCode::Return => {
                let value = self.stack.pop().unwrap_or(Value::Nil);
                let function = &self.module.functions[function_id];
                if function.expects_return_value && matches!(value, Value::Nil) {
                    return Err(vm_error(
                        format!(
                            "function '{}' fell through without returning a value",
                            function.name
                        ),
                        span,
                    ));
                }

                let Some(frame) = self.frames.pop() else {
                    return Err(vm_error("return without a call frame", span));
                };
                self.stack.truncate(frame.stack_base);
                if self.frames.is_empty() {
                    self.started = false;
                    self.preserve_existing_globals = false;
                    return Ok(Some(value));
                }
                self.stack.push(value);
            }
        }

        Ok(None)
    }

    fn is_safe_point(&self) -> bool {
        self.frames.len() <= 1
    }

    /// Rejects reloads that drop or retype live globals. Errors here
    /// deliberately carry `Span::default`: this is a cross-module relation
    /// and neither side retains a source location for a global (module
    /// specs and runtime values are span-free), so the global name and
    /// both kinds in the message are the location.
    fn validate_reload_compatibility(&self, next_module: &BytecodeModule) -> VmResult<()> {
        if next_module.entry_function >= next_module.functions.len() {
            return Err(vm_error(
                "reload module is missing a valid entry function",
                Span::default(),
            ));
        }

        for (name, value) in &self.globals {
            if matches!(value, Value::Native(_)) {
                continue;
            }

            let Some(expected_kind) = next_module.global_kind(name) else {
                return Err(vm_error(
                    format!(
                        "reload rejected: global '{}' is missing in new module",
                        name
                    ),
                    Span::default(),
                ));
            };

            if !value_matches_kind(value, expected_kind) {
                return Err(vm_error(
                    format!(
                        "reload rejected: global '{}' changed kind from {} to {:?}",
                        name,
                        value.kind_name(),
                        expected_kind
                    ),
                    Span::default(),
                ));
            }
        }

        Ok(())
    }

    fn current_chunk(&self, function_id: usize) -> &Chunk {
        &self.module.functions[function_id].chunk
    }

    fn read_u8(&mut self, frame_index: usize, span: Span) -> VmResult<u8> {
        let function_id = self.frames[frame_index].function_id;
        let ip = self.frames[frame_index].ip;
        let byte = *self
            .current_chunk(function_id)
            .code
            .get(ip)
            .ok_or_else(|| vm_error("instruction pointer out of range", span))?;
        self.frames[frame_index].ip += 1;
        Ok(byte)
    }

    fn read_u16(&mut self, frame_index: usize, span: Span) -> VmResult<u16> {
        let low = self.read_u8(frame_index, span)?;
        let high = self.read_u8(frame_index, span)?;
        Ok(u16::from_le_bytes([low, high]))
    }

    fn read_name(&mut self, frame_index: usize, span: Span) -> VmResult<String> {
        let function_id = self.frames[frame_index].function_id;
        let index = self.read_u16(frame_index, span)? as usize;
        let constant = self
            .current_chunk(function_id)
            .constants
            .get(index)
            .ok_or_else(|| vm_error(format!("invalid constant index {}", index), span))?;
        if let Constant::String(name) = constant {
            Ok(name.clone())
        } else {
            Err(vm_error("expected string constant", span))
        }
    }

    fn local_stack_index(&self, frame_index: usize, slot: usize, span: Span) -> VmResult<usize> {
        let function_id = self.frames[frame_index].function_id;
        let local_count = self.module.functions[function_id].local_count;
        if slot >= local_count {
            return Err(vm_error(format!("invalid local slot {}", slot), span));
        }
        Ok(self.frames[frame_index].stack_base + slot)
    }

    fn call_value(&mut self, arg_count: usize, span: Span) -> VmResult<()> {
        if self.stack.len() < arg_count + 1 {
            return Err(vm_error("stack underflow", span));
        }

        let callee_index = self.stack.len() - arg_count - 1;
        let callee = self.stack[callee_index].clone();
        match callee {
            Value::Function(function_id) => {
                self.stack.remove(callee_index);
                self.push_frame(function_id, arg_count, span)
            }
            Value::Native(kind) => {
                if !self.policy.is_native_allowed(kind) {
                    return Err(vm_error(
                        format!("native '{}' is disabled by host policy", native_name(kind)),
                        span,
                    ));
                }
                let max_elements = self.policy.max_tensor_elements;
                let caps = self.policy.host_caps();
                let result = invoke_native_with_limit(
                    kind,
                    &self.stack[callee_index + 1..],
                    span,
                    max_elements,
                    &caps,
                )?;
                self.stack.truncate(callee_index);
                self.stack.push(result);
                Ok(())
            }
            other => Err(vm_error(
                format!("{} is not callable", other.stringify()),
                span,
            )),
        }
    }

    fn push_frame(&mut self, function_id: usize, arg_count: usize, span: Span) -> VmResult<()> {
        let function = self
            .module
            .functions
            .get(function_id)
            .ok_or_else(|| vm_error(format!("invalid function id {}", function_id), span))?;
        if function.arity != arg_count {
            return Err(vm_error(
                format!(
                    "function '{}' expects {} arguments, got {}",
                    function.name, function.arity, arg_count
                ),
                span,
            ));
        }

        let stack_base = self.stack.len().saturating_sub(arg_count);
        for _ in arg_count..function.local_count {
            self.stack.push(Value::Nil);
        }
        self.frames.push(CallFrame {
            function_id,
            ip: 0,
            stack_base,
        });
        Ok(())
    }

    fn ordering_compare(
        &mut self,
        span: Span,
        predicate: impl FnOnce(std::cmp::Ordering) -> bool,
    ) -> VmResult<()> {
        let right = self.pop(span)?;
        let left = self.pop(span)?;
        match (left, right) {
            (Value::Int(left), Value::Int(right)) => {
                self.stack.push(Value::Bool(predicate(left.cmp(&right))));
                Ok(())
            }
            (Value::Float(left), Value::Float(right)) => {
                let ordering = left
                    .partial_cmp(&right)
                    .ok_or_else(|| vm_error("cannot compare NaN values", span))?;
                self.stack.push(Value::Bool(predicate(ordering)));
                Ok(())
            }
            (left, right) => Err(vm_error(
                format!(
                    "ordering comparison expects matching numeric types, got {} and {}",
                    left.stringify(),
                    right.stringify()
                ),
                span,
            )),
        }
    }

    fn constant_to_value(&self, constant: &Constant) -> Value {
        match constant {
            Constant::Int(value) => Value::Int(*value),
            Constant::Float(value) => Value::Float(*value),
            Constant::Bool(value) => Value::Bool(*value),
            Constant::String(value) => Value::String(Arc::<str>::from(value.as_str())),
            Constant::Function(value) => Value::Function(*value),
            Constant::Nil => Value::Nil,
        }
    }

    fn pop(&mut self, span: Span) -> VmResult<Value> {
        self.stack
            .pop()
            .ok_or_else(|| vm_error("stack underflow", span))
    }

    fn get_global_cached(&mut self, name: &str, span: Span) -> VmResult<Value> {
        if let Some(cached) = self.global_cache.get(name)
            && cached.epoch == self.globals_epoch
        {
            self.global_cache_stats.hits += 1;
            return Ok(cached.value.clone());
        }

        self.global_cache_stats.misses += 1;

        let value = self
            .globals
            .get(name)
            .cloned()
            .ok_or_else(|| vm_error(format!("unknown global '{}'", name), span))?;
        self.global_cache.insert(
            name.to_string(),
            CachedGlobal {
                value: value.clone(),
                epoch: self.globals_epoch,
            },
        );
        Ok(value)
    }

    fn bump_globals_epoch(&mut self) {
        self.globals_epoch = self.globals_epoch.wrapping_add(1);
        self.global_cache_stats.invalidations += 1;
        if !self.global_cache.is_empty() {
            self.global_cache.clear();
        }
    }

    /// Evicts a single global from the lookup cache after a write.
    ///
    /// Unlike [`Vm::bump_globals_epoch`], which clears the whole cache on
    /// reload, a write to one global must not evict unrelated globals:
    /// read/write-interleaved loops (for example `acc = acc + val`) would
    /// otherwise miss on every iteration. The invalidation counter still
    /// records the event even when the key was not cached.
    fn evict_global_key(&mut self, name: &str) {
        self.global_cache.remove(name);
        self.global_cache_stats.invalidations += 1;
    }

    fn invalidate_runtime_caches(&mut self) {
        self.bump_globals_epoch();
        if let Some(traces) = &mut self.traces {
            traces.clear();
        }
    }
}

fn value_matches_kind(value: &Value, kind: GlobalValueKind) -> bool {
    matches!(
        (value, kind),
        (Value::Int(_), GlobalValueKind::Int)
            | (Value::Float(_), GlobalValueKind::Float)
            | (Value::Bool(_), GlobalValueKind::Bool)
            | (Value::String(_), GlobalValueKind::String)
            | (Value::Tensor(_), GlobalValueKind::Tensor)
            | (Value::Function(_), GlobalValueKind::Function)
            | (Value::Record(_), GlobalValueKind::Record)
    )
}

fn first_validation_error(errors: Vec<MuninnError>) -> MuninnError {
    let first = errors.into_iter().next().unwrap_or_else(|| {
        MuninnError::new("compiler", "invalid bytecode module", Span::default())
    });
    vm_error(first.message, first.span)
}
