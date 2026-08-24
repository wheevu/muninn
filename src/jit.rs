use std::collections::{HashMap, HashSet};

use crate::bytecode::{BytecodeModule, Chunk, Constant, OpCode};
use crate::runtime::{VmResult, vm_error};
use crate::span::Span;
use crate::value::Value;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TraceKey {
    pub function_id: usize,
    pub loop_header_ip: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TraceRunMode {
    Interpreted,
    #[cfg(feature = "jit")]
    Native,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct TraceStats {
    pub hot_loops_seen: usize,
    pub traces_compiled: usize,
    pub traces_rejected: usize,
    pub interpreted_trace_runs: usize,
    pub native_trace_runs: usize,
    #[cfg(feature = "jit")]
    pub native_trace_bailouts: usize,
}

#[derive(Debug)]
pub enum TraceOutcome {
    Continue,
    ExitToInterpreter { ip: usize },
}

pub struct TraceEngine {
    threshold: usize,
    counters: HashMap<TraceKey, usize>,
    traces: HashMap<TraceKey, Trace>,
    rejected: HashSet<TraceKey>,
    stats: TraceStats,
}

impl TraceEngine {
    pub fn new(threshold: usize) -> Self {
        Self {
            threshold: threshold.max(1),
            counters: HashMap::new(),
            traces: HashMap::new(),
            rejected: HashSet::new(),
            stats: TraceStats::default(),
        }
    }

    pub fn stats(&self) -> TraceStats {
        self.stats
    }

    pub fn clear(&mut self) {
        self.counters.clear();
        self.traces.clear();
        self.rejected.clear();
    }

    pub fn observe_loop(&mut self, module: &BytecodeModule, key: TraceKey) {
        if self.traces.contains_key(&key) || self.rejected.contains(&key) {
            return;
        }

        let counter = self.counters.entry(key).or_insert(0);
        *counter += 1;
        self.stats.hot_loops_seen += 1;
        if *counter < self.threshold {
            return;
        }

        match Trace::compile(module, key) {
            Ok(trace) => {
                self.stats.traces_compiled += 1;
                self.traces.insert(key, trace);
            }
            Err(_) => {
                self.stats.traces_rejected += 1;
                self.rejected.insert(key);
            }
        }
    }

    pub fn run_if_ready(
        &mut self,
        key: TraceKey,
        module: &BytecodeModule,
        stack: &mut Vec<Value>,
        stack_base: usize,
    ) -> VmResult<Option<TraceOutcome>> {
        let Some(trace) = self.traces.get_mut(&key) else {
            return Ok(None);
        };

        let result = trace.run(module, stack, stack_base);
        match trace.last_run_mode() {
            TraceRunMode::Interpreted => self.stats.interpreted_trace_runs += 1,
            #[cfg(feature = "jit")]
            TraceRunMode::Native => {
                self.stats.native_trace_runs += 1;
                if trace.last_native_bailed_out() {
                    self.stats.native_trace_bailouts += 1;
                }
            }
        }

        match result? {
            TraceRunResult::LoopHeader => Ok(Some(TraceOutcome::Continue)),
            TraceRunResult::ExitToInterpreter { ip } if ip == key.loop_header_ip => {
                // The trace cannot make progress (for example, one of its
                // locals is not an Int): retrying it would spin forever at
                // the loop header. Drop the trace and let the interpreter
                // take over this loop.
                self.traces.remove(&key);
                self.rejected.insert(key);
                Ok(None)
            }
            TraceRunResult::ExitToInterpreter { ip } => {
                Ok(Some(TraceOutcome::ExitToInterpreter { ip }))
            }
        }
    }
}

pub struct Trace {
    key: TraceKey,
    ops: Vec<TraceOp>,
    local_slots: Vec<usize>,
    #[cfg_attr(not(feature = "jit"), allow(dead_code))]
    loop_span: Span,
    last_run_mode: TraceRunMode,
    #[cfg_attr(not(feature = "jit"), allow(dead_code))]
    last_native_bailed_out: bool,
    #[cfg(feature = "jit")]
    native: Option<NativeTrace>,
}

impl Trace {
    fn compile(module: &BytecodeModule, key: TraceKey) -> Result<Self, TraceBuildError> {
        let function = module
            .functions
            .get(key.function_id)
            .ok_or(TraceBuildError)?;
        let chunk = &function.chunk;
        let mut ip = key.loop_header_ip;
        let mut ops = Vec::new();
        let mut local_slots = Vec::new();
        let mut exit_ip = None;
        let mut stack_depth: isize = 0;
        let mut loop_span = chunk.span_at(ip);

        while ip < chunk.code.len() {
            let op_ip = ip;
            let span = chunk.span_at(op_ip);
            let op = read_op(chunk, &mut ip)?;
            match op {
                OpCode::Constant => {
                    let index = read_u16(chunk, &mut ip)? as usize;
                    match chunk.constants.get(index) {
                        Some(Constant::Int(value)) => {
                            ops.push(TraceOp::ConstInt(*value));
                            stack_depth += 1;
                        }
                        _ => return Err(TraceBuildError),
                    }
                }
                OpCode::True => {
                    ops.push(TraceOp::ConstBool(true));
                    stack_depth += 1;
                }
                OpCode::False => {
                    ops.push(TraceOp::ConstBool(false));
                    stack_depth += 1;
                }
                OpCode::GetLocal => {
                    let slot = read_u16(chunk, &mut ip)? as usize;
                    remember_slot(&mut local_slots, slot);
                    ops.push(TraceOp::GetLocal { slot, span });
                    stack_depth += 1;
                }
                OpCode::SetLocal => {
                    let slot = read_u16(chunk, &mut ip)? as usize;
                    remember_slot(&mut local_slots, slot);
                    ops.push(TraceOp::SetLocal { slot, span });
                    stack_depth -= 1;
                }
                OpCode::Add | OpCode::Subtract | OpCode::Multiply | OpCode::Divide => {
                    let kind = match op {
                        OpCode::Add => IntBinaryOp::Add,
                        OpCode::Subtract => IntBinaryOp::Subtract,
                        OpCode::Multiply => IntBinaryOp::Multiply,
                        OpCode::Divide => IntBinaryOp::Divide,
                        _ => unreachable!(),
                    };
                    ops.push(TraceOp::IntBinary { kind, span });
                    stack_depth -= 1;
                }
                OpCode::Equal | OpCode::Greater | OpCode::Less => {
                    let kind = match op {
                        OpCode::Equal => IntCompareOp::Equal,
                        OpCode::Greater => IntCompareOp::Greater,
                        OpCode::Less => IntCompareOp::Less,
                        _ => unreachable!(),
                    };
                    ops.push(TraceOp::IntCompare { kind, span });
                    stack_depth -= 1;
                }
                OpCode::Pop => {
                    ops.push(TraceOp::Pop { span });
                    stack_depth -= 1;
                }
                OpCode::JumpIfFalse => {
                    let jump = read_u16(chunk, &mut ip)? as usize;
                    let target_ip = ip + jump;
                    exit_ip = Some(target_ip);
                    ops.push(TraceOp::JumpIfFalse { target_ip, span });
                }
                OpCode::Loop => {
                    let jump = read_u16(chunk, &mut ip)? as usize;
                    let target = ip.saturating_sub(jump);
                    if target != key.loop_header_ip || stack_depth != 0 {
                        return Err(TraceBuildError);
                    }
                    loop_span = span;
                    ops.push(TraceOp::Loop);
                    break;
                }
                _ => return Err(TraceBuildError),
            }
        }

        if !matches!(ops.last(), Some(TraceOp::Loop)) || exit_ip.is_none() {
            return Err(TraceBuildError);
        }

        local_slots.sort_unstable();
        #[cfg(feature = "jit")]
        let native = None;
        #[cfg_attr(not(feature = "jit"), allow(unused_mut))]
        let mut trace = Self {
            key,
            ops,
            local_slots,
            loop_span,
            last_run_mode: TraceRunMode::Interpreted,
            last_native_bailed_out: false,
            #[cfg(feature = "jit")]
            native,
        };
        #[cfg(feature = "jit")]
        {
            trace.native = NativeTrace::compile(&trace).ok();
        }
        Ok(trace)
    }

    fn last_run_mode(&self) -> TraceRunMode {
        self.last_run_mode
    }

    #[cfg_attr(not(feature = "jit"), allow(dead_code))]
    fn last_native_bailed_out(&self) -> bool {
        self.last_native_bailed_out
    }

    fn run(
        &mut self,
        module: &BytecodeModule,
        stack: &mut Vec<Value>,
        stack_base: usize,
    ) -> VmResult<TraceRunResult> {
        if !self.locals_are_int(stack, stack_base) {
            self.last_run_mode = TraceRunMode::Interpreted;
            self.last_native_bailed_out = false;
            return Ok(TraceRunResult::ExitToInterpreter {
                ip: self.key.loop_header_ip,
            });
        }

        self.last_native_bailed_out = false;
        #[cfg(feature = "jit")]
        if let Some(native) = &self.native {
            let (bailed_out, result) = native.run(self, module, stack, stack_base);
            self.last_run_mode = TraceRunMode::Native;
            self.last_native_bailed_out = bailed_out;
            return result;
        }

        self.last_run_mode = TraceRunMode::Interpreted;
        self.run_interpreted(module, stack, stack_base)
    }

    fn locals_are_int(&self, stack: &[Value], stack_base: usize) -> bool {
        self.local_slots
            .iter()
            .all(|slot| matches!(stack.get(stack_base + slot), Some(Value::Int(_))))
    }

    fn run_interpreted(
        &self,
        _module: &BytecodeModule,
        stack: &mut Vec<Value>,
        stack_base: usize,
    ) -> VmResult<TraceRunResult> {
        let mut trace_stack = Vec::new();
        for op in &self.ops {
            match *op {
                TraceOp::ConstInt(value) => trace_stack.push(TraceValue::Int(value)),
                TraceOp::ConstBool(value) => trace_stack.push(TraceValue::Bool(value)),
                TraceOp::GetLocal { slot, span } => {
                    let value = read_int_local(stack, stack_base, slot, span)?;
                    trace_stack.push(TraceValue::Int(value));
                }
                TraceOp::SetLocal { slot, span } => {
                    let value = pop_int(&mut trace_stack, span)?;
                    write_int_local(stack, stack_base, slot, value, span)?;
                }
                TraceOp::IntBinary { kind, span } => {
                    let right = pop_int(&mut trace_stack, span)?;
                    let left = pop_int(&mut trace_stack, span)?;
                    let value = checked_binary(kind, left, right, span)?;
                    trace_stack.push(TraceValue::Int(value));
                }
                TraceOp::IntCompare { kind, span } => {
                    let right = pop_int(&mut trace_stack, span)?;
                    let left = pop_int(&mut trace_stack, span)?;
                    trace_stack.push(TraceValue::Bool(compare_int(kind, left, right)));
                }
                TraceOp::Pop { span } => {
                    trace_stack
                        .pop()
                        .ok_or_else(|| vm_error("stack underflow", span))?;
                }
                TraceOp::JumpIfFalse { target_ip, span } => {
                    let value = trace_stack
                        .last()
                        .copied()
                        .ok_or_else(|| vm_error("stack underflow", span))?;
                    match value {
                        TraceValue::Bool(false) => {
                            stack.push(Value::Bool(false));
                            return Ok(TraceRunResult::ExitToInterpreter { ip: target_ip });
                        }
                        TraceValue::Bool(true) => {}
                        TraceValue::Int(_) => {
                            return Err(vm_error("condition must be Bool, got Int", span));
                        }
                    }
                }
                TraceOp::Loop => return Ok(TraceRunResult::LoopHeader),
            }
        }
        Ok(TraceRunResult::ExitToInterpreter {
            ip: self.key.loop_header_ip,
        })
    }
}

#[derive(Debug, Clone, Copy)]
enum TraceRunResult {
    LoopHeader,
    ExitToInterpreter { ip: usize },
}

#[derive(Debug, Clone, Copy)]
enum TraceOp {
    ConstInt(i64),
    ConstBool(bool),
    GetLocal { slot: usize, span: Span },
    SetLocal { slot: usize, span: Span },
    IntBinary { kind: IntBinaryOp, span: Span },
    IntCompare { kind: IntCompareOp, span: Span },
    Pop { span: Span },
    JumpIfFalse { target_ip: usize, span: Span },
    Loop,
}

#[derive(Debug, Clone, Copy)]
enum TraceValue {
    Int(i64),
    Bool(bool),
}

#[derive(Debug, Clone, Copy)]
enum IntBinaryOp {
    Add,
    Subtract,
    Multiply,
    Divide,
}

#[derive(Debug, Clone, Copy)]
enum IntCompareOp {
    Equal,
    Greater,
    Less,
}

#[derive(Debug)]
struct TraceBuildError;

fn remember_slot(slots: &mut Vec<usize>, slot: usize) {
    if !slots.contains(&slot) {
        slots.push(slot);
    }
}

fn read_op(chunk: &Chunk, ip: &mut usize) -> Result<OpCode, TraceBuildError> {
    let byte = *chunk.code.get(*ip).ok_or(TraceBuildError)?;
    *ip += 1;
    OpCode::from_byte(byte).ok_or(TraceBuildError)
}

fn read_u16(chunk: &Chunk, ip: &mut usize) -> Result<u16, TraceBuildError> {
    let low = *chunk.code.get(*ip).ok_or(TraceBuildError)?;
    let high = *chunk.code.get(*ip + 1).ok_or(TraceBuildError)?;
    *ip += 2;
    Ok(u16::from_le_bytes([low, high]))
}

fn read_int_local(stack: &[Value], stack_base: usize, slot: usize, span: Span) -> VmResult<i64> {
    match stack.get(stack_base + slot) {
        Some(Value::Int(value)) => Ok(*value),
        Some(other) => Err(vm_error(
            format!("trace expected Int local, got {}", other.kind_name()),
            span,
        )),
        None => Err(vm_error("invalid local slot", span)),
    }
}

fn write_int_local(
    stack: &mut [Value],
    stack_base: usize,
    slot: usize,
    value: i64,
    span: Span,
) -> VmResult<()> {
    let Some(local) = stack.get_mut(stack_base + slot) else {
        return Err(vm_error("invalid local slot", span));
    };
    *local = Value::Int(value);
    Ok(())
}

fn pop_int(stack: &mut Vec<TraceValue>, span: Span) -> VmResult<i64> {
    match stack.pop() {
        Some(TraceValue::Int(value)) => Ok(value),
        Some(TraceValue::Bool(_)) => Err(vm_error("trace expected Int, got Bool", span)),
        None => Err(vm_error("stack underflow", span)),
    }
}

fn checked_binary(kind: IntBinaryOp, left: i64, right: i64, span: Span) -> VmResult<i64> {
    match kind {
        IntBinaryOp::Add => left
            .checked_add(right)
            .ok_or_else(|| vm_error("integer overflow in addition", span)),
        IntBinaryOp::Subtract => left
            .checked_sub(right)
            .ok_or_else(|| vm_error("integer overflow in subtraction", span)),
        IntBinaryOp::Multiply => left
            .checked_mul(right)
            .ok_or_else(|| vm_error("integer overflow in multiplication", span)),
        IntBinaryOp::Divide => {
            if right == 0 {
                return Err(vm_error("division by zero", span));
            }
            left.checked_div(right)
                .ok_or_else(|| vm_error("integer overflow in division", span))
        }
    }
}

fn compare_int(kind: IntCompareOp, left: i64, right: i64) -> bool {
    match kind {
        IntCompareOp::Equal => left == right,
        IntCompareOp::Greater => left > right,
        IntCompareOp::Less => left < right,
    }
}

#[cfg(feature = "jit")]
#[repr(C)]
struct NativeTraceState {
    locals: *mut i64,
    error_ip: usize,
}

#[cfg(feature = "jit")]
struct NativeTrace {
    _jit: cranelift_jit::JITModule,
    function: unsafe extern "C" fn(*mut NativeTraceState) -> i32,
}

#[cfg(feature = "jit")]
impl NativeTrace {
    fn compile(trace: &Trace) -> Result<Self, String> {
        use cranelift::prelude::*;
        use cranelift_jit::{JITBuilder, JITModule};
        use cranelift_module::{Linkage, Module};

        if trace.ops.iter().any(|op| {
            matches!(
                op,
                TraceOp::IntBinary {
                    kind: IntBinaryOp::Divide,
                    ..
                }
            )
        }) {
            return Err("native division traces are not enabled yet".to_string());
        }

        let mut flag_builder = settings::builder();
        flag_builder
            .set("use_colocated_libcalls", "false")
            .map_err(|error| error.to_string())?;
        flag_builder
            .set("is_pic", "false")
            .map_err(|error| error.to_string())?;
        let isa_builder = cranelift_native::builder().map_err(|error| error.to_string())?;
        let isa = isa_builder
            .finish(settings::Flags::new(flag_builder))
            .map_err(|error| error.to_string())?;
        let builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
        let mut module = JITModule::new(builder);
        let pointer_type = module.target_config().pointer_type();
        let mut ctx = module.make_context();
        ctx.func.signature.params.push(AbiParam::new(pointer_type));
        ctx.func.signature.returns.push(AbiParam::new(types::I32));

        let mut builder_context = FunctionBuilderContext::new();
        let mut fb = FunctionBuilder::new(&mut ctx.func, &mut builder_context);
        let entry = fb.create_block();
        fb.append_block_params_for_function_params(entry);
        fb.switch_to_block(entry);
        fb.seal_block(entry);

        let bailout_block = fb.create_block();

        let state = fb.block_params(entry)[0];
        let locals = fb.ins().load(pointer_type, MemFlags::trusted(), state, 0);
        let error_ip_offset = std::mem::offset_of!(NativeTraceState, error_ip) as i64;
        let error_ip_ptr = fb.ins().iadd_imm(state, error_ip_offset);
        let mut stack = Vec::new();
        for op in &trace.ops {
            match *op {
                TraceOp::ConstInt(value) => stack.push(fb.ins().iconst(types::I64, value)),
                TraceOp::ConstBool(value) => {
                    stack.push(fb.ins().iconst(types::I64, i64::from(value as u8)))
                }
                TraceOp::GetLocal { slot, .. } => {
                    let offset = (slot * std::mem::size_of::<i64>()) as i32;
                    stack.push(
                        fb.ins()
                            .load(types::I64, MemFlags::trusted(), locals, offset),
                    );
                }
                TraceOp::SetLocal { slot, .. } => {
                    let value = stack
                        .pop()
                        .ok_or_else(|| "native stack underflow".to_string())?;
                    let offset = (slot * std::mem::size_of::<i64>()) as i32;
                    fb.ins().store(MemFlags::trusted(), value, locals, offset);
                }
                TraceOp::IntBinary { kind, span: _ } => {
                    let right = stack
                        .pop()
                        .ok_or_else(|| "native stack underflow".to_string())?;
                    let left = stack
                        .pop()
                        .ok_or_else(|| "native stack underflow".to_string())?;
                    let result = match kind {
                        IntBinaryOp::Add => {
                            let (value, overflow) = fb.ins().sadd_overflow(left, right);
                            let continue_block = fb.create_block();
                            fb.ins()
                                .brif(overflow, bailout_block, &[], continue_block, &[]);
                            fb.switch_to_block(continue_block);
                            fb.seal_block(continue_block);
                            value
                        }
                        IntBinaryOp::Subtract => {
                            let (value, overflow) = fb.ins().ssub_overflow(left, right);
                            let continue_block = fb.create_block();
                            fb.ins()
                                .brif(overflow, bailout_block, &[], continue_block, &[]);
                            fb.switch_to_block(continue_block);
                            fb.seal_block(continue_block);
                            value
                        }
                        IntBinaryOp::Multiply => {
                            let (value, overflow) = fb.ins().smul_overflow(left, right);
                            let continue_block = fb.create_block();
                            fb.ins()
                                .brif(overflow, bailout_block, &[], continue_block, &[]);
                            fb.switch_to_block(continue_block);
                            fb.seal_block(continue_block);
                            value
                        }
                        IntBinaryOp::Divide => {
                            return Err("native division unsupported".to_string());
                        }
                    };
                    stack.push(result);
                }
                TraceOp::IntCompare { kind, .. } => {
                    let right = stack
                        .pop()
                        .ok_or_else(|| "native stack underflow".to_string())?;
                    let left = stack
                        .pop()
                        .ok_or_else(|| "native stack underflow".to_string())?;
                    let cc = match kind {
                        IntCompareOp::Equal => IntCC::Equal,
                        IntCompareOp::Greater => IntCC::SignedGreaterThan,
                        IntCompareOp::Less => IntCC::SignedLessThan,
                    };
                    let result = fb.ins().icmp(cc, left, right);
                    let one = fb.ins().iconst(types::I64, 1);
                    let zero = fb.ins().iconst(types::I64, 0);
                    stack.push(fb.ins().select(result, one, zero));
                }
                TraceOp::Pop { .. } => {
                    stack
                        .pop()
                        .ok_or_else(|| "native stack underflow".to_string())?;
                }
                TraceOp::JumpIfFalse { target_ip, .. } => {
                    let condition = *stack
                        .last()
                        .ok_or_else(|| "native stack underflow".to_string())?;
                    let is_false = fb.ins().icmp_imm(IntCC::Equal, condition, 0);
                    let continue_block = fb.create_block();
                    let exit_block = fb.create_block();
                    fb.ins()
                        .brif(is_false, exit_block, &[], continue_block, &[]);
                    fb.switch_to_block(exit_block);
                    let exit_target = fb.ins().iconst(pointer_type, target_ip as i64);
                    fb.ins()
                        .store(MemFlags::trusted(), exit_target, error_ip_ptr, 0);
                    let exit_code = fb.ins().iconst(types::I32, 1);
                    fb.ins().return_(&[exit_code]);
                    fb.seal_block(exit_block);
                    fb.switch_to_block(continue_block);
                    fb.seal_block(continue_block);
                }
                TraceOp::Loop => {
                    let continue_code = fb.ins().iconst(types::I32, 0);
                    fb.ins().return_(&[continue_code]);
                }
            }
        }
        fb.switch_to_block(bailout_block);
        let bailout_code = fb.ins().iconst(types::I32, 2);
        fb.ins().return_(&[bailout_code]);
        fb.seal_block(bailout_block);
        fb.finalize();

        let id = module
            .declare_function("muninn_trace", Linkage::Export, &ctx.func.signature)
            .map_err(|error| error.to_string())?;
        module
            .define_function(id, &mut ctx)
            .map_err(|error| error.to_string())?;
        module.clear_context(&mut ctx);
        module
            .finalize_definitions()
            .map_err(|error| error.to_string())?;
        let code = module.get_finalized_function(id);
        // SAFETY: Cranelift emitted code for the exact `extern "C" fn(*mut NativeTraceState) -> i32`
        // signature declared above, and the owning JITModule is stored in NativeTrace so the code stays alive.
        let function = unsafe {
            std::mem::transmute::<*const u8, unsafe extern "C" fn(*mut NativeTraceState) -> i32>(
                code,
            )
        };
        Ok(Self {
            _jit: module,
            function,
        })
    }

    fn run(
        &self,
        trace: &Trace,
        module: &BytecodeModule,
        stack: &mut Vec<Value>,
        stack_base: usize,
    ) -> (bool, VmResult<TraceRunResult>) {
        let mut locals = Vec::with_capacity(trace.local_slots.len().max(1));
        let max_slot = trace.local_slots.iter().copied().max().unwrap_or(0);
        locals.resize(max_slot + 1, 0i64);
        for slot in &trace.local_slots {
            let value = match read_int_local(stack, stack_base, *slot, trace.loop_span) {
                Ok(value) => value,
                Err(error) => return (false, Err(error)),
            };
            locals[*slot] = value;
        }
        let mut state = NativeTraceState {
            locals: locals.as_mut_ptr(),
            error_ip: trace.key.loop_header_ip,
        };
        // SAFETY: The function pointer is produced by `compile` for this exact state layout. `locals`
        // remains alive and mutable for the duration of the call, and traces are only run synchronously.
        let code = unsafe { (self.function)(&mut state) };
        match code {
            0 => {
                for slot in &trace.local_slots {
                    if let Err(error) =
                        write_int_local(stack, stack_base, *slot, locals[*slot], trace.loop_span)
                    {
                        return (false, Err(error));
                    }
                }
                (false, Ok(TraceRunResult::LoopHeader))
            }
            1 => {
                for slot in &trace.local_slots {
                    if let Err(error) =
                        write_int_local(stack, stack_base, *slot, locals[*slot], trace.loop_span)
                    {
                        return (false, Err(error));
                    }
                }
                stack.push(Value::Bool(false));
                (
                    false,
                    Ok(TraceRunResult::ExitToInterpreter { ip: state.error_ip }),
                )
            }
            2 => (true, trace.run_interpreted(module, stack, stack_base)),
            other => (
                true,
                Err(unknown_native_return_code_error(other, trace.loop_span)),
            ),
        }
    }
}

#[cfg(feature = "jit")]
fn unknown_native_return_code_error(code: i32, span: Span) -> crate::error::MuninnError {
    vm_error(format!("unknown native trace return code {code}"), span)
}

#[cfg(all(test, feature = "jit"))]
mod native_tests {
    use super::unknown_native_return_code_error;
    use crate::span::Span;

    #[test]
    fn unknown_native_return_code_is_vm_error() {
        let span = Span::range(2, 3, 4, 2, 8, 9);
        let error = unknown_native_return_code_error(99, span);

        assert_eq!(error.phase, "vm");
        assert_eq!(error.message, "unknown native trace return code 99");
        assert_eq!(error.span, span);
    }
}
