use std::collections::HashMap;
use std::sync::Arc;

use crate::bytecode::{BytecodeModule, Chunk, Constant, GlobalValueKind, OpCode, validate_module};
use crate::error::MuninnError;
use crate::native::{
    add_values, divide_values, invoke_native, multiply_values, registered_natives,
    subtract_values,
};
use crate::runtime::{VmError, VmResult};
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
    stack: Vec<Value>,
    frames: Vec<CallFrame>,
    started: bool,
    pending_reload: Option<BytecodeModule>,
    preserve_existing_globals: bool,
}

#[derive(Debug, Clone, Copy)]
struct CallFrame {
    function_id: usize,
    ip: usize,
    stack_base: usize,
}

impl Vm {
    pub fn new(module: BytecodeModule) -> Self {
        let mut vm = Self {
            globals: HashMap::new(),
            stack: Vec::new(),
            frames: Vec::new(),
            started: false,
            pending_reload: None,
            preserve_existing_globals: false,
            module,
        };
        vm.install_natives();
        vm.reserve_runtime_capacity(
            vm.module.estimated_stack_capacity(),
            vm.module.estimated_frame_capacity(),
        );
        vm
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
        self.ensure_started(Span::default())?;
        self.execute_instruction()
    }

    pub fn request_reload(&mut self, module: BytecodeModule) -> VmResult<()> {
        validate_module(&module).map_err(first_validation_error)?;
        self.pending_reload = Some(module);
        Ok(())
    }

    pub fn poll_safe_point(&self) -> ReloadStatus {
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
            return Err(VmError::new(
                "reload is only allowed at a safe point",
                Span::default(),
            ));
        }

        let pending = self.pending_reload.take().expect("pending reload");
        if let Err(error) = self.validate_reload_compatibility(&pending) {
            return Err(error);
        }

        self.module = pending;
        self.started = false;
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
        self.push_frame(self.module.entry_function, 0, span)?;
        self.started = true;
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
        let span = self.current_chunk(function_id).span_at(ip);
        let byte = *self
            .current_chunk(function_id)
            .code
            .get(ip)
            .ok_or_else(|| VmError::new("instruction pointer out of range", span))?;
        let op = OpCode::from_byte(byte)
            .ok_or_else(|| VmError::new(format!("invalid opcode {}", byte), span))?;
        self.frames[frame_index].ip += 1;

        match op {
            OpCode::Constant => {
                let index = self.read_u16(frame_index, span)? as usize;
                let constant = self
                    .current_chunk(function_id)
                    .constants
                    .get(index)
                    .ok_or_else(|| VmError::new(format!("invalid constant index {}", index), span))?;
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
                    self.globals.insert(name, value);
                }
            }
            OpCode::GetGlobal => {
                let name = self.read_name(frame_index, span)?;
                let value = self
                    .globals
                    .get(&name)
                    .cloned()
                    .ok_or_else(|| VmError::new(format!("unknown global '{}'", name), span))?;
                self.stack.push(value);
            }
            OpCode::SetGlobal => {
                let name = self.read_name(frame_index, span)?;
                let value = self.pop(span)?;
                if !self.globals.contains_key(&name) {
                    return Err(VmError::new(format!("unknown global '{}'", name), span));
                }
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
                            .ok_or_else(|| VmError::new("integer overflow in negation", span))?;
                        self.stack.push(Value::Int(negated));
                    }
                    Value::Float(value) => self.stack.push(Value::Float(-value)),
                    other => {
                        return Err(VmError::new(
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
                        return Err(VmError::new(
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
                    .ok_or_else(|| VmError::new("stack underflow", span))?;
                match condition {
                    Value::Bool(value) => {
                        if !value {
                            self.frames[frame_index].ip += jump;
                        }
                    }
                    other => {
                        return Err(VmError::new(
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
                self.frames[frame_index].ip = self.frames[frame_index].ip.saturating_sub(jump);
            }
            OpCode::Call => {
                let arg_count = self.read_u8(frame_index, span)? as usize;
                self.call_value(arg_count, span)?;
            }
            OpCode::Return => {
                let value = self.stack.pop().unwrap_or(Value::Nil);
                let function = &self.module.functions[function_id];
                if function.expects_return_value && matches!(value, Value::Nil) {
                    return Err(VmError::new(
                        format!(
                            "function '{}' fell through without returning a value",
                            function.name
                        ),
                        span,
                    ));
                }

                let frame = self.frames.pop().expect("frame");
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

    fn validate_reload_compatibility(&self, next_module: &BytecodeModule) -> VmResult<()> {
        if next_module.entry_function >= next_module.functions.len() {
            return Err(VmError::new(
                "reload module is missing a valid entry function",
                Span::default(),
            ));
        }

        for (name, value) in &self.globals {
            if matches!(value, Value::Native(_)) {
                continue;
            }

            let Some(expected_kind) = next_module.global_kind(name) else {
                return Err(VmError::new(
                    format!("reload rejected: global '{}' is missing in new module", name),
                    Span::default(),
                ));
            };

            if !value_matches_kind(value, expected_kind) {
                return Err(VmError::new(
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
            .ok_or_else(|| VmError::new("instruction pointer out of range", span))?;
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
            .ok_or_else(|| VmError::new(format!("invalid constant index {}", index), span))?;
        if let Constant::String(name) = constant {
            Ok(name.clone())
        } else {
            Err(VmError::new("expected string constant", span))
        }
    }

    fn local_stack_index(&self, frame_index: usize, slot: usize, span: Span) -> VmResult<usize> {
        let function_id = self.frames[frame_index].function_id;
        let local_count = self.module.functions[function_id].local_count;
        if slot >= local_count {
            return Err(VmError::new(format!("invalid local slot {}", slot), span));
        }
        Ok(self.frames[frame_index].stack_base + slot)
    }

    fn call_value(&mut self, arg_count: usize, span: Span) -> VmResult<()> {
        if self.stack.len() < arg_count + 1 {
            return Err(VmError::new("stack underflow", span));
        }

        let callee_index = self.stack.len() - arg_count - 1;
        let callee = self.stack[callee_index].clone();
        match callee {
            Value::Function(function_id) => {
                self.stack.remove(callee_index);
                self.push_frame(function_id, arg_count, span)
            }
            Value::Native(kind) => {
                let result = invoke_native(kind, &self.stack[callee_index + 1..], span)?;
                self.stack.truncate(callee_index);
                self.stack.push(result);
                Ok(())
            }
            other => Err(VmError::new(
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
            .ok_or_else(|| VmError::new(format!("invalid function id {}", function_id), span))?;
        if function.arity != arg_count {
            return Err(VmError::new(
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
                    .ok_or_else(|| VmError::new("cannot compare NaN values", span))?;
                self.stack.push(Value::Bool(predicate(ordering)));
                Ok(())
            }
            (left, right) => Err(VmError::new(
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
            .ok_or_else(|| VmError::new("stack underflow", span))
    }
}

fn value_matches_kind(value: &Value, kind: GlobalValueKind) -> bool {
    match (value, kind) {
        (Value::Int(_), GlobalValueKind::Int)
        | (Value::Float(_), GlobalValueKind::Float)
        | (Value::Bool(_), GlobalValueKind::Bool)
        | (Value::String(_), GlobalValueKind::String)
        | (Value::Tensor(_), GlobalValueKind::Tensor)
        | (Value::Function(_), GlobalValueKind::Function) => true,
        _ => false,
    }
}

fn first_validation_error(errors: Vec<MuninnError>) -> VmError {
    let first = errors
        .into_iter()
        .next()
        .unwrap_or_else(|| MuninnError::new("compiler", "invalid bytecode module", Span::default()));
    VmError::new(first.message, first.span)
}
