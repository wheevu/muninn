use std::collections::HashMap;
use std::fmt::{Display, Formatter};

use crate::builtins::{BuiltinKind, BUILTINS};
use crate::bytecode::{BytecodeModule, Chunk, Constant, OpCode};
use crate::span::Span;

pub type VmResult<T> = Result<T, VmError>;

type NativeFn = fn(&[Value], Span) -> VmResult<Value>;

#[derive(Debug, Clone)]
pub struct VmError {
    pub message: String,
    pub span: Span,
}

impl VmError {
    fn new(message: impl Into<String>, span: Span) -> Self {
        Self {
            message: message.into(),
            span,
        }
    }
}

#[derive(Debug, Clone)]
pub enum Value {
    Int(i64),
    Float(f64),
    Bool(bool),
    String(String),
    Function(usize),
    Native(BuiltinKind, NativeFn),
    Nil,
}

impl Value {
    pub fn stringify(&self) -> String {
        match self {
            Value::Int(value) => value.to_string(),
            Value::Float(value) => {
                if value.fract() == 0.0 {
                    format!("{value:.1}")
                } else {
                    value.to_string()
                }
            }
            Value::Bool(value) => value.to_string(),
            Value::String(value) => value.clone(),
            Value::Function(_) => "<fn>".to_string(),
            Value::Native(_, _) => "<builtin>".to_string(),
            Value::Nil => "nil".to_string(),
        }
    }

    fn equals(&self, other: &Value) -> bool {
        match (self, other) {
            (Value::Int(left), Value::Int(right)) => left == right,
            (Value::Float(left), Value::Float(right)) => left == right,
            (Value::Bool(left), Value::Bool(right)) => left == right,
            (Value::String(left), Value::String(right)) => left == right,
            (Value::Nil, Value::Nil) => true,
            _ => false,
        }
    }
}

impl Display for Value {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.stringify())
    }
}

pub struct Vm {
    module: BytecodeModule,
    globals: HashMap<String, Value>,
    stack: Vec<Value>,
    frames: Vec<CallFrame>,
}

struct CallFrame {
    function_id: usize,
    ip: usize,
    locals: Vec<Value>,
}

impl Vm {
    pub fn new(module: BytecodeModule) -> Self {
        let mut vm = Self {
            module,
            globals: HashMap::new(),
            stack: Vec::new(),
            frames: Vec::new(),
        };
        vm.install_builtins();
        vm
    }

    pub fn run(&mut self) -> VmResult<Value> {
        self.stack.clear();
        self.frames.clear();
        self.push_frame(self.module.entry_function, Vec::new(), Span::default())?;
        self.execute_loop()
    }

    fn install_builtins(&mut self) {
        for spec in BUILTINS {
            let native = match spec.kind {
                BuiltinKind::Print => Value::Native(BuiltinKind::Print, native_print),
                BuiltinKind::Assert => Value::Native(BuiltinKind::Assert, native_assert),
            };
            self.globals.insert(spec.name.to_string(), native);
        }
    }

    fn execute_loop(&mut self) -> VmResult<Value> {
        loop {
            if self.frames.is_empty() {
                return Ok(self.stack.pop().unwrap_or(Value::Nil));
            }

            let frame_index = self.frames.len() - 1;
            let function_id = self.frames[frame_index].function_id;
            let function = self
                .module
                .functions
                .get(function_id)
                .ok_or_else(|| {
                    VmError::new(
                        format!("invalid function id {}", function_id),
                        Span::default(),
                    )
                })?
                .clone();
            let ip = self.frames[frame_index].ip;
            let chunk = &function.chunk;
            let span = chunk.span_at(ip);
            let byte = *chunk
                .code
                .get(ip)
                .ok_or_else(|| VmError::new("instruction pointer out of range", span))?;
            let op = OpCode::from_byte(byte)
                .ok_or_else(|| VmError::new(format!("invalid opcode {}", byte), span))?;
            self.frames[frame_index].ip += 1;

            match op {
                OpCode::Constant => {
                    let index = self.read_u16(frame_index, chunk, span)? as usize;
                    let constant = chunk.constants.get(index).ok_or_else(|| {
                        VmError::new(format!("invalid constant index {}", index), span)
                    })?;
                    self.stack.push(self.constant_to_value(constant));
                }
                OpCode::Nil => self.stack.push(Value::Nil),
                OpCode::True => self.stack.push(Value::Bool(true)),
                OpCode::False => self.stack.push(Value::Bool(false)),
                OpCode::Pop => {
                    self.stack.pop();
                }
                OpCode::GetLocal => {
                    let slot = self.read_u16(frame_index, chunk, span)? as usize;
                    let value = self.frames[frame_index]
                        .locals
                        .get(slot)
                        .cloned()
                        .ok_or_else(|| {
                            VmError::new(format!("invalid local slot {}", slot), span)
                        })?;
                    self.stack.push(value);
                }
                OpCode::SetLocal => {
                    let slot = self.read_u16(frame_index, chunk, span)? as usize;
                    let value = self
                        .stack
                        .pop()
                        .ok_or_else(|| VmError::new("stack underflow", span))?;
                    if self.frames[frame_index].locals.len() <= slot {
                        self.frames[frame_index].locals.resize(slot + 1, Value::Nil);
                    }
                    self.frames[frame_index].locals[slot] = value;
                }
                OpCode::DefineGlobal => {
                    let name = self.read_name(frame_index, chunk, span)?;
                    let value = self
                        .stack
                        .pop()
                        .ok_or_else(|| VmError::new("stack underflow", span))?;
                    self.globals.insert(name, value);
                }
                OpCode::GetGlobal => {
                    let name = self.read_name(frame_index, chunk, span)?;
                    let value =
                        self.globals.get(&name).cloned().ok_or_else(|| {
                            VmError::new(format!("unknown global '{}'", name), span)
                        })?;
                    self.stack.push(value);
                }
                OpCode::SetGlobal => {
                    let name = self.read_name(frame_index, chunk, span)?;
                    let value = self
                        .stack
                        .pop()
                        .ok_or_else(|| VmError::new("stack underflow", span))?;
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
                    self.numeric_binary(span, IntNumericOp::Subtract, |left, right| left - right)?
                }
                OpCode::Multiply => {
                    self.numeric_binary(span, IntNumericOp::Multiply, |left, right| left * right)?
                }
                OpCode::Divide => {
                    self.numeric_binary(span, IntNumericOp::Divide, |left, right| left / right)?
                }
                OpCode::Negate => {
                    let value = self.pop(span)?;
                    match value {
                        Value::Int(value) => {
                            let negated = value.checked_neg().ok_or_else(|| {
                                VmError::new("integer overflow in negation", span)
                            })?;
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
                    let jump = self.read_u16(frame_index, chunk, span)? as usize;
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
                    let jump = self.read_u16(frame_index, chunk, span)? as usize;
                    self.frames[frame_index].ip += jump;
                }
                OpCode::Loop => {
                    let jump = self.read_u16(frame_index, chunk, span)? as usize;
                    self.frames[frame_index].ip = self.frames[frame_index].ip.saturating_sub(jump);
                }
                OpCode::Call => {
                    let arg_count = self.read_u8(frame_index, chunk, span)? as usize;
                    let mut args = Vec::with_capacity(arg_count);
                    for _ in 0..arg_count {
                        args.push(self.pop(span)?);
                    }
                    args.reverse();
                    let callee = self.pop(span)?;
                    match callee {
                        Value::Function(function_id) => {
                            self.push_frame(function_id, args, span)?;
                        }
                        Value::Native(_, function) => {
                            let value = function(&args, span)?;
                            self.stack.push(value);
                        }
                        other => {
                            return Err(VmError::new(
                                format!("{} is not callable", other.stringify()),
                                span,
                            ));
                        }
                    }
                }
                OpCode::Return => {
                    let value = self.stack.pop().unwrap_or(Value::Nil);
                    if function.expects_return_value && matches!(value, Value::Nil) {
                        return Err(VmError::new(
                            format!(
                                "function '{}' fell through without returning a value",
                                function.name
                            ),
                            span,
                        ));
                    }
                    self.frames.pop();
                    if self.frames.is_empty() {
                        return Ok(value);
                    }
                    self.stack.push(value);
                }
            }
        }
    }

    fn push_frame(&mut self, function_id: usize, args: Vec<Value>, span: Span) -> VmResult<()> {
        let function =
            self.module.functions.get(function_id).ok_or_else(|| {
                VmError::new(format!("invalid function id {}", function_id), span)
            })?;
        if function.arity != args.len() {
            return Err(VmError::new(
                format!(
                    "function '{}' expects {} arguments, got {}",
                    function.name,
                    function.arity,
                    args.len()
                ),
                span,
            ));
        }
        let mut locals = vec![Value::Nil; function.local_count.max(args.len())];
        for (slot, arg) in args.into_iter().enumerate() {
            locals[slot] = arg;
        }
        self.frames.push(CallFrame {
            function_id,
            ip: 0,
            locals,
        });
        Ok(())
    }

    fn numeric_binary(
        &mut self,
        span: Span,
        int_op: IntNumericOp,
        float_op: impl FnOnce(f64, f64) -> f64,
    ) -> VmResult<()> {
        let right = self.pop(span)?;
        let left = self.pop(span)?;
        match (left, right) {
            (Value::Int(left), Value::Int(right)) => {
                let value = match int_op {
                    IntNumericOp::Subtract => left
                        .checked_sub(right)
                        .ok_or_else(|| VmError::new("integer overflow in subtraction", span))?,
                    IntNumericOp::Multiply => left
                        .checked_mul(right)
                        .ok_or_else(|| VmError::new("integer overflow in multiplication", span))?,
                    IntNumericOp::Divide => {
                        if right == 0 {
                            return Err(VmError::new("division by zero", span));
                        }
                        left.checked_div(right)
                            .ok_or_else(|| VmError::new("integer overflow in division", span))?
                    }
                };
                self.stack.push(Value::Int(value));
                Ok(())
            }
            (Value::Float(left), Value::Float(right)) => {
                self.stack.push(Value::Float(float_op(left, right)));
                Ok(())
            }
            (left, right) => Err(VmError::new(
                format!(
                    "numeric operation expects matching numeric types, got {} and {}",
                    left.stringify(),
                    right.stringify()
                ),
                span,
            )),
        }
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

    fn read_u8(&mut self, frame_index: usize, chunk: &Chunk, span: Span) -> VmResult<u8> {
        let ip = self.frames[frame_index].ip;
        let byte = *chunk
            .code
            .get(ip)
            .ok_or_else(|| VmError::new("instruction pointer out of range", span))?;
        self.frames[frame_index].ip += 1;
        Ok(byte)
    }

    fn read_u16(&mut self, frame_index: usize, chunk: &Chunk, span: Span) -> VmResult<u16> {
        let low = self.read_u8(frame_index, chunk, span)?;
        let high = self.read_u8(frame_index, chunk, span)?;
        Ok(u16::from_le_bytes([low, high]))
    }

    fn read_name(&mut self, frame_index: usize, chunk: &Chunk, span: Span) -> VmResult<String> {
        let index = self.read_u16(frame_index, chunk, span)? as usize;
        let constant = chunk
            .constants
            .get(index)
            .ok_or_else(|| VmError::new(format!("invalid constant index {}", index), span))?;
        if let Constant::String(name) = constant {
            Ok(name.clone())
        } else {
            Err(VmError::new("expected string constant", span))
        }
    }

    fn constant_to_value(&self, constant: &Constant) -> Value {
        match constant {
            Constant::Int(value) => Value::Int(*value),
            Constant::Float(value) => Value::Float(*value),
            Constant::Bool(value) => Value::Bool(*value),
            Constant::String(value) => Value::String(value.clone()),
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

#[derive(Debug, Clone, Copy)]
enum IntNumericOp {
    Subtract,
    Multiply,
    Divide,
}

fn add_values(left: Value, right: Value, span: Span) -> VmResult<Value> {
    match (left, right) {
        (Value::Int(left), Value::Int(right)) => {
            let value = left
                .checked_add(right)
                .ok_or_else(|| VmError::new("integer overflow in addition", span))?;
            Ok(Value::Int(value))
        }
        (Value::Float(left), Value::Float(right)) => Ok(Value::Float(left + right)),
        (Value::String(left), Value::String(right)) => Ok(Value::String(left + &right)),
        (left, right) => Err(VmError::new(
            format!(
                "'+' expects matching Int, Float, or String operands, got {} and {}",
                left.stringify(),
                right.stringify()
            ),
            span,
        )),
    }
}

fn native_print(args: &[Value], span: Span) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::new("print expects exactly 1 argument", span));
    }
    println!("{}", args[0]);
    Ok(Value::Nil)
}

fn native_assert(args: &[Value], span: Span) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::new("assert expects exactly 1 argument", span));
    }
    match args[0] {
        Value::Bool(true) => Ok(Value::Nil),
        Value::Bool(false) => Err(VmError::new("assertion failed", span)),
        ref other => Err(VmError::new(
            format!("assert expects Bool, got {}", other.stringify()),
            span,
        )),
    }
}
