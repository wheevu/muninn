use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::fmt::{Display, Formatter};
use std::rc::Rc;

use crate::bytecode::{BytecodeModule, Chunk, Constant, OpCode};

pub type VmResult<T> = Result<T, String>;

type NativeFn = fn(&[Value]) -> VmResult<Value>;

#[derive(Clone)]
pub enum Value {
    Int(i64),
    Float(f64),
    Bool(bool),
    String(Rc<String>),
    Array(Rc<RefCell<Vec<Value>>>),
    Function(usize),
    Class(usize),
    Instance(Rc<RefCell<InstanceObj>>),
    BoundMethod {
        receiver: Rc<RefCell<InstanceObj>>,
        function_id: usize,
    },
    Native(NativeFn),
    Some(Box<Value>),
    None,
    Nil,
}

impl Value {
    fn truthy(&self) -> bool {
        match self {
            Value::Bool(v) => *v,
            Value::None => false,
            Value::Nil => false,
            _ => true,
        }
    }

    pub fn stringify(&self) -> String {
        match self {
            Value::Int(v) => v.to_string(),
            Value::Float(v) => {
                if v.fract() == 0.0 {
                    format!("{:.1}", v)
                } else {
                    v.to_string()
                }
            }
            Value::Bool(v) => v.to_string(),
            Value::String(v) => (**v).clone(),
            Value::Array(items) => {
                let values = items
                    .borrow()
                    .iter()
                    .map(Value::stringify)
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("[{}]", values)
            }
            Value::Function(_) => "<fn>".to_string(),
            Value::Class(_) => "<class>".to_string(),
            Value::Instance(instance) => {
                let class_id = instance.borrow().class_id;
                format!("<instance:{}>", class_id)
            }
            Value::BoundMethod { .. } => "<bound-method>".to_string(),
            Value::Native(_) => "<native>".to_string(),
            Value::Some(inner) => format!("Some({})", inner.stringify()),
            Value::None => "None".to_string(),
            Value::Nil => "nil".to_string(),
        }
    }

    fn equals(&self, other: &Value) -> bool {
        match (self, other) {
            (Value::Int(a), Value::Int(b)) => a == b,
            (Value::Float(a), Value::Float(b)) => a == b,
            (Value::Bool(a), Value::Bool(b)) => a == b,
            (Value::String(a), Value::String(b)) => a == b,
            (Value::Some(a), Value::Some(b)) => a.equals(b),
            (Value::None, Value::None) => true,
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

#[derive(Clone)]
pub struct InstanceObj {
    pub class_id: usize,
    pub fields: HashMap<String, Value>,
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
    kind: FrameKind,
}

enum FrameKind {
    Regular,
    Initializer { instance: Rc<RefCell<InstanceObj>> },
}

impl Vm {
    pub fn new(module: BytecodeModule) -> Self {
        let mut vm = Self {
            module,
            globals: HashMap::new(),
            stack: Vec::new(),
            frames: Vec::new(),
        };
        vm.install_natives();
        vm
    }

    pub fn run(&mut self) -> VmResult<Value> {
        self.stack.clear();
        self.frames.clear();
        self.push_frame(self.module.entry_function, Vec::new(), FrameKind::Regular)?;
        self.execute_loop()
    }

    fn install_natives(&mut self) {
        self.globals
            .insert("to_string".to_string(), Value::Native(native_to_string));
        self.globals
            .insert("print".to_string(), Value::Native(native_print));
        self.globals
            .insert("len".to_string(), Value::Native(native_len));
        self.globals
            .insert("sum".to_string(), Value::Native(native_sum));
        self.globals
            .insert("dot".to_string(), Value::Native(native_dot));
        self.globals
            .insert("zeros".to_string(), Value::Native(native_zeros));
        self.globals
            .insert("ones".to_string(), Value::Native(native_ones));
        self.globals
            .insert("some".to_string(), Value::Native(native_some));
        self.globals
            .insert("is_none".to_string(), Value::Native(native_is_none));
        self.globals
            .insert("unwrap".to_string(), Value::Native(native_unwrap));
        self.globals.insert("none".to_string(), Value::None);
        self.globals
            .insert("__some".to_string(), Value::Native(native_some));
        self.globals
            .insert("__is_none".to_string(), Value::Native(native_is_none));
        self.globals
            .insert("__unwrap".to_string(), Value::Native(native_unwrap));
        self.globals.insert("__none".to_string(), Value::None);
    }

    fn execute_loop(&mut self) -> VmResult<Value> {
        loop {
            if self.frames.is_empty() {
                return Ok(self.stack.pop().unwrap_or(Value::Nil));
            }

            let frame_idx = self.frames.len() - 1;
            let function = {
                let frame = &self.frames[frame_idx];
                self.module
                    .functions
                    .get(frame.function_id)
                    .ok_or_else(|| format!("invalid function id {}", frame.function_id))?
                    .clone()
            };
            let ip = self.frames[frame_idx].ip;
            let function_name = function.name.clone();
            let chunk = &function.chunk;

            let op = OpCode::from_byte(*chunk.code.get(ip).ok_or_else(|| {
                format!("instruction pointer out of range in '{}'", function_name)
            })?)
            .ok_or_else(|| format!("invalid opcode at {}", ip))?;
            self.frames[frame_idx].ip += 1;

            match op {
                OpCode::Constant => {
                    let idx = self.read_u16_from_frame(frame_idx, chunk)? as usize;
                    let constant = chunk
                        .constants
                        .get(idx)
                        .ok_or_else(|| format!("invalid constant index {}", idx))?
                        .clone();
                    self.stack.push(self.constant_to_value(constant));
                }
                OpCode::Nil => self.stack.push(Value::Nil),
                OpCode::True => self.stack.push(Value::Bool(true)),
                OpCode::False => self.stack.push(Value::Bool(false)),
                OpCode::Pop => {
                    self.stack.pop();
                }
                OpCode::GetLocal => {
                    let slot = self.read_u16_from_frame(frame_idx, chunk)? as usize;
                    let value = self
                        .frames
                        .get(frame_idx)
                        .and_then(|frame| frame.locals.get(slot))
                        .ok_or_else(|| format!("invalid local slot {}", slot))?
                        .clone();
                    self.stack.push(value);
                }
                OpCode::SetLocal => {
                    let slot = self.read_u16_from_frame(frame_idx, chunk)? as usize;
                    let value = self
                        .stack
                        .last()
                        .ok_or_else(|| "stack underflow on SetLocal".to_string())?
                        .clone();
                    if slot >= self.frames[frame_idx].locals.len() {
                        self.frames[frame_idx].locals.resize(slot + 1, Value::Nil);
                    }
                    self.frames[frame_idx].locals[slot] = value;
                }
                OpCode::DefineGlobal => {
                    let name = self.read_name_constant(chunk, frame_idx)?.to_string();
                    let value = self
                        .stack
                        .pop()
                        .ok_or_else(|| "stack underflow on DefineGlobal".to_string())?;
                    self.globals.insert(name, value);
                }
                OpCode::GetGlobal => {
                    let name = self.read_name_constant(chunk, frame_idx)?;
                    let value = self
                        .globals
                        .get(name)
                        .ok_or_else(|| format!("undefined global '{}'", name))?
                        .clone();
                    self.stack.push(value);
                }
                OpCode::SetGlobal => {
                    let name = self.read_name_constant(chunk, frame_idx)?;
                    let value = self
                        .stack
                        .last()
                        .ok_or_else(|| "stack underflow on SetGlobal".to_string())?
                        .clone();
                    if let Some(existing) = self.globals.get_mut(name) {
                        *existing = value;
                    } else {
                        self.globals.insert(name.to_string(), value);
                    }
                }
                OpCode::Add => {
                    let right = pop_stack(&mut self.stack, "Add")?;
                    let left = pop_stack(&mut self.stack, "Add")?;
                    self.stack.push(add_values(left, right)?);
                }
                OpCode::Subtract => {
                    let right = pop_stack(&mut self.stack, "Subtract")?;
                    let left = pop_stack(&mut self.stack, "Subtract")?;
                    self.stack
                        .push(numeric_binary(left, right, |a, b| a - b, |a, b| a - b)?);
                }
                OpCode::Multiply => {
                    let right = pop_stack(&mut self.stack, "Multiply")?;
                    let left = pop_stack(&mut self.stack, "Multiply")?;
                    self.stack
                        .push(numeric_binary(left, right, |a, b| a * b, |a, b| a * b)?);
                }
                OpCode::Divide => {
                    let right = pop_stack(&mut self.stack, "Divide")?;
                    let left = pop_stack(&mut self.stack, "Divide")?;
                    self.stack
                        .push(numeric_binary(left, right, |a, b| a / b, |a, b| a / b)?);
                }
                OpCode::Negate => {
                    let value = pop_stack(&mut self.stack, "Negate")?;
                    self.stack.push(match value {
                        Value::Int(v) => Value::Int(-v),
                        Value::Float(v) => Value::Float(-v),
                        _ => return Err("negate expects Int or Float".to_string()),
                    });
                }
                OpCode::Not => {
                    let value = pop_stack(&mut self.stack, "Not")?;
                    self.stack.push(Value::Bool(!value.truthy()));
                }
                OpCode::Equal => {
                    let right = pop_stack(&mut self.stack, "Equal")?;
                    let left = pop_stack(&mut self.stack, "Equal")?;
                    self.stack.push(Value::Bool(left.equals(&right)));
                }
                OpCode::Greater => {
                    let right = pop_stack(&mut self.stack, "Greater")?;
                    let left = pop_stack(&mut self.stack, "Greater")?;
                    self.stack
                        .push(compare_values(left, right, Ordering::Greater)?);
                }
                OpCode::Less => {
                    let right = pop_stack(&mut self.stack, "Less")?;
                    let left = pop_stack(&mut self.stack, "Less")?;
                    self.stack
                        .push(compare_values(left, right, Ordering::Less)?);
                }
                OpCode::JumpIfFalse => {
                    let offset = self.read_u16_from_frame(frame_idx, chunk)? as usize;
                    let condition = self
                        .stack
                        .last()
                        .ok_or_else(|| "stack underflow on JumpIfFalse".to_string())?;
                    if !condition.truthy() {
                        self.frames[frame_idx].ip += offset;
                    }
                }
                OpCode::Jump => {
                    let offset = self.read_u16_from_frame(frame_idx, chunk)? as usize;
                    self.frames[frame_idx].ip += offset;
                }
                OpCode::Loop => {
                    let offset = self.read_u16_from_frame(frame_idx, chunk)? as usize;
                    self.frames[frame_idx].ip = self.frames[frame_idx].ip.saturating_sub(offset);
                }
                OpCode::Call => {
                    let argc = self.read_u8_from_frame(frame_idx, chunk)? as usize;
                    let args = pop_n(&mut self.stack, argc)?;
                    let callee = pop_stack(&mut self.stack, "Call")?;
                    self.call_value(callee, args)?;
                }
                OpCode::Return => {
                    let result = self.stack.pop().unwrap_or(Value::Nil);
                    let frame = self
                        .frames
                        .pop()
                        .ok_or_else(|| "return with empty frame stack".to_string())?;
                    let value = match frame.kind {
                        FrameKind::Regular => result,
                        FrameKind::Initializer { instance } => Value::Instance(instance),
                    };

                    if self.frames.is_empty() {
                        return Ok(value);
                    }
                    self.stack.push(value);
                }
                OpCode::BuildArray => {
                    let count = self.read_u16_from_frame(frame_idx, chunk)? as usize;
                    let items = pop_n(&mut self.stack, count)?;
                    self.stack.push(Value::Array(Rc::new(RefCell::new(items))));
                }
                OpCode::BuildArrayNil => {
                    let count = self.read_u16_from_frame(frame_idx, chunk)? as usize;
                    let items = vec![Value::Nil; count];
                    self.stack.push(Value::Array(Rc::new(RefCell::new(items))));
                }
                OpCode::GetIndex => {
                    let index = pop_stack(&mut self.stack, "GetIndex")?;
                    let target = pop_stack(&mut self.stack, "GetIndex")?;
                    self.stack.push(self.get_index(target, index)?);
                }
                OpCode::SetIndex => {
                    let value = pop_stack(&mut self.stack, "SetIndex")?;
                    let index = pop_stack(&mut self.stack, "SetIndex")?;
                    let target = pop_stack(&mut self.stack, "SetIndex")?;
                    self.set_index(target, index, value.clone())?;
                    self.stack.push(value);
                }
                OpCode::GetProperty => {
                    let name = self.read_name_constant(chunk, frame_idx)?;
                    let object = pop_stack(&mut self.stack, "GetProperty")?;
                    self.stack.push(self.get_property(object, name)?);
                }
                OpCode::SetProperty => {
                    let name = self.read_name_constant(chunk, frame_idx)?;
                    let value = pop_stack(&mut self.stack, "SetProperty")?;
                    let object = pop_stack(&mut self.stack, "SetProperty")?;
                    self.set_property(object, name, value.clone())?;
                    self.stack.push(value);
                }
                OpCode::Invoke => {
                    let name = self.read_name_constant(chunk, frame_idx)?.to_string();
                    let argc = self.read_u8_from_frame(frame_idx, chunk)? as usize;
                    let args = pop_n(&mut self.stack, argc)?;
                    let receiver = pop_stack(&mut self.stack, "Invoke")?;
                    self.invoke(receiver, &name, args)?;
                }
            }
        }
    }

    fn constant_to_value(&self, constant: Constant) -> Value {
        match constant {
            Constant::Int(v) => Value::Int(v),
            Constant::Float(v) => Value::Float(v),
            Constant::Bool(v) => Value::Bool(v),
            Constant::String(v) => Value::String(Rc::new(v)),
            Constant::Function(id) => Value::Function(id),
            Constant::Class(id) => Value::Class(id),
            Constant::Nil => Value::Nil,
        }
    }

    fn read_name_constant<'a>(&mut self, chunk: &'a Chunk, frame_idx: usize) -> VmResult<&'a str> {
        let idx = self.read_u16_from_frame(frame_idx, chunk)? as usize;
        match chunk.constants.get(idx) {
            Some(Constant::String(name)) => Ok(name.as_str()),
            _ => Err("expected string constant for identifier".to_string()),
        }
    }

    fn read_u8_from_frame(&mut self, frame_idx: usize, chunk: &Chunk) -> VmResult<u8> {
        let ip = self.frames[frame_idx].ip;
        let byte = *chunk
            .code
            .get(ip)
            .ok_or_else(|| "instruction read out of bounds".to_string())?;
        self.frames[frame_idx].ip += 1;
        Ok(byte)
    }

    fn read_u16_from_frame(&mut self, frame_idx: usize, chunk: &Chunk) -> VmResult<u16> {
        let low = self.read_u8_from_frame(frame_idx, chunk)?;
        let high = self.read_u8_from_frame(frame_idx, chunk)?;
        Ok(u16::from_le_bytes([low, high]))
    }

    fn push_frame(
        &mut self,
        function_id: usize,
        args: Vec<Value>,
        kind: FrameKind,
    ) -> VmResult<()> {
        let function = self
            .module
            .functions
            .get(function_id)
            .ok_or_else(|| format!("invalid function id {}", function_id))?;

        if args.len() != function.arity {
            return Err(format!(
                "function '{}' expects {} args, got {}",
                function.name,
                function.arity,
                args.len()
            ));
        }

        let mut locals = vec![Value::Nil; function.local_count.max(args.len())];
        for (idx, value) in args.into_iter().enumerate() {
            locals[idx] = value;
        }

        self.frames.push(CallFrame {
            function_id,
            ip: 0,
            locals,
            kind,
        });
        Ok(())
    }

    fn call_value(&mut self, callee: Value, args: Vec<Value>) -> VmResult<()> {
        match callee {
            Value::Function(function_id) => self.push_frame(function_id, args, FrameKind::Regular),
            Value::Native(function) => {
                let result = function(&args)?;
                self.stack.push(result);
                Ok(())
            }
            Value::Class(class_id) => self.instantiate_class(class_id, args),
            Value::BoundMethod {
                receiver,
                function_id,
            } => {
                let mut full_args = Vec::with_capacity(args.len() + 1);
                full_args.push(Value::Instance(receiver));
                full_args.extend(args);
                self.push_frame(function_id, full_args, FrameKind::Regular)
            }
            other => Err(format!("value '{}' is not callable", other)),
        }
    }

    fn instantiate_class(&mut self, class_id: usize, args: Vec<Value>) -> VmResult<()> {
        let class = self
            .module
            .classes
            .get(class_id)
            .ok_or_else(|| format!("invalid class id {}", class_id))?
            .clone();
        let mut fields = HashMap::new();
        for field in &class.fields {
            fields.insert(field.clone(), Value::Nil);
        }
        let instance = Rc::new(RefCell::new(InstanceObj { class_id, fields }));
        if let Some(init_id) = class.init {
            let mut init_args = Vec::with_capacity(args.len() + 1);
            init_args.push(Value::Instance(instance.clone()));
            init_args.extend(args);
            self.push_frame(
                init_id,
                init_args,
                FrameKind::Initializer {
                    instance: instance.clone(),
                },
            )?;
        } else if !args.is_empty() {
            return Err(format!("class '{}' constructor takes no args", class.name));
        } else {
            self.stack.push(Value::Instance(instance));
        }
        Ok(())
    }

    fn invoke(&mut self, receiver: Value, method_name: &str, args: Vec<Value>) -> VmResult<()> {
        let Value::Instance(instance) = receiver else {
            return Err("invoke target must be an instance".to_string());
        };

        let class_id = instance.borrow().class_id;
        let class = self
            .module
            .classes
            .get(class_id)
            .ok_or_else(|| format!("invalid class id {}", class_id))?;

        let function_id = class
            .methods
            .get(method_name)
            .copied()
            .ok_or_else(|| format!("undefined method '{}'", method_name))?;

        let mut full_args = Vec::with_capacity(args.len() + 1);
        full_args.push(Value::Instance(instance));
        full_args.extend(args);
        self.push_frame(function_id, full_args, FrameKind::Regular)
    }

    fn get_property(&self, object: Value, name: &str) -> VmResult<Value> {
        let Value::Instance(instance) = object else {
            return Err("property access requires instance".to_string());
        };

        if let Some(field) = instance.borrow().fields.get(name) {
            return Ok(field.clone());
        }

        let class_id = instance.borrow().class_id;
        let class = self
            .module
            .classes
            .get(class_id)
            .ok_or_else(|| format!("invalid class id {}", class_id))?;

        if let Some(method_id) = class.methods.get(name) {
            return Ok(Value::BoundMethod {
                receiver: instance,
                function_id: *method_id,
            });
        }

        Err(format!("unknown property '{}'", name))
    }

    fn set_property(&self, object: Value, name: &str, value: Value) -> VmResult<()> {
        let Value::Instance(instance) = object else {
            return Err("property assignment requires instance".to_string());
        };
        instance.borrow_mut().fields.insert(name.to_string(), value);
        Ok(())
    }

    fn get_index(&self, target: Value, index: Value) -> VmResult<Value> {
        let idx = match index {
            Value::Int(v) if v >= 0 => v as usize,
            _ => return Err("array index must be non-negative Int".to_string()),
        };

        match target {
            Value::Array(items) => items
                .borrow()
                .get(idx)
                .cloned()
                .ok_or_else(|| format!("array index {} out of bounds", idx)),
            _ => Err("index target must be array".to_string()),
        }
    }

    fn set_index(&self, target: Value, index: Value, value: Value) -> VmResult<()> {
        let idx = match index {
            Value::Int(v) if v >= 0 => v as usize,
            _ => return Err("array index must be non-negative Int".to_string()),
        };

        match target {
            Value::Array(items) => {
                if idx >= items.borrow().len() {
                    return Err(format!("array index {} out of bounds", idx));
                }
                items.borrow_mut()[idx] = value;
                Ok(())
            }
            _ => Err("index assignment target must be array".to_string()),
        }
    }
}

fn pop_stack(stack: &mut Vec<Value>, op: &str) -> VmResult<Value> {
    stack
        .pop()
        .ok_or_else(|| format!("stack underflow on {}", op))
}

fn pop_n(stack: &mut Vec<Value>, count: usize) -> VmResult<Vec<Value>> {
    if stack.len() < count {
        return Err("stack underflow while collecting call arguments".to_string());
    }
    let mut values = Vec::with_capacity(count);
    for _ in 0..count {
        values.push(stack.pop().expect("checked length"));
    }
    values.reverse();
    Ok(values)
}

fn add_values(left: Value, right: Value) -> VmResult<Value> {
    match (left, right) {
        (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a + b)),
        (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a + b)),
        (Value::String(a), Value::String(b)) => Ok(Value::String(Rc::new(format!("{}{}", a, b)))),
        (Value::String(a), b) => Ok(Value::String(Rc::new(format!("{}{}", a, b.stringify())))),
        (a, Value::String(b)) => Ok(Value::String(Rc::new(format!("{}{}", a.stringify(), b)))),
        _ => Err("'+' operands must be numeric or string-compatible".to_string()),
    }
}

fn numeric_binary(
    left: Value,
    right: Value,
    int_op: fn(i64, i64) -> i64,
    float_op: fn(f64, f64) -> f64,
) -> VmResult<Value> {
    match (left, right) {
        (Value::Int(a), Value::Int(b)) => Ok(Value::Int(int_op(a, b))),
        (Value::Float(a), Value::Float(b)) => Ok(Value::Float(float_op(a, b))),
        _ => Err("numeric operands must have the same type".to_string()),
    }
}

fn compare_values(left: Value, right: Value, expected: Ordering) -> VmResult<Value> {
    match (left, right) {
        (Value::Int(a), Value::Int(b)) => Ok(Value::Bool(a.cmp(&b) == expected)),
        (Value::Float(a), Value::Float(b)) => {
            let ordering = a
                .partial_cmp(&b)
                .ok_or_else(|| "cannot compare NaN values".to_string())?;
            Ok(Value::Bool(ordering == expected))
        }
        (Value::String(a), Value::String(b)) => {
            Ok(Value::Bool(a.as_str().cmp(b.as_str()) == expected))
        }
        _ => Err("comparison operands must be matching numeric or string".to_string()),
    }
}

fn native_to_string(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err("to_string expects 1 argument".to_string());
    }
    Ok(Value::String(Rc::new(args[0].stringify())))
}

fn native_print(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err("print expects 1 argument".to_string());
    }
    println!("{}", args[0]);
    Ok(Value::Nil)
}

fn native_some(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err("some expects 1 argument".to_string());
    }
    Ok(Value::Some(Box::new(args[0].clone())))
}

fn native_len(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err("len expects 1 argument".to_string());
    }

    match &args[0] {
        Value::Array(items) => Ok(Value::Int(items.borrow().len() as i64)),
        Value::String(text) => Ok(Value::Int(text.len() as i64)),
        _ => Err("len expects an array or string".to_string()),
    }
}

fn native_sum(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err("sum expects 1 argument".to_string());
    }

    let Value::Array(items) = &args[0] else {
        return Err("sum expects an array".to_string());
    };

    let values = items.borrow();
    if values.is_empty() {
        return Err("sum expects a non-empty array".to_string());
    }

    match &values[0] {
        Value::Int(_) => {
            let mut total = 0i64;
            for value in values.iter() {
                let Value::Int(int_value) = value else {
                    return Err("sum requires homogeneous numeric arrays".to_string());
                };
                total += int_value;
            }
            Ok(Value::Int(total))
        }
        Value::Float(_) => {
            let mut total = 0.0f64;
            for value in values.iter() {
                let Value::Float(float_value) = value else {
                    return Err("sum requires homogeneous numeric arrays".to_string());
                };
                total += float_value;
            }
            Ok(Value::Float(total))
        }
        _ => Err("sum expects Int[] or Float[]".to_string()),
    }
}

fn native_dot(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err("dot expects 2 arguments".to_string());
    }

    let Value::Array(left) = &args[0] else {
        return Err("dot expects arrays".to_string());
    };
    let Value::Array(right) = &args[1] else {
        return Err("dot expects arrays".to_string());
    };

    let left_values = left.borrow();
    let right_values = right.borrow();
    if left_values.len() != right_values.len() {
        return Err("dot expects same-length arrays".to_string());
    }

    if left_values.is_empty() {
        return Err("dot expects non-empty arrays".to_string());
    }

    match (&left_values[0], &right_values[0]) {
        (Value::Int(_), Value::Int(_)) => {
            let mut total = 0i64;
            for (left_value, right_value) in left_values.iter().zip(right_values.iter()) {
                let (Value::Int(l), Value::Int(r)) = (left_value, right_value) else {
                    return Err("dot requires matching homogeneous numeric arrays".to_string());
                };
                total += l * r;
            }
            Ok(Value::Int(total))
        }
        (Value::Float(_), Value::Float(_)) => {
            let mut total = 0.0f64;
            for (left_value, right_value) in left_values.iter().zip(right_values.iter()) {
                let (Value::Float(l), Value::Float(r)) = (left_value, right_value) else {
                    return Err("dot requires matching homogeneous numeric arrays".to_string());
                };
                total += l * r;
            }
            Ok(Value::Float(total))
        }
        _ => Err("dot expects matching Int[] or Float[]".to_string()),
    }
}

fn native_zeros(args: &[Value]) -> VmResult<Value> {
    make_filled_float_array(args, 0.0, "zeros")
}

fn native_ones(args: &[Value]) -> VmResult<Value> {
    make_filled_float_array(args, 1.0, "ones")
}

fn make_filled_float_array(args: &[Value], fill: f64, name: &str) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(format!("{} expects 1 argument", name));
    }

    let Value::Int(len) = args[0] else {
        return Err(format!("{} expects an Int length", name));
    };

    if len < 0 {
        return Err(format!("{} length must be non-negative", name));
    }

    let items = vec![Value::Float(fill); len as usize];
    Ok(Value::Array(Rc::new(RefCell::new(items))))
}

fn native_is_none(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err("is_none expects 1 argument".to_string());
    }
    Ok(Value::Bool(matches!(args[0], Value::None)))
}

fn native_unwrap(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err("unwrap expects 1 argument".to_string());
    }

    match &args[0] {
        Value::Some(inner) => Ok((**inner).clone()),
        Value::None => Err("attempted to unwrap None".to_string()),
        _ => Err("unwrap expects Option value".to_string()),
    }
}
