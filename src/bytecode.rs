use std::collections::HashMap;

use crate::error::MuninnError;
use crate::span::Span;

#[derive(Debug, Clone)]
pub struct BytecodeModule {
    pub functions: Vec<FunctionBytecode>,
    pub entry_function: usize,
}

impl BytecodeModule {
    pub fn new() -> Self {
        Self {
            functions: Vec::new(),
            entry_function: 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct FunctionBytecode {
    pub name: String,
    pub arity: usize,
    pub local_count: usize,
    pub expects_return_value: bool,
    pub chunk: Chunk,
}

#[derive(Debug, Clone)]
pub struct Chunk {
    pub code: Vec<u8>,
    pub spans: Vec<Span>,
    pub constants: Vec<Constant>,
    constant_index: HashMap<ConstantKey, u16>,
}

impl Chunk {
    pub fn new() -> Self {
        Self {
            code: Vec::new(),
            spans: Vec::new(),
            constants: Vec::new(),
            constant_index: HashMap::new(),
        }
    }

    pub fn write_op(&mut self, op: OpCode, span: Span) {
        self.code.push(op as u8);
        self.spans.push(span);
    }

    pub fn write_u8(&mut self, value: u8, span: Span) {
        self.code.push(value);
        self.spans.push(span);
    }

    pub fn write_u16(&mut self, value: u16, span: Span) {
        for byte in value.to_le_bytes() {
            self.code.push(byte);
            self.spans.push(span);
        }
    }

    pub fn add_constant(&mut self, constant: Constant) -> Result<u16, String> {
        let key = ConstantKey::from_constant(&constant);
        if let Some(index) = self.constant_index.get(&key) {
            return Ok(*index);
        }
        if self.constants.len() >= u16::MAX as usize {
            return Err(format!(
                "constant pool overflow: maximum of {} entries",
                u16::MAX
            ));
        }
        let index = self.constants.len() as u16;
        self.constants.push(constant);
        self.constant_index.insert(key, index);
        Ok(index)
    }

    pub fn span_at(&self, ip: usize) -> Span {
        self.spans.get(ip).copied().unwrap_or_default()
    }
}

#[derive(Debug, Clone)]
pub enum Constant {
    Int(i64),
    Float(f64),
    Bool(bool),
    String(String),
    Function(usize),
    Nil,
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpCode {
    Constant = 0,
    Nil = 1,
    True = 2,
    False = 3,
    Pop = 4,
    GetLocal = 5,
    SetLocal = 6,
    DefineGlobal = 7,
    GetGlobal = 8,
    SetGlobal = 9,
    Add = 10,
    Subtract = 11,
    Multiply = 12,
    Divide = 13,
    Negate = 14,
    Not = 15,
    Equal = 16,
    Greater = 17,
    Less = 18,
    JumpIfFalse = 19,
    Jump = 20,
    Loop = 21,
    Call = 22,
    Return = 23,
}

impl OpCode {
    pub fn from_byte(byte: u8) -> Option<Self> {
        match byte {
            0 => Some(Self::Constant),
            1 => Some(Self::Nil),
            2 => Some(Self::True),
            3 => Some(Self::False),
            4 => Some(Self::Pop),
            5 => Some(Self::GetLocal),
            6 => Some(Self::SetLocal),
            7 => Some(Self::DefineGlobal),
            8 => Some(Self::GetGlobal),
            9 => Some(Self::SetGlobal),
            10 => Some(Self::Add),
            11 => Some(Self::Subtract),
            12 => Some(Self::Multiply),
            13 => Some(Self::Divide),
            14 => Some(Self::Negate),
            15 => Some(Self::Not),
            16 => Some(Self::Equal),
            17 => Some(Self::Greater),
            18 => Some(Self::Less),
            19 => Some(Self::JumpIfFalse),
            20 => Some(Self::Jump),
            21 => Some(Self::Loop),
            22 => Some(Self::Call),
            23 => Some(Self::Return),
            _ => None,
        }
    }
}

pub fn validate_module(module: &BytecodeModule) -> Result<(), Vec<MuninnError>> {
    let mut errors = Vec::new();

    if module.functions.is_empty() {
        errors.push(MuninnError::new(
            "compiler",
            "bytecode module has no functions",
            Span::default(),
        ));
        return Err(errors);
    }

    if module.entry_function >= module.functions.len() {
        errors.push(MuninnError::new(
            "compiler",
            format!(
                "entry function {} is out of bounds for {} functions",
                module.entry_function,
                module.functions.len()
            ),
            Span::default(),
        ));
    }

    for (function_id, function) in module.functions.iter().enumerate() {
        if function.local_count < function.arity {
            errors.push(MuninnError::new(
                "compiler",
                format!(
                    "function '{}' local_count {} is smaller than arity {}",
                    function.name, function.local_count, function.arity
                ),
                Span::default(),
            ));
        }

        if function.chunk.code.len() != function.chunk.spans.len() {
            errors.push(MuninnError::new(
                "compiler",
                format!(
                    "function '{}' has {} op bytes but {} span entries",
                    function.name,
                    function.chunk.code.len(),
                    function.chunk.spans.len()
                ),
                Span::default(),
            ));
            continue;
        }

        for constant in &function.chunk.constants {
            if let Constant::Function(target) = constant
                && *target >= module.functions.len()
            {
                errors.push(MuninnError::new(
                    "compiler",
                    format!(
                        "function '{}' references missing function id {}",
                        function.name, target
                    ),
                    Span::default(),
                ));
            }
        }

        let mut ip = 0usize;
        while ip < function.chunk.code.len() {
            let span = function.chunk.span_at(ip);
            let byte = function.chunk.code[ip];
            let Some(op) = OpCode::from_byte(byte) else {
                errors.push(MuninnError::new(
                    "compiler",
                    format!("invalid opcode {} in function '{}'", byte, function.name),
                    span,
                ));
                break;
            };

            let width = instruction_width(op);
            if ip + width > function.chunk.code.len() {
                errors.push(MuninnError::new(
                    "compiler",
                    format!(
                        "truncated {:?} instruction at byte {} in function '{}'",
                        op, ip, function.name
                    ),
                    span,
                ));
                break;
            }

            match op {
                OpCode::Constant => {
                    let index = read_u16(&function.chunk.code, ip + 1) as usize;
                    if index >= function.chunk.constants.len() {
                        errors.push(MuninnError::new(
                            "compiler",
                            format!(
                                "constant index {} out of bounds in function '{}'",
                                index, function.name
                            ),
                            span,
                        ));
                    }
                }
                OpCode::DefineGlobal | OpCode::GetGlobal | OpCode::SetGlobal => {
                    let index = read_u16(&function.chunk.code, ip + 1) as usize;
                    match function.chunk.constants.get(index) {
                        Some(Constant::String(_)) => {}
                        Some(_) => errors.push(MuninnError::new(
                            "compiler",
                            format!(
                                "global name constant at index {} is not a string in function '{}'",
                                index, function.name
                            ),
                            span,
                        )),
                        None => errors.push(MuninnError::new(
                            "compiler",
                            format!(
                                "global name constant index {} out of bounds in function '{}'",
                                index, function.name
                            ),
                            span,
                        )),
                    }
                }
                OpCode::GetLocal | OpCode::SetLocal => {
                    let slot = read_u16(&function.chunk.code, ip + 1) as usize;
                    if slot >= function.local_count {
                        errors.push(MuninnError::new(
                            "compiler",
                            format!(
                                "local slot {} out of bounds (local_count {}) in function '{}'",
                                slot, function.local_count, function.name
                            ),
                            span,
                        ));
                    }
                }
                OpCode::Jump | OpCode::JumpIfFalse => {
                    let jump = read_u16(&function.chunk.code, ip + 1) as usize;
                    let target = ip + width + jump;
                    if target > function.chunk.code.len() {
                        errors.push(MuninnError::new(
                            "compiler",
                            format!(
                                "forward jump target {} out of bounds in function '{}'",
                                target, function.name
                            ),
                            span,
                        ));
                    }
                }
                OpCode::Loop => {
                    let jump = read_u16(&function.chunk.code, ip + 1) as usize;
                    if jump > ip + width {
                        errors.push(MuninnError::new(
                            "compiler",
                            format!(
                                "backward loop jump {} underflows instruction pointer in function '{}'",
                                jump, function.name
                            ),
                            span,
                        ));
                    }
                }
                OpCode::Call
                | OpCode::Nil
                | OpCode::True
                | OpCode::False
                | OpCode::Pop
                | OpCode::Add
                | OpCode::Subtract
                | OpCode::Multiply
                | OpCode::Divide
                | OpCode::Negate
                | OpCode::Not
                | OpCode::Equal
                | OpCode::Greater
                | OpCode::Less
                | OpCode::Return => {}
            }

            ip += width;
        }

        if function_id == module.entry_function && function.chunk.code.is_empty() {
            errors.push(MuninnError::new(
                "compiler",
                "entry function has empty bytecode",
                Span::default(),
            ));
        }
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

fn instruction_width(op: OpCode) -> usize {
    match op {
        OpCode::Constant
        | OpCode::GetLocal
        | OpCode::SetLocal
        | OpCode::DefineGlobal
        | OpCode::GetGlobal
        | OpCode::SetGlobal
        | OpCode::JumpIfFalse
        | OpCode::Jump
        | OpCode::Loop => 3,
        OpCode::Call => 2,
        OpCode::Nil
        | OpCode::True
        | OpCode::False
        | OpCode::Pop
        | OpCode::Add
        | OpCode::Subtract
        | OpCode::Multiply
        | OpCode::Divide
        | OpCode::Negate
        | OpCode::Not
        | OpCode::Equal
        | OpCode::Greater
        | OpCode::Less
        | OpCode::Return => 1,
    }
}

fn read_u16(code: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([code[at], code[at + 1]])
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum ConstantKey {
    Int(i64),
    Float(u64),
    Bool(bool),
    String(String),
    Function(usize),
    Nil,
}

impl ConstantKey {
    fn from_constant(constant: &Constant) -> Self {
        match constant {
            Constant::Int(value) => Self::Int(*value),
            Constant::Float(value) => Self::Float(value.to_bits()),
            Constant::Bool(value) => Self::Bool(*value),
            Constant::String(value) => Self::String(value.clone()),
            Constant::Function(value) => Self::Function(*value),
            Constant::Nil => Self::Nil,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::span::Span;

    use super::{BytecodeModule, Chunk, Constant, FunctionBytecode, OpCode, validate_module};

    #[test]
    fn deduplicates_identical_constants() {
        let mut chunk = Chunk::new();
        let first = chunk
            .add_constant(Constant::String("name".to_string()))
            .expect("first constant");
        let second = chunk
            .add_constant(Constant::String("name".to_string()))
            .expect("second constant");
        assert_eq!(first, second);
        assert_eq!(chunk.constants.len(), 1);
    }

    #[test]
    fn validator_rejects_local_slot_overflow() {
        let mut chunk = Chunk::new();
        chunk.write_op(OpCode::GetLocal, Span::default());
        chunk.write_u16(9, Span::default());
        chunk.write_op(OpCode::Return, Span::default());

        let module = BytecodeModule {
            functions: vec![FunctionBytecode {
                name: "bad".to_string(),
                arity: 0,
                local_count: 1,
                expects_return_value: false,
                chunk,
            }],
            entry_function: 0,
        };

        let errors = validate_module(&module).expect_err("validator errors");
        assert!(errors
            .iter()
            .any(|error| error.message.contains("local slot 9 out of bounds")));
    }
}
