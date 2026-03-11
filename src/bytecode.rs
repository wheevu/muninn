use std::collections::HashMap;

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
    use super::{Chunk, Constant};

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
}
