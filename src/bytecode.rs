use std::collections::HashMap;
use std::rc::Rc;

#[derive(Debug, Clone)]
pub struct BytecodeModule {
    pub functions: Vec<Rc<FunctionBytecode>>,
    pub classes: Vec<Rc<ClassBytecode>>,
    pub entry_function: usize,
}

impl BytecodeModule {
    pub fn new() -> Self {
        Self {
            functions: Vec::new(),
            classes: Vec::new(),
            entry_function: 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct FunctionBytecode {
    pub name: String,
    pub arity: usize,
    pub local_count: usize,
    pub chunk: Chunk,
}

#[derive(Debug, Clone)]
pub struct ClassBytecode {
    pub name: String,
    pub fields: Vec<String>,
    pub methods: HashMap<String, usize>,
    pub init: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct Chunk {
    pub code: Vec<u8>,
    pub constants: Vec<Constant>,
}

impl Chunk {
    pub fn new() -> Self {
        Self {
            code: Vec::new(),
            constants: Vec::new(),
        }
    }

    pub fn write_op(&mut self, op: OpCode) {
        self.code.push(op as u8);
    }

    pub fn write_u8(&mut self, value: u8) {
        self.code.push(value);
    }

    pub fn write_u16(&mut self, value: u16) {
        self.code.extend_from_slice(&value.to_le_bytes());
    }

    pub fn add_constant(&mut self, constant: Constant) -> u16 {
        if self.constants.len() >= u16::MAX as usize {
            panic!("constant pool overflow: maximum of {} entries", u16::MAX);
        }
        self.constants.push(constant);
        (self.constants.len() - 1) as u16
    }
}

#[derive(Debug, Clone)]
pub enum Constant {
    Int(i64),
    Float(f64),
    Bool(bool),
    String(String),
    Function(usize),
    Class(usize),
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
    BuildArray = 24,
    GetIndex = 25,
    SetIndex = 26,
    GetProperty = 27,
    SetProperty = 28,
    Invoke = 29,
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
            24 => Some(Self::BuildArray),
            25 => Some(Self::GetIndex),
            26 => Some(Self::SetIndex),
            27 => Some(Self::GetProperty),
            28 => Some(Self::SetProperty),
            29 => Some(Self::Invoke),
            _ => None,
        }
    }
}
