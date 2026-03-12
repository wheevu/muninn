use std::collections::HashMap;
use std::fmt::{Display, Formatter};

use crate::error::MuninnError;
use crate::span::Span;

const MUBC_MAGIC: &[u8; 4] = b"MUBC";
const MUBC_VERSION: u16 = 1;

#[derive(Debug, Clone)]
pub struct BytecodeModule {
    pub functions: Vec<FunctionBytecode>,
    pub entry_function: usize,
    pub globals: Vec<GlobalSpec>,
}

impl BytecodeModule {
    pub fn new() -> Self {
        Self {
            functions: Vec::new(),
            entry_function: 0,
            globals: Vec::new(),
        }
    }

    pub fn global_kind(&self, name: &str) -> Option<GlobalValueKind> {
        self.globals
            .iter()
            .find(|global| global.name == name)
            .map(|global| global.kind)
    }

    pub fn estimated_stack_capacity(&self) -> usize {
        self.functions
            .iter()
            .map(|function| function.local_count.max(1))
            .sum::<usize>()
            .max(16)
    }

    pub fn estimated_frame_capacity(&self) -> usize {
        self.functions.len().max(4)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GlobalSpec {
    pub name: String,
    pub kind: GlobalValueKind,
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GlobalValueKind {
    Int = 0,
    Float = 1,
    Bool = 2,
    String = 3,
    Tensor = 4,
    Function = 5,
}

impl GlobalValueKind {
    fn from_tag(tag: u8) -> Result<Self, BytecodeDecodeError> {
        match tag {
            0 => Ok(Self::Int),
            1 => Ok(Self::Float),
            2 => Ok(Self::Bool),
            3 => Ok(Self::String),
            4 => Ok(Self::Tensor),
            5 => Ok(Self::Function),
            _ => Err(BytecodeDecodeError::new(format!(
                "unknown global kind tag {}",
                tag
            ))),
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

    pub fn from_parts(code: Vec<u8>, spans: Vec<Span>, constants: Vec<Constant>) -> Self {
        let mut chunk = Self {
            code,
            spans,
            constants,
            constant_index: HashMap::new(),
        };
        chunk.rebuild_constant_index();
        chunk
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

    fn rebuild_constant_index(&mut self) {
        self.constant_index.clear();
        for (index, constant) in self.constants.iter().enumerate() {
            self.constant_index
                .entry(ConstantKey::from_constant(constant))
                .or_insert(index as u16);
        }
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BytecodeDecodeError {
    pub message: String,
}

impl BytecodeDecodeError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl Display for BytecodeDecodeError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for BytecodeDecodeError {}

pub fn encode_bytecode_module(module: &BytecodeModule) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(MUBC_MAGIC);
    write_u16(&mut bytes, MUBC_VERSION);
    write_u32(&mut bytes, module.entry_function as u32);
    write_u32(&mut bytes, module.globals.len() as u32);
    for global in &module.globals {
        write_string(&mut bytes, &global.name);
        bytes.push(global.kind as u8);
    }
    write_u32(&mut bytes, module.functions.len() as u32);
    for function in &module.functions {
        write_string(&mut bytes, &function.name);
        write_u32(&mut bytes, function.arity as u32);
        write_u32(&mut bytes, function.local_count as u32);
        bytes.push(u8::from(function.expects_return_value));
        write_bytes(&mut bytes, &function.chunk.code);
        write_u32(&mut bytes, function.chunk.spans.len() as u32);
        for span in &function.chunk.spans {
            write_span(&mut bytes, *span);
        }
        write_u32(&mut bytes, function.chunk.constants.len() as u32);
        for constant in &function.chunk.constants {
            write_constant(&mut bytes, constant);
        }
    }
    bytes
}

pub fn decode_bytecode_module(bytes: &[u8]) -> Result<BytecodeModule, BytecodeDecodeError> {
    let mut reader = BytecodeReader::new(bytes);
    reader.expect_magic(MUBC_MAGIC)?;
    let version = reader.read_u16()?;
    if version != MUBC_VERSION {
        return Err(BytecodeDecodeError::new(format!(
            "unsupported .mubc version {} (expected {})",
            version, MUBC_VERSION
        )));
    }

    let entry_function = reader.read_u32()? as usize;
    let global_count = reader.read_u32()? as usize;
    let mut globals = Vec::with_capacity(global_count);
    for _ in 0..global_count {
        let name = reader.read_string()?;
        let kind = GlobalValueKind::from_tag(reader.read_u8()?)?;
        globals.push(GlobalSpec { name, kind });
    }

    let function_count = reader.read_u32()? as usize;
    let mut functions = Vec::with_capacity(function_count);
    for _ in 0..function_count {
        let name = reader.read_string()?;
        let arity = reader.read_u32()? as usize;
        let local_count = reader.read_u32()? as usize;
        let expects_return_value = reader.read_u8()? != 0;

        let code = reader.read_bytes()?;
        let span_count = reader.read_u32()? as usize;
        let mut spans = Vec::with_capacity(span_count);
        for _ in 0..span_count {
            spans.push(reader.read_span()?);
        }

        let constant_count = reader.read_u32()? as usize;
        let mut constants = Vec::with_capacity(constant_count);
        for _ in 0..constant_count {
            constants.push(reader.read_constant()?);
        }

        functions.push(FunctionBytecode {
            name,
            arity,
            local_count,
            expects_return_value,
            chunk: Chunk::from_parts(code, spans, constants),
        });
    }

    reader.finish()?;

    let module = BytecodeModule {
        functions,
        entry_function,
        globals,
    };
    validate_module(&module)
        .map_err(|errors| BytecodeDecodeError::new(errors[0].to_string()))?;
    Ok(module)
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

fn write_u16(bytes: &mut Vec<u8>, value: u16) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn write_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn write_i64(bytes: &mut Vec<u8>, value: i64) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn write_f64(bytes: &mut Vec<u8>, value: f64) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn write_bytes(bytes: &mut Vec<u8>, value: &[u8]) {
    write_u32(bytes, value.len() as u32);
    bytes.extend_from_slice(value);
}

fn write_string(bytes: &mut Vec<u8>, value: &str) {
    write_bytes(bytes, value.as_bytes());
}

fn write_span(bytes: &mut Vec<u8>, span: Span) {
    write_u32(bytes, span.line as u32);
    write_u32(bytes, span.column as u32);
    write_u32(bytes, span.offset as u32);
    write_u32(bytes, span.end_line as u32);
    write_u32(bytes, span.end_column as u32);
    write_u32(bytes, span.end_offset as u32);
}

fn write_constant(bytes: &mut Vec<u8>, constant: &Constant) {
    match constant {
        Constant::Int(value) => {
            bytes.push(0);
            write_i64(bytes, *value);
        }
        Constant::Float(value) => {
            bytes.push(1);
            write_f64(bytes, *value);
        }
        Constant::Bool(value) => {
            bytes.push(2);
            bytes.push(u8::from(*value));
        }
        Constant::String(value) => {
            bytes.push(3);
            write_string(bytes, value);
        }
        Constant::Function(value) => {
            bytes.push(4);
            write_u32(bytes, *value as u32);
        }
        Constant::Nil => bytes.push(5),
    }
}

struct BytecodeReader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> BytecodeReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn finish(&self) -> Result<(), BytecodeDecodeError> {
        if self.offset == self.bytes.len() {
            Ok(())
        } else {
            Err(BytecodeDecodeError::new("trailing bytes in .mubc payload"))
        }
    }

    fn expect_magic(&mut self, magic: &[u8]) -> Result<(), BytecodeDecodeError> {
        let found = self.read_exact(magic.len())?;
        if found == magic {
            Ok(())
        } else {
            Err(BytecodeDecodeError::new("invalid .mubc magic header"))
        }
    }

    fn read_u8(&mut self) -> Result<u8, BytecodeDecodeError> {
        Ok(self.read_exact(1)?[0])
    }

    fn read_u16(&mut self) -> Result<u16, BytecodeDecodeError> {
        let bytes = self.read_exact(2)?;
        Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
    }

    fn read_u32(&mut self) -> Result<u32, BytecodeDecodeError> {
        let bytes = self.read_exact(4)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn read_i64(&mut self) -> Result<i64, BytecodeDecodeError> {
        let bytes = self.read_exact(8)?;
        Ok(i64::from_le_bytes(bytes.try_into().expect("8-byte array")))
    }

    fn read_f64(&mut self) -> Result<f64, BytecodeDecodeError> {
        let bytes = self.read_exact(8)?;
        Ok(f64::from_le_bytes(bytes.try_into().expect("8-byte array")))
    }

    fn read_bytes(&mut self) -> Result<Vec<u8>, BytecodeDecodeError> {
        let len = self.read_u32()? as usize;
        Ok(self.read_exact(len)?.to_vec())
    }

    fn read_string(&mut self) -> Result<String, BytecodeDecodeError> {
        let bytes = self.read_bytes()?;
        String::from_utf8(bytes)
            .map_err(|_| BytecodeDecodeError::new("invalid UTF-8 string in .mubc payload"))
    }

    fn read_span(&mut self) -> Result<Span, BytecodeDecodeError> {
        Ok(Span::range(
            self.read_u32()? as usize,
            self.read_u32()? as usize,
            self.read_u32()? as usize,
            self.read_u32()? as usize,
            self.read_u32()? as usize,
            self.read_u32()? as usize,
        ))
    }

    fn read_constant(&mut self) -> Result<Constant, BytecodeDecodeError> {
        match self.read_u8()? {
            0 => Ok(Constant::Int(self.read_i64()?)),
            1 => Ok(Constant::Float(self.read_f64()?)),
            2 => Ok(Constant::Bool(self.read_u8()? != 0)),
            3 => Ok(Constant::String(self.read_string()?)),
            4 => Ok(Constant::Function(self.read_u32()? as usize)),
            5 => Ok(Constant::Nil),
            tag => Err(BytecodeDecodeError::new(format!(
                "unknown constant tag {}",
                tag
            ))),
        }
    }

    fn read_exact(&mut self, len: usize) -> Result<&'a [u8], BytecodeDecodeError> {
        let end = self.offset.saturating_add(len);
        if end > self.bytes.len() {
            return Err(BytecodeDecodeError::new(
                "unexpected end of .mubc payload",
            ));
        }
        let slice = &self.bytes[self.offset..end];
        self.offset = end;
        Ok(slice)
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
    use crate::span::Span;

    use super::{
        BytecodeModule, Chunk, Constant, FunctionBytecode, GlobalSpec, GlobalValueKind, OpCode,
        decode_bytecode_module, encode_bytecode_module, validate_module,
    };

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
            globals: Vec::new(),
        };

        let errors = validate_module(&module).expect_err("validator errors");
        assert!(errors
            .iter()
            .any(|error| error.message.contains("local slot 9 out of bounds")));
    }

    #[test]
    fn round_trips_bytecode_module_through_mubc() {
        let mut chunk = Chunk::new();
        let name_index = chunk
            .add_constant(Constant::String("value".to_string()))
            .expect("name");
        chunk.write_op(OpCode::Constant, Span::range(1, 1, 0, 1, 2, 1));
        chunk.write_u16(name_index, Span::range(1, 1, 0, 1, 2, 1));
        chunk.write_op(OpCode::Return, Span::range(1, 1, 0, 1, 2, 1));

        let module = BytecodeModule {
            functions: vec![FunctionBytecode {
                name: "entry".to_string(),
                arity: 0,
                local_count: 0,
                expects_return_value: false,
                chunk,
            }],
            entry_function: 0,
            globals: vec![GlobalSpec {
                name: "counter".to_string(),
                kind: GlobalValueKind::Int,
            }],
        };

        let bytes = encode_bytecode_module(&module);
        let decoded = decode_bytecode_module(&bytes).expect("decode");

        assert_eq!(decoded.entry_function, module.entry_function);
        assert_eq!(decoded.globals, module.globals);
        assert_eq!(decoded.functions.len(), 1);
        assert_eq!(decoded.functions[0].name, module.functions[0].name);
        assert_eq!(decoded.functions[0].chunk.code, module.functions[0].chunk.code);
        assert_eq!(decoded.functions[0].chunk.spans, module.functions[0].chunk.spans);
        assert_eq!(decoded.functions[0].chunk.constants.len(), 1);
    }

    #[test]
    fn rejects_invalid_magic() {
        let error = decode_bytecode_module(b"NOPE").expect_err("decode error");
        assert!(error.message.contains("magic"));
    }

    #[test]
    fn rejects_unsupported_version() {
        let mut bytes = Vec::from(*b"MUBC");
        bytes.extend_from_slice(&999u16.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());

        let error = decode_bytecode_module(&bytes).expect_err("decode error");
        assert!(error.message.contains("unsupported"));
    }

    #[test]
    fn rejects_truncated_payload() {
        let module = BytecodeModule::new();
        let mut bytes = encode_bytecode_module(&BytecodeModule {
            functions: vec![FunctionBytecode {
                name: "entry".to_string(),
                arity: 0,
                local_count: 0,
                expects_return_value: false,
                chunk: Chunk::from_parts(vec![OpCode::Return as u8], vec![Span::default()], Vec::new()),
            }],
            entry_function: 0,
            globals: module.globals,
        });
        bytes.pop();

        let error = decode_bytecode_module(&bytes).expect_err("decode error");
        assert!(error.message.contains("unexpected end"));
    }
}
