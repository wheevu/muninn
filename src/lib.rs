pub mod ast;
pub mod bytecode;
pub mod compiler;
pub mod error;
pub mod frontend;
pub mod lexer;
pub mod native;
pub mod parser;
pub mod runtime;
pub mod source;
pub mod span;
pub mod tensor;
pub mod token;
pub mod typecheck;
pub mod value;
pub mod vm;

pub use bytecode::{
    BytecodeDecodeError, BytecodeModule, GlobalSpec, GlobalValueKind, decode_bytecode_module,
    encode_bytecode_module,
};
pub use frontend::{FrontendAnalysis, analyze_document, check_document, lex_document, parse_document};
pub use typecheck::{SemanticModel, Symbol, SymbolKind, Ty};
pub use value::Value;

use bytecode::{GlobalSpec as ModuleGlobalSpec, GlobalValueKind as ModuleGlobalValueKind};
use compiler::compile_program;
use error::MuninnError;
use typecheck::check_program;
use vm::Vm;

pub fn compile_to_bytecode(source: &str) -> Result<BytecodeModule, Vec<MuninnError>> {
    let program = parse_document(source)?;
    let semantics = check_program(&program)?;
    let mut module = compile_program(&program)?;
    module.globals = semantics
        .symbols
        .iter()
        .filter_map(|symbol| match symbol.kind {
            SymbolKind::Global => Some(ModuleGlobalSpec {
                name: symbol.name.clone(),
                kind: global_kind_from_ty(&symbol.ty)?,
            }),
            SymbolKind::Function => Some(ModuleGlobalSpec {
                name: symbol.name.clone(),
                kind: ModuleGlobalValueKind::Function,
            }),
            SymbolKind::Local | SymbolKind::Parameter | SymbolKind::NativeFunction(_) => None,
        })
        .collect();
    Ok(module)
}

pub fn run_bytecode_module(module: BytecodeModule) -> Result<Value, Vec<MuninnError>> {
    bytecode::validate_module(&module)?;
    let mut vm = Vm::new(module);
    vm.run()
        .map_err(|error| vec![MuninnError::new("vm", error.message, error.span)])
}

pub fn compile_and_run(source: &str) -> Result<Value, Vec<MuninnError>> {
    let module = compile_to_bytecode(source)?;
    run_bytecode_module(module)
}

fn global_kind_from_ty(ty: &Ty) -> Option<ModuleGlobalValueKind> {
    match ty {
        Ty::Int => Some(ModuleGlobalValueKind::Int),
        Ty::Float => Some(ModuleGlobalValueKind::Float),
        Ty::Bool => Some(ModuleGlobalValueKind::Bool),
        Ty::String => Some(ModuleGlobalValueKind::String),
        Ty::Tensor => Some(ModuleGlobalValueKind::Tensor),
        Ty::Void | Ty::Function(_, _) | Ty::NativeFunction(_) | Ty::Error => None,
    }
}
