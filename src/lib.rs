pub mod ast;
pub mod autodiff;
pub mod bytecode;
pub mod compiler;
pub mod error;
pub mod format;
pub mod frontend;
pub mod jit;
pub mod lexer;
pub mod native;
pub mod optim;
pub mod parser;
pub mod runtime;
pub mod source;
pub mod span;
pub mod tensor;
pub mod token;
pub mod typecheck;
pub mod value;
pub mod vm;

pub use autodiff::{AutodiffError, AutodiffErrorKind, Tape, TensorExpr, Variable, grad};
pub use bytecode::{
    BytecodeDecodeError, BytecodeModule, GlobalSpec, GlobalValueKind, decode_bytecode_module,
    encode_bytecode_module,
};
pub use format::format_source;
pub use frontend::{
    FrontendAnalysis, analyze_document, check_document, is_rename_identifier, lex_document,
    parse_document, references_to_target,
};
pub use tensor::Tensor;
pub use typecheck::{SemanticModel, Symbol, SymbolKind, Ty};
pub use value::Value;
pub use vm::{HostPolicy, Vm};

use bytecode::{GlobalSpec as ModuleGlobalSpec, GlobalValueKind as ModuleGlobalValueKind};
use compiler::compile_program;
use error::MuninnError;
use typecheck::check_program;
use vm::VmOptions;

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
            SymbolKind::Local
            | SymbolKind::Parameter
            | SymbolKind::NativeFunction(_) => None,
        })
        .collect();
    Ok(module)
}

pub fn run_bytecode_module(module: BytecodeModule) -> Result<Value, Vec<MuninnError>> {
    run_bytecode_module_with_options(module, VmOptions::default())
}

pub fn run_bytecode_module_with_options(
    module: BytecodeModule,
    options: VmOptions,
) -> Result<Value, Vec<MuninnError>> {
    run_bytecode_module_with_policy(module, HostPolicy::default(), options)
}

/// Runs a validated module under an explicit host policy. Use one call per
/// untrusted script: budgets and globals are per-VM, never shared.
pub fn run_bytecode_module_with_policy(
    module: BytecodeModule,
    policy: HostPolicy,
    options: VmOptions,
) -> Result<Value, Vec<MuninnError>> {
    bytecode::validate_module(&module)?;
    let mut vm = Vm::new_with_policy_and_options(module, policy, options);
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
