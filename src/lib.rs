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
    BytecodeDecodeError, BytecodeModule, GlobalSpec, GlobalValueKind, MUBC_VERSION,
    decode_bytecode_module, encode_bytecode_module,
};
pub use format::format_source;
pub use frontend::{
    FileDiagnostic, FrontendAnalysis, LoadedProject, analyze_document, check_document,
    is_rename_identifier, lex_document, load_project, parse_document, references_to_target,
};
pub use native::parse_exit_code;
pub use tensor::Tensor;
pub use typecheck::{SemanticModel, Symbol, SymbolKind, Ty};
pub use value::Value;
pub use vm::{HostPolicy, Vm};

use std::path::Path;

use bytecode::{GlobalSpec as ModuleGlobalSpec, GlobalValueKind as ModuleGlobalValueKind};
use compiler::compile_program;
use error::MuninnError;
use typecheck::check_program;
use vm::VmOptions;

pub fn compile_to_bytecode(source: &str) -> Result<BytecodeModule, Vec<MuninnError>> {
    let program = parse_document(source)?;
    let semantics = check_program(&program)?;
    let mut module = compile_program(&program)?;
    module.globals = globals_from_semantics(&semantics);
    Ok(module)
}

/// Compiles an entry file plus its transitive imports. Diagnostics name
/// the file that owns each span.
pub fn compile_project(
    entry: &Path,
) -> Result<(BytecodeModule, LoadedProject), Vec<FileDiagnostic>> {
    let project = load_project(entry)?;
    let module = compile_loaded_project(&project)?;
    Ok((module, project))
}

/// Compiles an already-loaded project (used by the `run` bytecode
/// cache, which hashes the combined source before compiling).
pub fn compile_loaded_project(
    project: &LoadedProject,
) -> Result<BytecodeModule, Vec<FileDiagnostic>> {
    let program = parse_document(&project.combined).map_err(|errors| project.map_errors(errors))?;
    let semantics = check_program(&program).map_err(|errors| project.map_errors(errors))?;
    let mut module = compile_program(&program).map_err(|errors| project.map_errors(errors))?;
    module.globals = globals_from_semantics(&semantics);
    Ok(module)
}

/// Runs a project entry file under an explicit host policy. Budgets and
/// globals are per-VM, never shared: one call per untrusted script.
pub fn run_project_with_policy(
    entry: &Path,
    policy: HostPolicy,
    options: VmOptions,
) -> Result<Value, Vec<FileDiagnostic>> {
    let (module, project) = compile_project(entry)?;
    run_bytecode_module_with_policy(module, policy, options)
        .map_err(|errors| project.map_errors(errors))
}

fn globals_from_semantics(semantics: &typecheck::SemanticModel) -> Vec<ModuleGlobalSpec> {
    semantics
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
            | SymbolKind::Record
            | SymbolKind::NativeFunction(_) => None,
        })
        .collect()
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
        Ty::Record(_) => Some(ModuleGlobalValueKind::Record),
        Ty::Void | Ty::Function(_, _) | Ty::NativeFunction(_) | Ty::Error => None,
    }
}
