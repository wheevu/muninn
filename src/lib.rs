pub mod ast;
pub mod builtins;
pub mod bytecode;
pub mod compiler;
pub mod error;
pub mod frontend;
pub mod lexer;
pub mod parser;
pub mod source;
pub mod span;
pub mod token;
pub mod typecheck;
pub mod vm;

pub use frontend::{FrontendAnalysis, analyze_document, check_document, lex_document, parse_document};
pub use typecheck::{SemanticModel, Symbol, SymbolKind, Ty};
pub use vm::Value;

use compiler::compile_program;
use error::MuninnError;
use typecheck::check_program;
use vm::Vm;

pub fn compile_and_run(source: &str) -> Result<Value, Vec<MuninnError>> {
    let program = parse_document(source)?;
    check_program(&program)?;
    let module = compile_program(&program)?;
    let mut vm = Vm::new(module);
    vm.run()
        .map_err(|error| vec![MuninnError::new("vm", error.message, error.span)])
}
