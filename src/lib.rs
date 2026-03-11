pub mod ast;
pub mod bytecode;
pub mod compiler;
pub mod desugar;
pub mod error;
pub mod frontend;
pub mod lexer;
pub mod lower;
pub mod parser;
pub mod span;
pub mod token;
pub mod typecheck;
pub mod vm;

use ast::Program;
use compiler::compile_program;
use desugar::desugar_program;
use error::MuninnError;
pub use frontend::{
    FrontendAnalysis, analyze_document, check_document, lex_document, parse_document,
};
use lexer::Lexer;
use lower::lower_program;
use parser::Parser;
use typecheck::check_program;
use vm::{Value, Vm};

pub fn compile_and_run(source: &str) -> Result<Value, Vec<MuninnError>> {
    let tokens = Lexer::new(source).lex()?;
    let mut parser = Parser::new(tokens);
    let parsed = parser.parse_program()?;
    let desugared: Program = desugar_program(parsed).map_err(|err| vec![err])?;
    let type_context = check_program(&desugared)?;
    let lowered = lower_program(desugared, &type_context);
    let module = compile_program(&lowered)?;
    let mut vm = Vm::new(module);
    vm.run()
        .map_err(|msg| vec![MuninnError::new("vm", msg, Default::default())])
}
