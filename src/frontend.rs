use crate::ast::Program;
use crate::error::MuninnError;
use crate::lexer::Lexer;
use crate::token::Token;
use crate::typecheck::{SemanticModel, analyze_program, check_program};
use crate::parser::Parser;

#[derive(Debug, Clone, Default)]
pub struct FrontendAnalysis {
    pub parsed: Option<Program>,
    pub semantics: Option<SemanticModel>,
    pub diagnostics: Vec<MuninnError>,
}

impl FrontendAnalysis {
    pub fn is_ok(&self) -> bool {
        self.diagnostics.is_empty()
    }
}

pub fn lex_document(source: &str) -> Result<Vec<Token>, Vec<MuninnError>> {
    Lexer::new(source).lex()
}

pub fn parse_document(source: &str) -> Result<Program, Vec<MuninnError>> {
    let tokens = lex_document(source)?;
    let mut parser = Parser::new(tokens);
    parser.parse_program()
}

pub fn check_document(program: &Program) -> Result<SemanticModel, Vec<MuninnError>> {
    check_program(program)
}

pub fn analyze_document(source: &str) -> FrontendAnalysis {
    let parsed = match parse_document(source) {
        Ok(program) => program,
        Err(diagnostics) => {
            return FrontendAnalysis {
                parsed: None,
                semantics: None,
                diagnostics,
            };
        }
    };

    let semantics = analyze_program(&parsed);
    let diagnostics = semantics.diagnostics.clone();
    FrontendAnalysis {
        parsed: Some(parsed),
        semantics: Some(semantics),
        diagnostics,
    }
}
