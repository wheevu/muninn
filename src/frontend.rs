use crate::ast::Program;
use crate::desugar::desugar_program;
use crate::error::MuninnError;
use crate::lexer::Lexer;
use crate::parser::Parser;
use crate::token::Token;
use crate::typecheck::{TypeContext, check_program};

#[derive(Debug, Clone, Default)]
pub struct FrontendAnalysis {
    pub parsed: Option<Program>,
    pub type_context: Option<TypeContext>,
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

pub fn check_document(program: &Program) -> Result<TypeContext, Vec<MuninnError>> {
    check_program(program)
}

pub fn analyze_document(source: &str) -> FrontendAnalysis {
    let mut diagnostics = Vec::<MuninnError>::new();

    let tokens = match lex_document(source) {
        Ok(tokens) => tokens,
        Err(errors) => {
            diagnostics.extend(errors);
            return FrontendAnalysis {
                parsed: None,
                type_context: None,
                diagnostics,
            };
        }
    };

    let mut parser = Parser::new(tokens);
    let parsed = match parser.parse_program() {
        Ok(program) => program,
        Err(errors) => {
            diagnostics.extend(errors);
            return FrontendAnalysis {
                parsed: None,
                type_context: None,
                diagnostics,
            };
        }
    };

    let desugared = match desugar_program(parsed.clone()) {
        Ok(program) => program,
        Err(error) => {
            diagnostics.push(error);
            return FrontendAnalysis {
                parsed: Some(parsed),
                type_context: None,
                diagnostics,
            };
        }
    };

    let type_context = match check_program(&desugared) {
        Ok(context) => Some(context),
        Err(errors) => {
            diagnostics.extend(errors);
            None
        }
    };

    FrontendAnalysis {
        parsed: Some(parsed),
        type_context,
        diagnostics,
    }
}
