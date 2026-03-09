use std::fmt::{Display, Formatter};

use crate::span::Span;

#[derive(Debug, Clone)]
pub struct MuninnError {
    pub phase: &'static str,
    pub message: String,
    pub span: Span,
}

impl MuninnError {
    pub fn new(phase: &'static str, message: impl Into<String>, span: Span) -> Self {
        Self {
            phase,
            message: message.into(),
            span,
        }
    }
}

impl Display for MuninnError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} error at {}:{}: {}",
            self.phase, self.span.line, self.span.column, self.message
        )
    }
}

impl std::error::Error for MuninnError {}
