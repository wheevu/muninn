use crate::span::Span;

pub type VmResult<T> = Result<T, VmError>;

#[derive(Debug, Clone)]
pub struct VmError {
    pub message: String,
    pub span: Span,
}

impl VmError {
    pub fn new(message: impl Into<String>, span: Span) -> Self {
        Self {
            message: message.into(),
            span,
        }
    }
}
