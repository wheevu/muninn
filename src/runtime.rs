use crate::error::MuninnError;
use crate::span::Span;

pub type VmResult<T> = Result<T, MuninnError>;

pub fn vm_error(message: impl Into<String>, span: Span) -> MuninnError {
    MuninnError::new("vm", message, span)
}
