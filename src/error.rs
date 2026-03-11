use std::fmt::{Display, Formatter};

use crate::span::Span;

#[derive(Debug, Clone)]
pub struct MuninnError {
    pub phase: &'static str,
    pub message: String,
    pub span: Span,
    pub notes: Vec<String>,
}

impl MuninnError {
    pub fn new(phase: &'static str, message: impl Into<String>, span: Span) -> Self {
        Self {
            phase,
            message: message.into(),
            span,
            notes: Vec::new(),
        }
    }

    pub fn with_note(mut self, note: impl Into<String>) -> Self {
        self.notes.push(note.into());
        self
    }

    pub fn render_with_source(&self, source: &str) -> String {
        if self.span.line == 0 {
            return self.to_string();
        }

        let line_text = source
            .lines()
            .nth(self.span.line.saturating_sub(1))
            .unwrap_or_default();

        let line_len = line_text.chars().count();
        let start_column = self.span.column.saturating_sub(1).min(line_len);
        let raw_width = if self.span.line == self.span.end_line {
            self.span.end_column.saturating_sub(self.span.column)
        } else {
            self.span.width()
        };
        let width = raw_width.max(1).min(line_len.saturating_sub(start_column).max(1));
        let marker = format!("{}{}", " ".repeat(start_column), "^".repeat(width));

        let mut rendered = format!(
            "{} error: {}\n --> {}:{}\n  |\n{:>3} | {}\n  | {}",
            self.phase,
            self.message,
            self.span.line,
            self.span.column,
            self.span.line,
            line_text,
            marker,
        );

        for note in &self.notes {
            rendered.push_str("\n  = note: ");
            rendered.push_str(note);
        }

        rendered
    }
}

impl Display for MuninnError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} error at {}:{}: {}",
            self.phase, self.span.line, self.span.column, self.message
        )?;
        for note in &self.notes {
            write!(f, "\n  note: {}", note)?;
        }
        Ok(())
    }
}

impl std::error::Error for MuninnError {}

#[cfg(test)]
mod tests {
    use super::MuninnError;
    use crate::span::Span;

    #[test]
    fn renders_error_with_source_context() {
        let source = "let x: Int = \"oops\";";
        let err = MuninnError::new(
            "typecheck",
            "expected Int, got String",
            Span::range(1, 14, 13, 1, 20, 19),
        )
        .with_note("try changing the literal type");

        let rendered = err.render_with_source(source);
        assert!(rendered.contains("typecheck error"));
        assert!(rendered.contains("let x: Int = \"oops\";"));
        assert!(rendered.contains("note:"));
    }
}
