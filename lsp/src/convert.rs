use muninn::error::MuninnError;
use muninn::source::offset_to_utf16_position;
use muninn::span::Span;
use tower_lsp::lsp_types::{Diagnostic, DiagnosticSeverity, Position, Range, Url};

pub fn span_to_range(source: &str, line_starts: &[usize], span: Span) -> Range {
    let (start_line, start_character) = offset_to_utf16_position(source, line_starts, span.offset);
    let (mut end_line, mut end_character) = offset_to_utf16_position(
        source,
        line_starts,
        span.end_offset.max(span.offset + 1),
    );

    if end_line < start_line || (end_line == start_line && end_character < start_character) {
        end_line = start_line;
        end_character = start_character.saturating_add(1);
    }

    Range {
        start: Position {
            line: start_line,
            character: start_character,
        },
        end: Position {
            line: end_line,
            character: end_character,
        },
    }
}

pub fn errors_to_diagnostics(
    source: &str,
    line_starts: &[usize],
    _uri: &Url,
    errors: &[MuninnError],
) -> Vec<Diagnostic> {
    errors
        .iter()
        .map(|error| Diagnostic {
            range: span_to_range(source, line_starts, error.span),
            severity: Some(DiagnosticSeverity::ERROR),
            code: None,
            code_description: None,
            source: Some(format!("muninn-{}", error.phase)),
            message: error.message.clone(),
            related_information: None,
            tags: None,
            data: None,
        })
        .collect()
}
