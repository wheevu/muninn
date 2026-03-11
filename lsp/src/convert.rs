use muninn::error::MuninnError;
use muninn::span::Span;
use tower_lsp::lsp_types::{
    Diagnostic, DiagnosticRelatedInformation, DiagnosticSeverity, Location, Position, Range, Url,
};

pub fn span_to_range(span: Span) -> Range {
    let start = Position {
        line: span.line.saturating_sub(1) as u32,
        character: span.column.saturating_sub(1) as u32,
    };
    let mut end = Position {
        line: span.end_line.saturating_sub(1) as u32,
        character: span.end_column.saturating_sub(1) as u32,
    };

    if end.line < start.line || (end.line == start.line && end.character < start.character) {
        end = start;
    }

    if end == start {
        end.character = end.character.saturating_add(1);
    }

    Range { start, end }
}

pub fn errors_to_diagnostics(uri: &Url, errors: &[MuninnError]) -> Vec<Diagnostic> {
    errors
        .iter()
        .map(|error| error_to_diagnostic(uri, error))
        .collect()
}

fn phase_severity(phase: &str) -> DiagnosticSeverity {
    match phase {
        "lexer" | "parser" | "desugar" | "typecheck" | "compiler" | "vm" => {
            DiagnosticSeverity::ERROR
        }
        _ => DiagnosticSeverity::ERROR,
    }
}

fn error_to_diagnostic(uri: &Url, error: &MuninnError) -> Diagnostic {
    let related_information = if error.notes.is_empty() {
        None
    } else {
        Some(
            error
                .notes
                .iter()
                .map(|note| DiagnosticRelatedInformation {
                    location: Location {
                        uri: uri.clone(),
                        range: span_to_range(error.span),
                    },
                    message: note.clone(),
                })
                .collect(),
        )
    };

    Diagnostic {
        range: span_to_range(error.span),
        severity: Some(phase_severity(error.phase)),
        code: None,
        code_description: None,
        source: Some(format!("muninn-{}", error.phase)),
        message: error.message.clone(),
        related_information,
        tags: None,
        data: None,
    }
}
