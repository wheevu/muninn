use std::collections::HashMap;
use std::sync::Arc;

use muninn::frontend::FrontendAnalysis;
use muninn::source::{compute_line_starts, utf16_position_to_offset};
use tower_lsp::lsp_types::{TextDocumentContentChangeEvent, Url};

#[derive(Debug)]
pub struct DocumentState {
    pub version: i32,
    pub source: String,
    pub analysis: FrontendAnalysis,
    pub line_starts: Vec<usize>,
}

impl DocumentState {
    pub fn new(version: i32, source: String, analysis: FrontendAnalysis) -> Self {
        let line_starts = compute_line_starts(&source);
        Self {
            version,
            source,
            analysis,
            line_starts,
        }
    }

    pub fn offset_at(&self, line: u32, character: u32) -> Option<usize> {
        utf16_position_to_offset(&self.source, &self.line_starts, line, character)
    }
}

#[derive(Default)]
pub struct ServerState {
    pub docs: HashMap<Url, Arc<DocumentState>>,
}

pub fn apply_content_changes(
    original: &str,
    changes: &[TextDocumentContentChangeEvent],
) -> Option<String> {
    if changes.is_empty() {
        return Some(original.to_string());
    }

    let mut text = original.to_string();
    for change in changes {
        let Some(range) = change.range else {
            text = change.text.clone();
            continue;
        };

        let line_starts = compute_line_starts(&text);
        let start = utf16_position_to_offset(&text, &line_starts, range.start.line, range.start.character)?;
        let end = utf16_position_to_offset(&text, &line_starts, range.end.line, range.end.character)?;
        if start > end || end > text.len() {
            return None;
        }
        text.replace_range(start..end, &change.text);
    }

    Some(text)
}

#[cfg(test)]
mod tests {
    use tower_lsp::lsp_types::{Position, Range, TextDocumentContentChangeEvent};

    use super::apply_content_changes;

    #[test]
    fn applies_utf16_safe_incremental_insert() {
        let original = "let bird = \"🐦\";\n";
        let changes = vec![TextDocumentContentChangeEvent {
            range: Some(Range {
                start: Position {
                    line: 0,
                    character: 14,
                },
                end: Position {
                    line: 0,
                    character: 14,
                },
            }),
            range_length: None,
            text: "!".to_string(),
        }];

        let updated = apply_content_changes(original, &changes).expect("updated");
        assert_eq!(updated, "let bird = \"🐦!\";\n");
    }
}
