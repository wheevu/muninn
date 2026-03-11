use std::collections::HashMap;
use std::sync::Arc;

use tower_lsp::lsp_types::{DocumentSymbol, Position, TextDocumentContentChangeEvent, Url};

use crate::analysis::SymbolIndex;

#[derive(Debug)]
pub struct DocumentState {
    pub version: i32,
    pub source: String,
    pub symbols: Option<SymbolIndex>,
    pub document_symbols: Vec<DocumentSymbol>,
    pub line_offsets: Vec<usize>,
}

impl DocumentState {
    pub fn new(
        version: i32,
        source: String,
        symbols: Option<SymbolIndex>,
        document_symbols: Vec<DocumentSymbol>,
    ) -> Self {
        let line_offsets = compute_line_offsets(&source);
        Self {
            version,
            source,
            symbols,
            document_symbols,
            line_offsets,
        }
    }

    pub fn offset_at(&self, line_zero_based: u32, character_zero_based: u32) -> usize {
        let line = line_zero_based as usize;
        let character = character_zero_based as usize;
        let line_start = self
            .line_offsets
            .get(line)
            .copied()
            .or_else(|| self.line_offsets.last().copied())
            .unwrap_or(0);
        line_start.saturating_add(character).min(self.source.len())
    }

    pub fn line_prefix_bounds(
        &self,
        line_zero_based: u32,
        character_zero_based: u32,
    ) -> (usize, usize) {
        let line = line_zero_based as usize;
        let start = self
            .line_offsets
            .get(line)
            .copied()
            .or_else(|| self.line_offsets.last().copied())
            .unwrap_or(0);
        let cursor = self.offset_at(line_zero_based, character_zero_based);
        (start, cursor.min(self.source.len()))
    }
}

#[derive(Default)]
pub struct ServerState {
    pub docs: HashMap<Url, Arc<DocumentState>>,
}

pub fn compute_line_offsets(source: &str) -> Vec<usize> {
    let mut offsets = vec![0usize];
    for (index, ch) in source.char_indices() {
        if ch == '\n' {
            offsets.push(index + 1);
        }
    }
    offsets
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

        let offsets = compute_line_offsets(&text);
        let start = position_to_offset(range.start, &offsets, text.len())?;
        let end = position_to_offset(range.end, &offsets, text.len())?;
        if start > end || end > text.len() {
            return None;
        }

        text.replace_range(start..end, &change.text);
    }

    Some(text)
}

fn position_to_offset(
    position: Position,
    line_offsets: &[usize],
    text_len: usize,
) -> Option<usize> {
    let line = position.line as usize;
    let character = position.character as usize;
    let start = *line_offsets.get(line)?;
    Some((start + character).min(text_len))
}

#[cfg(test)]
mod tests {
    use tower_lsp::lsp_types::{Position, Range, TextDocumentContentChangeEvent};

    use super::apply_content_changes;

    #[test]
    fn applies_incremental_insert() {
        let original = "let x = 1;\nlet y = 2;\n";
        let changes = vec![TextDocumentContentChangeEvent {
            range: Some(Range {
                start: Position {
                    line: 1,
                    character: 9,
                },
                end: Position {
                    line: 1,
                    character: 9,
                },
            }),
            range_length: None,
            text: " + 3".to_string(),
        }];

        let updated = apply_content_changes(original, &changes).expect("updated");
        assert_eq!(updated, "let x = 1;\nlet y = 2 + 3;\n");
    }

    #[test]
    fn applies_full_replace() {
        let original = "let x = 1;";
        let changes = vec![TextDocumentContentChangeEvent {
            range: None,
            range_length: None,
            text: "let x = 10;".to_string(),
        }];

        let updated = apply_content_changes(original, &changes).expect("updated");
        assert_eq!(updated, "let x = 10;");
    }
}
