//! Minimal idempotent formatter for Muninn source.
//!
//! The formatter is deliberately lexical, not semantic: it normalizes
//! indentation (4 spaces per brace depth), trims trailing whitespace,
//! collapses runs of blank lines, and ends the file with a single newline.
//! It never reorders tokens or changes semantics, so it is safe to run on
//! files with diagnostics. [`format_source`] is idempotent by construction:
//! formatting twice yields the same text.

/// Formats Muninn source with normalized indentation and whitespace.
pub fn format_source(source: &str) -> String {
    let mut out = String::with_capacity(source.len() + 16);
    let mut depth: usize = 0;
    let mut blank_run = 0;

    for raw_line in source.lines() {
        let trimmed = raw_line.trim();
        if trimmed.is_empty() {
            blank_run += 1;
            if blank_run == 1 {
                out.push('\n');
            }
            continue;
        }
        blank_run = 0;

        // A leading `}` closes one level before the line is emitted.
        let closes_leading = trimmed.strip_prefix('}').is_some();
        let level = depth.saturating_sub(usize::from(closes_leading));
        out.push_str(&"    ".repeat(level));
        out.push_str(trimmed);
        out.push('\n');

        // Net brace depth change, ignoring braces inside strings and
        // comments. Muninn strings use double quotes without escapes. A
        // leading `}` was already accounted for in `level`, so it is not
        // counted again here.
        let opens = count_char_outside_string(trimmed, '{');
        let closes = count_char_outside_string(trimmed, '}');
        let trailing_closes = closes.saturating_sub(usize::from(closes_leading));
        depth = level.saturating_add(opens).saturating_sub(trailing_closes);
    }

    if out.is_empty() {
        return String::new();
    }
    out
}

/// Counts `needle` occurrences outside `"..."` string literals.
fn count_char_outside_string(line: &str, needle: char) -> usize {
    let mut count = 0;
    let mut in_string = false;
    for ch in line.chars() {
        if ch == '"' {
            in_string = !in_string;
        } else if ch == needle && !in_string {
            count += 1;
        }
    }
    count
}

#[cfg(test)]
mod tests {
    use super::format_source;

    #[test]
    fn normalizes_indentation_and_trailing_whitespace() {
        let source = "fn add(a: Int, b: Int) -> Int {\nreturn a + b;   \n}\n";
        assert_eq!(
            format_source(source),
            "fn add(a: Int, b: Int) -> Int {\n    return a + b;\n}\n"
        );
    }

    #[test]
    fn collapses_blank_runs_and_ends_with_one_newline() {
        let source = "let a: Int = 1;\n\n\nlet b: Int = 2;";
        assert_eq!(
            format_source(source),
            "let a: Int = 1;\n\nlet b: Int = 2;\n"
        );
    }

    #[test]
    fn braces_inside_strings_do_not_change_depth() {
        let source = "let s: String = \"}{\";\nlet t: Int = 1;\n";
        assert_eq!(format_source(source), source);
    }

    #[test]
    fn nested_blocks_indent_per_level() {
        let source = "fn f() -> Int {\nif (true) {\nreturn 1;\n}\nreturn 0;\n}\n";
        assert_eq!(
            format_source(source),
            "fn f() -> Int {\n    if (true) {\n        return 1;\n    }\n    return 0;\n}\n"
        );
    }

    #[test]
    fn empty_source_stays_empty() {
        assert_eq!(format_source(""), "");
    }
}
