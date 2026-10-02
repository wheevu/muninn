//! A JSON writer, because the site protocol is a small amount of generated
//! text and this crate keeps Muninn's no-dependency rule.
//!
//! The page never parses: `app.js` hands the runner an action code and a
//! UTF-8 source buffer, and reads back a UTF-8 JSON document. Only writing
//! needs to exist, so only writing is here. Strings escape the two JSON
//! metacharacters and every control character, and pass the rest through as
//! UTF-8, which is what a browser's `JSON.parse` expects.
//!
//! # Why closures
//!
//! Every nested value is written inside a closure, and the value is created
//! and closed within that call. A child that borrowed its parent for as long
//! as it lived would make the borrow checker, not the writer, responsible for
//! closing brackets, and a missed close would corrupt every field after it.
//! The closure is the bracket.

/// The longest value the page is given for one stack slot, one operand, or
/// one printed line. A tensor with ten thousand elements is a real value; it
/// is just not something a 12px monospace panel should receive in full.
const MAX_VALUE_CHARS: usize = 4_000;

/// The document being written. The top level owns the text; everything below
/// it borrows from here, inside a closure call.
pub struct Document {
    buffer: String,
}

impl Document {
    pub fn new() -> Self {
        Self {
            buffer: String::new(),
        }
    }

    /// Writes one root object and returns the finished text.
    pub fn object(&mut self, write: impl FnOnce(&mut Object<'_>)) -> String {
        self.buffer.push('{');
        {
            let mut root = Object {
                buffer: &mut self.buffer,
                empty: true,
            };
            write(&mut root);
        }
        self.buffer.push('}');
        std::mem::take(&mut self.buffer)
    }
}

pub struct Object<'a> {
    buffer: &'a mut String,
    empty: bool,
}

impl<'a> Object<'a> {
    pub fn string(&mut self, key: &str, value: &str) -> &mut Self {
        self.key(key);
        push_string(self.buffer, value);
        self
    }

    /// A string the page shows in a fixed space, shortened rather than cut in
    /// half mid-number.
    pub fn short_string(&mut self, key: &str, value: &str) -> &mut Self {
        self.key(key);
        push_string(self.buffer, &truncate(value));
        self
    }

    pub fn number(&mut self, key: &str, value: i64) -> &mut Self {
        self.key(key);
        self.buffer.push_str(&value.to_string());
        self
    }

    pub fn bool(&mut self, key: &str, value: bool) -> &mut Self {
        self.key(key);
        self.buffer.push_str(if value { "true" } else { "false" });
        self
    }

    pub fn null(&mut self, key: &str) -> &mut Self {
        self.key(key);
        self.buffer.push_str("null");
        self
    }

    pub fn object(&mut self, key: &str, write: impl FnOnce(&mut Object<'_>)) -> &mut Self {
        self.key(key);
        self.buffer.push('{');
        {
            let mut child = Object {
                buffer: self.buffer,
                empty: true,
            };
            write(&mut child);
        }
        self.buffer.push('}');
        self
    }

    pub fn array(&mut self, key: &str, write: impl FnOnce(&mut Array<'_>)) -> &mut Self {
        self.key(key);
        self.buffer.push('[');
        {
            let mut child = Array {
                buffer: self.buffer,
                empty: true,
            };
            write(&mut child);
        }
        self.buffer.push(']');
        self
    }

    fn key(&mut self, key: &str) {
        self.comma();
        push_string(self.buffer, key);
        self.buffer.push(':');
    }

    fn comma(&mut self) {
        if self.empty {
            self.empty = false;
        } else {
            self.buffer.push(',');
        }
    }
}

pub struct Array<'a> {
    buffer: &'a mut String,
    empty: bool,
}

impl<'a> Array<'a> {
    /// The protocol only ever puts strings and tensor line numbers in an
    /// array: locals, stack slots, native names, diagnostics, instruction
    /// rows, and the 1-based output lines whose printed value was a tensor.
    /// Anything else would mean a shape the page does not read, so it is not
    /// here to be used by accident.
    pub fn short_string(&mut self, value: &str) -> &mut Self {
        self.comma();
        push_string(self.buffer, &truncate(value));
        self
    }

    /// A tensor line number inside an array, which needs no shortening.
    pub fn number(&mut self, value: i64) -> &mut Self {
        self.comma();
        self.buffer.push_str(&value.to_string());
        self
    }

    /// An object inside an array, which has no key of its own.
    pub fn object(&mut self, write: impl FnOnce(&mut Object<'_>)) -> &mut Self {
        self.comma();
        self.buffer.push('{');
        {
            let mut child = Object {
                buffer: self.buffer,
                empty: true,
            };
            write(&mut child);
        }
        self.buffer.push('}');
        self
    }

    fn comma(&mut self) {
        if self.empty {
            self.empty = false;
        } else {
            self.buffer.push(',');
        }
    }
}

fn truncate(value: &str) -> String {
    if value.chars().count() <= MAX_VALUE_CHARS {
        return value.to_string();
    }
    let kept: String = value.chars().take(MAX_VALUE_CHARS).collect();
    format!("{kept}... ({} characters total)", value.chars().count())
}

fn push_string(buffer: &mut String, value: &str) {
    buffer.push('"');
    for character in value.chars() {
        match character {
            '"' => buffer.push_str("\\\""),
            '\\' => buffer.push_str("\\\\"),
            '\n' => buffer.push_str("\\n"),
            '\r' => buffer.push_str("\\r"),
            '\t' => buffer.push_str("\\t"),
            control if control < ' ' => {
                buffer.push_str(&format!("\\u{:04x}", control as u32));
            }
            other => buffer.push(other),
        }
    }
    buffer.push('"');
}

#[cfg(test)]
mod tests {
    use super::Document;

    #[test]
    fn escapes_what_json_requires_and_passes_utf8_through() {
        let mut control = String::from("a ");
        control.push('\u{1}');
        let mut document = Document::new();
        let json = document.object(|root| {
            root.string("quotes", "a \"quoted\" \\ line");
            root.string("lines", "next\nnext\ttab\r");
            root.string("control", &control);
            root.string("unicode", "\u{1f426} muninn");
        });
        assert_eq!(
            json,
            concat!(
                r#"{"quotes":"a \"quoted\" \\ line","lines":"next\nnext\ttab\r","#,
                r#""control":"a \u0001","unicode":"🐦 muninn"}"#,
            )
        );
    }

    #[test]
    fn an_empty_object_is_still_braces() {
        let mut document = Document::new();
        assert_eq!(document.object(|_| {}), "{}");
    }

    #[test]
    fn nested_values_keep_their_commas_in_the_right_place() {
        let mut document = Document::new();
        let json = document.object(|root| {
            root.object("frame", |frame| {
                frame.string("function", "gcd");
                frame.number("ip", 12);
                frame.array("locals", |locals| {
                    locals.short_string("a: 84");
                    locals.short_string("b: 30");
                });
            });
            root.array("functions", |functions| {
                functions.object(|entry| {
                    entry.string("name", "gcd");
                    entry.number("id", 0);
                });
                functions.short_string("main");
            });
            root.array("grid", |grid| {
                grid.object(|cell| {
                    cell.number("row", 0);
                    cell.number("column", 1);
                    cell.short_string("value", "1.5");
                });
            });
        });
        assert_eq!(
            json,
            concat!(
                r#"{"frame":{"function":"gcd","ip":12,"locals":["a: 84","b: 30"]},"#,
                r#""functions":[{"name":"gcd","id":0},"main"],"#,
                r#""grid":[{"row":0,"column":1,"value":"1.5"}]}"#,
            )
        );
    }

    #[test]
    fn a_long_value_is_shortened_at_a_character_boundary() {
        let long = "é".repeat(5_000);
        let shortened = super::truncate(&long);
        assert!(
            shortened.ends_with("... (5000 characters total)"),
            "the page is told what it is not being shown: {shortened}"
        );
        let kept = shortened
            .trim_end_matches("... (5000 characters total)")
            .to_string();
        assert_eq!(kept.chars().count(), 4_000);
        assert!(kept.chars().all(|character| character == 'é'));
    }
}
