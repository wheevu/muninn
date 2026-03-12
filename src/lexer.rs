use crate::error::MuninnError;
use crate::span::Span;
use crate::token::{Token, TokenKind};

pub struct Lexer<'a> {
    source: &'a str,
    chars: Vec<(usize, char)>,
    start: usize,
    current: usize,
    start_line: usize,
    start_column: usize,
    line: usize,
    column: usize,
    errors: Vec<MuninnError>,
}

impl<'a> Lexer<'a> {
    pub fn new(source: &'a str) -> Self {
        Self {
            source,
            chars: source.char_indices().collect(),
            start: 0,
            current: 0,
            start_line: 1,
            start_column: 1,
            line: 1,
            column: 1,
            errors: Vec::new(),
        }
    }

    pub fn lex(mut self) -> Result<Vec<Token>, Vec<MuninnError>> {
        let mut tokens = Vec::new();
        while !self.is_at_end() {
            self.skip_whitespace();
            if self.is_at_end() {
                break;
            }

            self.start = self.current;
            self.start_line = self.line;
            self.start_column = self.column;

            if let Some(token) = self.scan_token() {
                tokens.push(token);
            }
        }

        tokens.push(Token::new(
            TokenKind::Eof,
            Span::new(self.line, self.column, self.current_offset()),
        ));

        if self.errors.is_empty() {
            Ok(tokens)
        } else {
            Err(self.errors)
        }
    }

    fn scan_token(&mut self) -> Option<Token> {
        let ch = self.advance()?;
        let kind = match ch {
            '(' => TokenKind::LeftParen,
            ')' => TokenKind::RightParen,
            '{' => TokenKind::LeftBrace,
            '}' => TokenKind::RightBrace,
            ',' => TokenKind::Comma,
            ';' => TokenKind::Semicolon,
            ':' => TokenKind::Colon,
            '+' => TokenKind::Plus,
            '-' => {
                if self.match_char('>') {
                    TokenKind::Arrow
                } else {
                    TokenKind::Minus
                }
            }
            '*' => TokenKind::Star,
            '/' => TokenKind::Slash,
            '!' => {
                if self.match_char('=') {
                    TokenKind::BangEqual
                } else {
                    TokenKind::Bang
                }
            }
            '=' => {
                if self.match_char('=') {
                    TokenKind::EqualEqual
                } else {
                    TokenKind::Equal
                }
            }
            '>' => {
                if self.match_char('=') {
                    TokenKind::GreaterEqual
                } else {
                    TokenKind::Greater
                }
            }
            '<' => {
                if self.match_char('=') {
                    TokenKind::LessEqual
                } else {
                    TokenKind::Less
                }
            }
            '&' => {
                if self.match_char('&') {
                    TokenKind::AndAnd
                } else {
                    self.error("unexpected '&' (did you mean '&&'?)");
                    return None;
                }
            }
            '|' => {
                if self.match_char('|') {
                    TokenKind::OrOr
                } else {
                    self.error("unexpected '|' (did you mean '||'?)");
                    return None;
                }
            }
            '"' => return self.scan_string(),
            ch if ch.is_ascii_digit() => return self.scan_number(ch),
            ch if is_identifier_start(ch) => return Some(self.scan_identifier(ch)),
            _ => {
                self.error(format!("unexpected character '{}'", ch));
                return None;
            }
        };

        Some(Token::new(kind, self.make_span()))
    }

    fn scan_string(&mut self) -> Option<Token> {
        let mut value = String::new();
        while let Some(ch) = self.peek() {
            if ch == '"' {
                self.advance();
                return Some(Token::new(
                    TokenKind::StringLiteral(value),
                    self.make_span(),
                ));
            }

            if ch == '\\' {
                self.advance();
                let Some(escaped) = self.advance() else {
                    self.error("unterminated escape sequence");
                    return None;
                };
                value.push(match escaped {
                    'n' => '\n',
                    'r' => '\r',
                    't' => '\t',
                    '"' => '"',
                    '\\' => '\\',
                    other => {
                        self.error(format!("unsupported escape sequence '\\{}'", other));
                        other
                    }
                });
                continue;
            }

            value.push(ch);
            self.advance();
        }

        self.error("unterminated string literal");
        None
    }

    fn scan_number(&mut self, first: char) -> Option<Token> {
        let mut literal = String::from(first);
        while self.peek().is_some_and(|ch| ch.is_ascii_digit()) {
            literal.push(self.advance()?);
        }

        if self.peek() == Some('.') && self.peek_next().is_some_and(|ch| ch.is_ascii_digit()) {
            literal.push(self.advance()?);
            while self.peek().is_some_and(|ch| ch.is_ascii_digit()) {
                literal.push(self.advance()?);
            }
            let value = match literal.parse::<f64>() {
                Ok(value) => value,
                Err(_) => {
                    self.error(format!("invalid float literal '{}'", literal));
                    return None;
                }
            };
            return Some(Token::new(TokenKind::FloatLiteral(value), self.make_span()));
        }

        let value = match literal.parse::<i64>() {
            Ok(value) => value,
            Err(_) => {
                self.error(format!("invalid integer literal '{}'", literal));
                return None;
            }
        };
        Some(Token::new(TokenKind::IntLiteral(value), self.make_span()))
    }

    fn scan_identifier(&mut self, first: char) -> Token {
        let mut value = String::from(first);
        while self.peek().is_some_and(is_identifier_continue) {
            value.push(self.advance().expect("identifier char"));
        }

        let kind = match value.as_str() {
            "fn" => TokenKind::Fn,
            "let" => TokenKind::Let,
            "mut" => TokenKind::Mut,
            "if" => TokenKind::If,
            "else" => TokenKind::Else,
            "return" => TokenKind::Return,
            "while" => TokenKind::While,
            "true" => TokenKind::True,
            "false" => TokenKind::False,
            "Int" => TokenKind::TypeInt,
            "Float" => TokenKind::TypeFloat,
            "Bool" => TokenKind::TypeBool,
            "String" => TokenKind::TypeString,
            "Void" => TokenKind::TypeVoid,
            _ => TokenKind::Identifier(value),
        };

        Token::new(kind, self.make_span())
    }

    fn skip_whitespace(&mut self) {
        loop {
            match self.peek() {
                Some(ch) if ch.is_whitespace() => {
                    self.advance();
                }
                Some('/') if self.peek_next() == Some('/') => {
                    while self.peek().is_some_and(|ch| ch != '\n') {
                        self.advance();
                    }
                }
                _ => break,
            }
        }
    }

    fn make_span(&self) -> Span {
        Span::range(
            self.start_line,
            self.start_column,
            self.start_offset(),
            self.line,
            self.column,
            self.current_offset(),
        )
    }

    fn error(&mut self, message: impl Into<String>) {
        self.errors
            .push(MuninnError::new("lexer", message, self.make_span()));
    }

    fn start_offset(&self) -> usize {
        self.chars
            .get(self.start)
            .map(|(offset, _)| *offset)
            .unwrap_or(self.source.len())
    }

    fn current_offset(&self) -> usize {
        self.chars
            .get(self.current)
            .map(|(offset, _)| *offset)
            .unwrap_or(self.source.len())
    }

    fn is_at_end(&self) -> bool {
        self.current >= self.chars.len()
    }

    fn peek(&self) -> Option<char> {
        self.chars.get(self.current).map(|(_, ch)| *ch)
    }

    fn peek_next(&self) -> Option<char> {
        self.chars.get(self.current + 1).map(|(_, ch)| *ch)
    }

    fn match_char(&mut self, expected: char) -> bool {
        if self.peek() == Some(expected) {
            self.advance();
            true
        } else {
            false
        }
    }

    fn advance(&mut self) -> Option<char> {
        let (_, ch) = *self.chars.get(self.current)?;
        self.current += 1;
        if ch == '\n' {
            self.line += 1;
            self.column = 1;
        } else {
            self.column += 1;
        }
        Some(ch)
    }
}

fn is_identifier_start(ch: char) -> bool {
    ch == '_' || ch.is_ascii_alphabetic()
}

fn is_identifier_continue(ch: char) -> bool {
    is_identifier_start(ch) || ch.is_ascii_digit()
}

#[cfg(test)]
mod tests {
    use super::Lexer;
    use crate::token::TokenKind;

    #[test]
    fn lexes_keywords_and_types() {
        let tokens = Lexer::new("fn main() -> Void { let mut x: Int = 1; }")
            .lex()
            .expect("tokens");
        assert!(matches!(tokens[0].kind, TokenKind::Fn));
        assert!(tokens
            .iter()
            .any(|token| matches!(token.kind, TokenKind::TypeVoid)));
        assert!(tokens
            .iter()
            .any(|token| matches!(token.kind, TokenKind::TypeInt)));
    }

    #[test]
    fn lexes_utf8_string_literal_offsets() {
        let tokens = Lexer::new("let bird: String = \"🐦\";")
            .lex()
            .expect("tokens");
        let string = tokens
            .iter()
            .find(|token| matches!(token.kind, TokenKind::StringLiteral(_)))
            .expect("string token");
        assert!(string.span.end_offset > string.span.offset);
    }

    #[test]
    fn reports_invalid_integer_literal() {
        let errors = Lexer::new("let x: Int = 9223372036854775808;")
            .lex()
            .expect_err("lexer error");
        assert!(errors
            .iter()
            .any(|error| error.message.contains("invalid integer literal")));
    }
}
