use crate::error::MuninnError;
use crate::span::Span;
use crate::token::{Token, TokenKind};

pub struct Lexer<'a> {
    chars: Vec<char>,
    _source: &'a str,
    start: usize,
    current: usize,
    line: usize,
    column: usize,
    start_line: usize,
    start_column: usize,
    errors: Vec<MuninnError>,
}

impl<'a> Lexer<'a> {
    pub fn new(source: &'a str) -> Self {
        Self {
            chars: source.chars().collect(),
            _source: source,
            start: 0,
            current: 0,
            line: 1,
            column: 1,
            start_line: 1,
            start_column: 1,
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
            Span::new(self.line, self.column, self.current),
        ));

        if self.errors.is_empty() {
            Ok(tokens)
        } else {
            Err(self.errors)
        }
    }

    fn scan_token(&mut self) -> Option<Token> {
        let ch = self.advance();
        let kind = match ch {
            '(' => TokenKind::LeftParen,
            ')' => TokenKind::RightParen,
            '{' => TokenKind::LeftBrace,
            '}' => TokenKind::RightBrace,
            '[' => TokenKind::LeftBracket,
            ']' => TokenKind::RightBracket,
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
            '/' => {
                if self.match_char('/') {
                    while self.peek().is_some_and(|c| c != '\n') {
                        self.advance();
                    }
                    return None;
                }
                TokenKind::Slash
            }
            '?' => TokenKind::Question,
            '.' => {
                if self.match_char('.') {
                    TokenKind::DotDot
                } else {
                    TokenKind::Dot
                }
            }
            '|' => {
                if self.match_char('>') {
                    TokenKind::PipeGreater
                } else {
                    TokenKind::Pipe
                }
            }
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
            '<' => {
                if self.match_char('=') {
                    TokenKind::LessEqual
                } else {
                    TokenKind::Less
                }
            }
            '>' => {
                if self.match_char('=') {
                    TokenKind::GreaterEqual
                } else {
                    TokenKind::Greater
                }
            }
            '"' => return self.string_token(),
            c if c.is_ascii_digit() => return Some(self.number_token(c)),
            c if is_identifier_start(c) => return Some(self.identifier_token()),
            other => {
                self.errors.push(MuninnError::new(
                    "lexer",
                    format!("unexpected character '{}'", other),
                    self.span(),
                ));
                return None;
            }
        };

        Some(Token::new(kind, self.span()))
    }

    fn string_token(&mut self) -> Option<Token> {
        let mut value = String::new();

        while let Some(ch) = self.peek() {
            if ch == '"' {
                self.advance();
                return Some(Token::new(TokenKind::StringLiteral(value), self.span()));
            }

            if ch == '\\' {
                self.advance();
                let escaped = match self.peek() {
                    Some('n') => '\n',
                    Some('r') => '\r',
                    Some('t') => '\t',
                    Some('"') => '"',
                    Some('\\') => '\\',
                    Some('{') => '{',
                    Some('}') => '}',
                    Some(other) => other,
                    None => {
                        self.errors.push(MuninnError::new(
                            "lexer",
                            "unterminated escape in string",
                            self.span(),
                        ));
                        return None;
                    }
                };
                self.advance();
                value.push(escaped);
                continue;
            }

            value.push(ch);
            self.advance();
        }

        self.errors.push(MuninnError::new(
            "lexer",
            "unterminated string",
            self.span(),
        ));
        None
    }

    fn number_token(&mut self, _first: char) -> Token {
        while self.peek().is_some_and(|c| c.is_ascii_digit()) {
            self.advance();
        }

        let mut is_float = false;
        if self.peek() == Some('.')
            && self.peek_next().is_some_and(|c| c.is_ascii_digit())
            && self.peek_next() != Some('.')
        {
            is_float = true;
            self.advance();
            while self.peek().is_some_and(|c| c.is_ascii_digit()) {
                self.advance();
            }
        }

        let literal: String = self.chars[self.start..self.current].iter().collect();
        let kind = if is_float {
            TokenKind::FloatLiteral(literal.parse::<f64>().unwrap_or_default())
        } else {
            TokenKind::IntLiteral(literal.parse::<i64>().unwrap_or_default())
        };

        Token::new(kind, self.span())
    }

    fn identifier_token(&mut self) -> Token {
        while self.peek().is_some_and(is_identifier_continue) {
            self.advance();
        }
        let text: String = self.chars[self.start..self.current].iter().collect();

        let kind = match text.as_str() {
            "class" => TokenKind::Class,
            "fn" => TokenKind::Fn,
            "let" => TokenKind::Let,
            "mut" => TokenKind::Mut,
            "if" => TokenKind::If,
            "else" => TokenKind::Else,
            "unless" => TokenKind::Unless,
            "return" => TokenKind::Return,
            "while" => TokenKind::While,
            "for" => TokenKind::For,
            "in" => TokenKind::In,
            "true" => TokenKind::True,
            "false" => TokenKind::False,
            "init" => TokenKind::Init,
            "self" => TokenKind::SelfKw,
            "Int" => TokenKind::TypeInt,
            "Float" => TokenKind::TypeFloat,
            "String" => TokenKind::TypeString,
            "Bool" => TokenKind::TypeBool,
            "Void" => TokenKind::TypeVoid,
            _ => TokenKind::Identifier(text),
        };

        Token::new(kind, self.span())
    }

    fn skip_whitespace(&mut self) {
        while let Some(ch) = self.peek() {
            match ch {
                ' ' | '\r' | '\t' => {
                    self.advance();
                }
                '\n' => {
                    self.advance();
                }
                _ => break,
            }
        }
    }

    fn is_at_end(&self) -> bool {
        self.current >= self.chars.len()
    }

    fn advance(&mut self) -> char {
        let ch = self.chars[self.current];
        self.current += 1;
        if ch == '\n' {
            self.line += 1;
            self.column = 1;
        } else {
            self.column += 1;
        }
        ch
    }

    fn peek(&self) -> Option<char> {
        self.chars.get(self.current).copied()
    }

    fn peek_next(&self) -> Option<char> {
        self.chars.get(self.current + 1).copied()
    }

    fn match_char(&mut self, expected: char) -> bool {
        if self.peek() == Some(expected) {
            self.advance();
            true
        } else {
            false
        }
    }

    fn span(&self) -> Span {
        Span::range(
            self.start_line,
            self.start_column,
            self.start,
            self.line,
            self.column,
            self.current,
        )
    }
}

fn is_identifier_start(ch: char) -> bool {
    ch == '_' || ch.is_ascii_alphabetic()
}

fn is_identifier_continue(ch: char) -> bool {
    ch == '_' || ch.is_ascii_alphanumeric()
}

#[cfg(test)]
mod tests {
    use super::Lexer;
    use crate::token::TokenKind;

    #[test]
    fn lexes_pipeline_for_and_range() {
        let source = "for i in 0..10 { i |> f(); }";
        let tokens = Lexer::new(source).lex().expect("tokens");
        assert!(tokens.iter().any(|t| matches!(t.kind, TokenKind::For)));
        assert!(tokens.iter().any(|t| matches!(t.kind, TokenKind::In)));
        assert!(tokens.iter().any(|t| matches!(t.kind, TokenKind::DotDot)));
        assert!(tokens
            .iter()
            .any(|t| matches!(t.kind, TokenKind::PipeGreater)));
    }

    #[test]
    fn keeps_range_distinct_from_float() {
        let source = "1..10 1.25";
        let tokens = Lexer::new(source).lex().expect("tokens");
        assert!(matches!(tokens[0].kind, TokenKind::IntLiteral(1)));
        assert!(matches!(tokens[1].kind, TokenKind::DotDot));
        assert!(matches!(tokens[2].kind, TokenKind::IntLiteral(10)));
        assert!(matches!(tokens[3].kind, TokenKind::FloatLiteral(_)));
    }

    #[test]
    fn lexes_question_operator() {
        let source = "maybe()?;";
        let tokens = Lexer::new(source).lex().expect("tokens");
        assert!(tokens
            .iter()
            .any(|token| matches!(token.kind, TokenKind::Question)));
    }
}
