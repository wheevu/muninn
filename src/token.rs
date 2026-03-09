use crate::span::Span;

#[derive(Debug, Clone, PartialEq)]
pub enum TokenKind {
    LeftParen,
    RightParen,
    LeftBrace,
    RightBrace,
    LeftBracket,
    RightBracket,
    Comma,
    Dot,
    DotDot,
    Semicolon,
    Colon,
    Plus,
    Minus,
    Star,
    Slash,
    Question,
    Pipe,
    PipeGreater,
    Bang,
    BangEqual,
    Equal,
    EqualEqual,
    Greater,
    GreaterEqual,
    Less,
    LessEqual,
    Arrow,

    Identifier(String),
    IntLiteral(i64),
    FloatLiteral(f64),
    StringLiteral(String),

    Class,
    Fn,
    Let,
    Mut,
    If,
    Else,
    Unless,
    Return,
    While,
    For,
    In,
    True,
    False,
    Init,
    SelfKw,

    TypeInt,
    TypeFloat,
    TypeString,
    TypeBool,
    TypeVoid,

    Eof,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
}

impl Token {
    pub const fn new(kind: TokenKind, span: Span) -> Self {
        Self { kind, span }
    }
}
