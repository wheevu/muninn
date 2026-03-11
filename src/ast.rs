use crate::span::Span;

#[derive(Debug, Clone)]
pub struct Program {
    pub statements: Vec<Stmt>,
}

#[derive(Debug, Clone)]
pub enum Stmt {
    Let {
        name: String,
        mutable: bool,
        ty: Option<TypeExpr>,
        initializer: Expr,
        span: Span,
    },
    Function(FunctionDecl),
    Class(ClassDecl),
    Enum(EnumDecl),
    Return {
        value: Option<Expr>,
        span: Span,
    },
    Break {
        span: Span,
    },
    Continue {
        span: Span,
    },
    While {
        condition: Expr,
        body: BlockExpr,
        span: Span,
    },
    ForRange {
        var_name: String,
        start: Expr,
        end: Expr,
        body: BlockExpr,
        span: Span,
    },
    Expression {
        expr: Expr,
        span: Span,
    },
}

#[derive(Debug, Clone)]
pub struct FunctionDecl {
    pub name: String,
    pub type_params: Vec<String>,
    pub params: Vec<Param>,
    pub return_type: Option<TypeExpr>,
    pub body: BlockExpr,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct ClassDecl {
    pub name: String,
    pub type_params: Vec<String>,
    pub fields: Vec<FieldDecl>,
    pub methods: Vec<FunctionDecl>,
    pub init: Option<FunctionDecl>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct EnumDecl {
    pub name: String,
    pub variants: Vec<String>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct FieldDecl {
    pub name: String,
    pub ty: TypeExpr,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct Param {
    pub name: String,
    pub ty: TypeExpr,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct BlockExpr {
    pub statements: Vec<Stmt>,
    pub tail: Option<Box<Expr>>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum Expr {
    Int(i64, Span),
    Float(f64, Span),
    Bool(bool, Span),
    String(String, Span),
    Variable(String, Span),
    SelfRef(Span),
    ArrayLiteral(Vec<Expr>, Span),
    Block(BlockExpr),
    Grouping(Box<Expr>, Span),
    Unary {
        op: UnaryOp,
        expr: Box<Expr>,
        span: Span,
    },
    Binary {
        left: Box<Expr>,
        op: BinaryOp,
        right: Box<Expr>,
        span: Span,
    },
    VecBinary {
        left: Box<Expr>,
        op: BinaryOp,
        right: Box<Expr>,
        len: usize,
        mode: VecBinaryMode,
        span: Span,
    },
    If {
        condition: Box<Expr>,
        then_branch: BlockExpr,
        else_branch: BlockExpr,
        span: Span,
    },
    Unless {
        condition: Box<Expr>,
        then_branch: BlockExpr,
        else_branch: Option<BlockExpr>,
        span: Span,
    },
    Match {
        scrutinee: Box<Expr>,
        arms: Vec<MatchArm>,
        span: Span,
    },
    Call {
        callee: Box<Expr>,
        args: Vec<Expr>,
        span: Span,
    },
    EnumVariant {
        enum_name: String,
        variant_name: String,
        span: Span,
    },
    Pipeline {
        lhs: Box<Expr>,
        callee: Box<Expr>,
        args: Vec<Expr>,
        span: Span,
    },
    Property {
        object: Box<Expr>,
        name: String,
        span: Span,
    },
    Index {
        target: Box<Expr>,
        index: Box<Expr>,
        span: Span,
    },
    GridIndex {
        target: Box<Expr>,
        x: Box<Expr>,
        y: Box<Expr>,
        span: Span,
    },
    Assign {
        target: AssignTarget,
        value: Box<Expr>,
        span: Span,
    },
    Try {
        expr: Box<Expr>,
        span: Span,
    },
    StringInterpolation {
        parts: Vec<InterpolationPart>,
        span: Span,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VecBinaryMode {
    ArrayArray,
    ArrayScalarRight,
    ScalarArrayLeft,
}

#[derive(Debug, Clone)]
pub enum AssignTarget {
    Variable(String, Span),
    Property {
        object: Box<Expr>,
        name: String,
        span: Span,
    },
    Index {
        target: Box<Expr>,
        index: Box<Expr>,
        span: Span,
    },
    GridIndex {
        target: Box<Expr>,
        x: Box<Expr>,
        y: Box<Expr>,
        span: Span,
    },
}

#[derive(Debug, Clone)]
pub enum InterpolationPart {
    Text(String),
    Expr(Expr),
}

#[derive(Debug, Clone)]
pub struct MatchArm {
    pub pattern: MatchPattern,
    pub expr: Expr,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum MatchPattern {
    EnumVariant {
        enum_name: String,
        variant_name: String,
    },
    Variant(String),
    Wildcard,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Negate,
    Not,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    And,
    Or,
    Add,
    Subtract,
    Multiply,
    Divide,
    Equal,
    NotEqual,
    Greater,
    GreaterEqual,
    Less,
    LessEqual,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TypeExpr {
    Int,
    Float,
    String,
    Bool,
    Void,
    Named(String),
    Applied {
        name: String,
        args: Vec<TypeExpr>,
    },
    Option(Box<TypeExpr>),
    Array {
        element: Box<TypeExpr>,
        len: usize,
    },
    Grid {
        element: Box<TypeExpr>,
        width: usize,
        height: usize,
    },
}

impl Expr {
    pub fn span(&self) -> Span {
        match self {
            Expr::Int(_, span)
            | Expr::Float(_, span)
            | Expr::Bool(_, span)
            | Expr::String(_, span)
            | Expr::Variable(_, span)
            | Expr::SelfRef(span)
            | Expr::ArrayLiteral(_, span)
            | Expr::Grouping(_, span)
            | Expr::Try { span, .. }
            | Expr::StringInterpolation { span, .. } => *span,
            Expr::Block(block) => block.span,
            Expr::Unary { span, .. }
            | Expr::Binary { span, .. }
            | Expr::VecBinary { span, .. }
            | Expr::If { span, .. }
            | Expr::Unless { span, .. }
            | Expr::Match { span, .. }
            | Expr::Call { span, .. }
            | Expr::Pipeline { span, .. }
            | Expr::EnumVariant { span, .. }
            | Expr::Property { span, .. }
            | Expr::Index { span, .. }
            | Expr::GridIndex { span, .. }
            | Expr::Assign { span, .. } => *span,
        }
    }
}
