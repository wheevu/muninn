use crate::span::Span;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId(pub u32);

#[derive(Debug, Clone)]
pub struct Program {
    pub statements: Vec<Stmt>,
}

#[derive(Debug, Clone)]
pub struct Stmt {
    pub id: NodeId,
    pub kind: StmtKind,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum StmtKind {
    Let {
        name: String,
        name_span: Span,
        mutable: bool,
        ty: Option<TypeExpr>,
        initializer: Expr,
    },
    Function(FunctionDecl),
    Record(RecordDecl),
    Return(Option<Expr>),
    While {
        condition: Expr,
        body: Block,
    },
    If {
        condition: Expr,
        then_branch: Block,
        else_branch: Option<Block>,
    },
    Assign {
        name: String,
        name_span: Span,
        value: Expr,
    },
    Expr(Expr),
}

#[derive(Debug, Clone)]
pub struct FunctionDecl {
    pub id: NodeId,
    pub name: String,
    pub name_span: Span,
    pub params: Vec<Param>,
    pub return_type: TypeExpr,
    pub body: Block,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct Param {
    pub id: NodeId,
    pub name: String,
    pub ty: TypeExpr,
    pub span: Span,
}

/// A nominal record declaration: `record Point { x: Int, y: Int }`.
/// Records are top-level only, mirroring functions.
#[derive(Debug, Clone)]
pub struct RecordDecl {
    pub id: NodeId,
    pub name: String,
    pub name_span: Span,
    pub fields: Vec<RecordField>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct RecordField {
    pub id: NodeId,
    pub name: String,
    pub name_span: Span,
    pub ty: TypeExpr,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct Block {
    pub id: NodeId,
    pub statements: Vec<Stmt>,
    pub value: Option<Box<Expr>>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct Expr {
    pub id: NodeId,
    pub kind: ExprKind,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum ExprKind {
    Int(i64),
    Float(f64),
    Bool(bool),
    String(String),
    Variable(String),
    Grouping(Box<Expr>),
    Block(Block),
    Unary {
        op: UnaryOp,
        expr: Box<Expr>,
    },
    Binary {
        left: Box<Expr>,
        op: BinaryOp,
        right: Box<Expr>,
    },
    Call {
        callee: Box<Expr>,
        args: Vec<Expr>,
    },
    /// A record construction: `Point { x: 1, y: 2 }`. The name span feeds
    /// go-to-definition for the record type.
    RecordLit {
        name: String,
        name_span: Span,
        fields: Vec<RecordLitField>,
    },
    /// Field read: `point.x`. Field assignment is out of scope; whole
    /// record replacement through `Assign` covers mutation.
    Field {
        base: Box<Expr>,
        field: String,
        field_span: Span,
    },
    If {
        condition: Box<Expr>,
        then_branch: Block,
        else_branch: Block,
    },
}

#[derive(Debug, Clone)]
pub struct RecordLitField {
    pub name: String,
    pub name_span: Span,
    pub value: Expr,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Negate,
    Not,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
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
    And,
    Or,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TypeExpr {
    Int,
    Float,
    Bool,
    String,
    Tensor,
    Void,
    /// A nominal record type by declaration name. Resolution against the
    /// record table happens in semantic analysis, not parsing.
    Record(String),
}
