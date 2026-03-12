use std::collections::{BTreeSet, HashMap};

use crate::ast::{
    BinaryOp, Block, Expr, ExprKind, FunctionDecl, NodeId, Program, Stmt, StmtKind, TypeExpr,
    UnaryOp,
};
use crate::error::MuninnError;
use crate::native::{
    NativeFunctionKind, NativeSignature, NativeType, format_native_overload, native_by_kind,
    native_by_name, registered_natives,
};
use crate::span::Span;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Ty {
    Int,
    Float,
    Bool,
    String,
    Tensor,
    Void,
    Function(Vec<Ty>, Box<Ty>),
    NativeFunction(NativeFunctionKind),
    Error,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymbolKind {
    Global,
    Local,
    Parameter,
    Function,
    NativeFunction(NativeFunctionKind),
}

#[derive(Debug, Clone)]
pub struct Symbol {
    pub id: usize,
    pub name: String,
    pub kind: SymbolKind,
    pub span: Span,
    pub detail: String,
    pub ty: Ty,
    pub mutable: bool,
}

#[derive(Debug, Clone)]
pub struct Reference {
    pub span: Span,
    pub target: usize,
}

#[derive(Debug, Clone, Default)]
pub struct SemanticModel {
    pub diagnostics: Vec<MuninnError>,
    pub expr_types: HashMap<NodeId, Ty>,
    pub symbols: Vec<Symbol>,
    pub references: Vec<Reference>,
}

impl SemanticModel {
    pub fn ty_for_expr(&self, id: NodeId) -> Option<&Ty> {
        self.expr_types.get(&id)
    }

    pub fn symbol_by_id(&self, id: usize) -> Option<&Symbol> {
        self.symbols.get(id)
    }

    pub fn symbol_at_offset(&self, offset: usize) -> Option<&Symbol> {
        self.symbols
            .iter()
            .filter(|symbol| symbol.span.contains_offset(offset))
            .min_by_key(|symbol| symbol.span.width().max(1))
    }

    pub fn reference_at_offset(&self, offset: usize) -> Option<&Reference> {
        self.references
            .iter()
            .filter(|reference| reference.span.contains_offset(offset))
            .min_by_key(|reference| reference.span.width().max(1))
    }

    pub fn definition_at_offset(&self, offset: usize) -> Option<&Symbol> {
        if let Some(reference) = self.reference_at_offset(offset) {
            return self.symbol_by_id(reference.target);
        }
        self.symbol_at_offset(offset)
    }
}

pub fn analyze_program(program: &Program) -> SemanticModel {
    let mut analyzer = Analyzer::new();
    analyzer.analyze(program);
    analyzer.finish()
}

pub fn check_program(program: &Program) -> Result<SemanticModel, Vec<MuninnError>> {
    let model = analyze_program(program);
    if model.diagnostics.is_empty() {
        Ok(model)
    } else {
        Err(model.diagnostics.clone())
    }
}

struct Analyzer {
    model: SemanticModel,
    scopes: Vec<HashMap<String, usize>>,
    current_return: Option<Ty>,
    inside_function: bool,
}

impl Analyzer {
    fn new() -> Self {
        let mut analyzer = Self {
            model: SemanticModel::default(),
            scopes: vec![HashMap::new()],
            current_return: None,
            inside_function: false,
        };
        analyzer.install_natives();
        analyzer
    }

    fn finish(self) -> SemanticModel {
        self.model
    }

    fn analyze(&mut self, program: &Program) {
        self.collect_functions(program);
        for statement in &program.statements {
            self.check_stmt(statement, true);
        }
    }

    fn install_natives(&mut self) {
        for native in registered_natives() {
            let symbol_id = self.model.symbols.len();
            self.scopes[0].insert(native.name.to_string(), symbol_id);
            self.model.symbols.push(Symbol {
                id: symbol_id,
                name: native.name.to_string(),
                kind: SymbolKind::NativeFunction(native.kind),
                span: Span::default(),
                detail: native.detail.to_string(),
                ty: Ty::NativeFunction(native.kind),
                mutable: false,
            });
        }
    }

    fn collect_functions(&mut self, program: &Program) {
        for statement in &program.statements {
            if let StmtKind::Function(function) = &statement.kind {
                let symbol = Symbol {
                    id: self.model.symbols.len(),
                    name: function.name.clone(),
                    kind: SymbolKind::Function,
                    span: function.name_span,
                    detail: format_function_signature(function),
                    ty: Ty::Function(
                        function
                            .params
                            .iter()
                            .map(|param| ty_from_type_expr(param.ty))
                            .collect(),
                        Box::new(ty_from_type_expr(function.return_type)),
                    ),
                    mutable: false,
                };
                self.define_global(symbol);
            }
        }
    }

    fn check_stmt(&mut self, stmt: &Stmt, top_level: bool) {
        match &stmt.kind {
            StmtKind::Let {
                name,
                name_span,
                mutable,
                ty,
                initializer,
            } => {
                let initializer_ty = self.check_expr(initializer);
                let declared_ty = ty.map(ty_from_type_expr);
                let final_ty = match declared_ty {
                    Some(expected) => {
                        if !self.ty_compatible(&expected, &initializer_ty) {
                            self.error(
                                initializer.span,
                                format!(
                                    "expected initializer of type {}, got {}",
                                    display_ty(&expected),
                                    display_ty(&initializer_ty)
                                ),
                            );
                        }
                        expected
                    }
                    None => initializer_ty.clone(),
                };
                let kind = if self.scopes.len() == 1 {
                    SymbolKind::Global
                } else {
                    SymbolKind::Local
                };
                let detail = format!("{}: {}", name, display_ty(&final_ty));
                self.define_symbol(name.clone(), kind, *name_span, detail, final_ty, *mutable);
            }
            StmtKind::Function(function) => {
                if !top_level {
                    self.error(stmt.span, "nested functions are not supported".to_string());
                    return;
                }
                self.check_function(function);
            }
            StmtKind::Return(value) => {
                if !self.inside_function {
                    self.error(
                        stmt.span,
                        "return can only appear inside a function".to_string(),
                    );
                    return;
                }
                let actual = value
                    .as_ref()
                    .map(|expr| self.check_expr(expr))
                    .unwrap_or(Ty::Void);
                let expected = self.current_return.clone().unwrap_or(Ty::Void);
                if !self.ty_compatible(&expected, &actual) {
                    self.error(
                        stmt.span,
                        format!(
                            "return type mismatch: expected {}, got {}",
                            display_ty(&expected),
                            display_ty(&actual)
                        ),
                    );
                }
            }
            StmtKind::While { condition, body } => {
                let condition_ty = self.check_expr(condition);
                if condition_ty != Ty::Bool && condition_ty != Ty::Error {
                    self.error(condition.span, "while condition must be Bool".to_string());
                }
                self.check_block(body);
            }
            StmtKind::If {
                condition,
                then_branch,
                else_branch,
            } => {
                let condition_ty = self.check_expr(condition);
                if condition_ty != Ty::Bool && condition_ty != Ty::Error {
                    self.error(condition.span, "if condition must be Bool".to_string());
                }
                self.check_block(then_branch);
                if let Some(block) = else_branch {
                    self.check_block(block);
                }
            }
            StmtKind::Assign {
                name,
                name_span,
                value,
            } => {
                let value_ty = self.check_expr(value);
                match self.lookup_symbol(name) {
                    Some(symbol_id) => {
                        let symbol = self.model.symbols[symbol_id].clone();
                        self.model.references.push(Reference {
                            span: *name_span,
                            target: symbol_id,
                        });
                        if !symbol.mutable {
                            self.error(*name_span, format!("'{}' is not mutable", name));
                        }
                        if matches!(
                            symbol.kind,
                            SymbolKind::Function | SymbolKind::NativeFunction(_)
                        ) {
                            self.error(*name_span, format!("cannot assign to '{}'", name));
                        } else if !self.ty_compatible(&symbol.ty, &value_ty) {
                            self.error(
                                *name_span,
                                format!(
                                    "cannot assign {} to {} of type {}",
                                    display_ty(&value_ty),
                                    name,
                                    display_ty(&symbol.ty)
                                ),
                            );
                        }
                    }
                    None => {
                        self.error(*name_span, format!("unknown variable '{}'", name));
                    }
                }
            }
            StmtKind::Expr(expr) => {
                self.check_expr(expr);
            }
        }
    }

    fn check_function(&mut self, function: &FunctionDecl) {
        let previous_return = self.current_return.clone();
        let previous_inside_function = self.inside_function;
        self.current_return = Some(ty_from_type_expr(function.return_type));
        self.inside_function = true;
        self.enter_scope();
        for param in &function.params {
            let ty = ty_from_type_expr(param.ty);
            self.define_symbol(
                param.name.clone(),
                SymbolKind::Parameter,
                param.span,
                format!("{}: {}", param.name, display_ty(&ty)),
                ty,
                false,
            );
        }
        self.check_block(&function.body);
        let expected_return = self.current_return.clone().unwrap_or(Ty::Void);
        if expected_return != Ty::Void && !self.block_guarantees_return(&function.body) {
            self.error(
                function.name_span,
                format!(
                    "function '{}' may fall through without returning {}",
                    function.name,
                    display_ty(&expected_return)
                ),
            );
        }
        self.exit_scope();
        self.current_return = previous_return;
        self.inside_function = previous_inside_function;
    }

    fn check_block(&mut self, block: &Block) -> Ty {
        self.enter_scope();
        let mut reached_terminal = false;
        for statement in &block.statements {
            if reached_terminal {
                self.error(statement.span, "unreachable statement".to_string());
                continue;
            }
            self.check_stmt(statement, false);
            if self.stmt_guarantees_return(statement) {
                reached_terminal = true;
            }
        }

        let value_ty = if let Some(value) = &block.value {
            if reached_terminal {
                self.error(value.span, "unreachable block value".to_string());
                Ty::Error
            } else {
                self.check_expr(value)
            }
        } else {
            Ty::Void
        };
        self.exit_scope();
        value_ty
    }

    fn check_expr(&mut self, expr: &Expr) -> Ty {
        let ty = match &expr.kind {
            ExprKind::Int(_) => Ty::Int,
            ExprKind::Float(_) => Ty::Float,
            ExprKind::Bool(_) => Ty::Bool,
            ExprKind::String(_) => Ty::String,
            ExprKind::Variable(name) => match self.lookup_symbol(name) {
                Some(symbol_id) => {
                    self.model.references.push(Reference {
                        span: expr.span,
                        target: symbol_id,
                    });
                    self.model.symbols[symbol_id].ty.clone()
                }
                None => {
                    self.error(expr.span, format!("unknown name '{}'", name));
                    Ty::Error
                }
            },
            ExprKind::Grouping(inner) => self.check_expr(inner),
            ExprKind::Block(block) => self.check_block(block),
            ExprKind::Unary { op, expr: inner } => {
                let inner_ty = self.check_expr(inner);
                match op {
                    UnaryOp::Negate => {
                        if matches!(inner_ty, Ty::Int | Ty::Float | Ty::Error) {
                            inner_ty
                        } else {
                            self.error(expr.span, "unary '-' expects Int or Float".to_string());
                            Ty::Error
                        }
                    }
                    UnaryOp::Not => {
                        if matches!(inner_ty, Ty::Bool | Ty::Error) {
                            Ty::Bool
                        } else {
                            self.error(expr.span, "unary '!' expects Bool".to_string());
                            Ty::Error
                        }
                    }
                }
            }
            ExprKind::Binary { left, op, right } => {
                let left_ty = self.check_expr(left);
                let right_ty = self.check_expr(right);
                self.check_binary(*op, &left_ty, &right_ty, expr.span)
            }
            ExprKind::Call { callee, args } => {
                let callee_ty = self.check_expr(callee);
                let arg_types = args
                    .iter()
                    .map(|arg| self.check_expr(arg))
                    .collect::<Vec<_>>();
                self.check_call(callee, &callee_ty, &arg_types, expr.span)
            }
            ExprKind::If {
                condition,
                then_branch,
                else_branch,
            } => {
                let condition_ty = self.check_expr(condition);
                if condition_ty != Ty::Bool && condition_ty != Ty::Error {
                    self.error(condition.span, "if condition must be Bool".to_string());
                }
                let then_ty = self.check_block(then_branch);
                let else_ty = self.check_block(else_branch);
                if self.ty_compatible(&then_ty, &else_ty) {
                    then_ty
                } else if matches!(then_ty, Ty::Error) || matches!(else_ty, Ty::Error) {
                    Ty::Error
                } else {
                    self.error(
                        expr.span,
                        format!(
                            "if expression branches must return the same type, got {} and {}",
                            display_ty(&then_ty),
                            display_ty(&else_ty)
                        ),
                    );
                    Ty::Error
                }
            }
        };
        self.model.expr_types.insert(expr.id, ty.clone());
        ty
    }

    fn check_binary(&mut self, op: BinaryOp, left: &Ty, right: &Ty, span: Span) -> Ty {
        match op {
            BinaryOp::Add => {
                if left == &Ty::String && right == &Ty::String {
                    Ty::String
                } else if left == &Ty::Int && right == &Ty::Int {
                    Ty::Int
                } else if left == &Ty::Float && right == &Ty::Float {
                    Ty::Float
                } else if self.is_tensor_numeric_pair(left, right) {
                    Ty::Tensor
                } else if matches!(left, Ty::Error) || matches!(right, Ty::Error) {
                    Ty::Error
                } else {
                    self.error(
                        span,
                        format!(
                            "'+' expects Int/Int, Float/Float, String/String, Tensor/Tensor, or Tensor/Float, got {} and {}",
                            display_ty(left),
                            display_ty(right)
                        ),
                    );
                    Ty::Error
                }
            }
            BinaryOp::Subtract | BinaryOp::Multiply | BinaryOp::Divide => {
                if left == &Ty::Int && right == &Ty::Int {
                    Ty::Int
                } else if left == &Ty::Float && right == &Ty::Float {
                    Ty::Float
                } else if self.is_tensor_numeric_pair(left, right) {
                    Ty::Tensor
                } else if matches!(left, Ty::Error) || matches!(right, Ty::Error) {
                    Ty::Error
                } else {
                    self.error(
                        span,
                        format!(
                            "numeric operator expects matching numeric types or Tensor/Float combinations, got {} and {}",
                            display_ty(left),
                            display_ty(right)
                        ),
                    );
                    Ty::Error
                }
            }
            BinaryOp::Equal | BinaryOp::NotEqual => {
                if self.scalar_equality_compatible(left, right) {
                    Ty::Bool
                } else if matches!(left, Ty::Error) || matches!(right, Ty::Error) {
                    Ty::Error
                } else {
                    self.error(
                        span,
                        format!(
                            "comparison expects matching scalar or String types, got {} and {}",
                            display_ty(left),
                            display_ty(right)
                        ),
                    );
                    Ty::Error
                }
            }
            BinaryOp::Greater | BinaryOp::GreaterEqual | BinaryOp::Less | BinaryOp::LessEqual => {
                if (left == &Ty::Int && right == &Ty::Int)
                    || (left == &Ty::Float && right == &Ty::Float)
                {
                    Ty::Bool
                } else if matches!(left, Ty::Error) || matches!(right, Ty::Error) {
                    Ty::Error
                } else {
                    self.error(
                        span,
                        format!(
                            "ordering comparison expects Int/Int or Float/Float, got {} and {}",
                            display_ty(left),
                            display_ty(right)
                        ),
                    );
                    Ty::Error
                }
            }
            BinaryOp::And | BinaryOp::Or => {
                if left == &Ty::Bool && right == &Ty::Bool {
                    Ty::Bool
                } else if matches!(left, Ty::Error) || matches!(right, Ty::Error) {
                    Ty::Error
                } else {
                    self.error(
                        span,
                        format!(
                            "logical operator expects Bool/Bool, got {} and {}",
                            display_ty(left),
                            display_ty(right)
                        ),
                    );
                    Ty::Error
                }
            }
        }
    }

    fn check_call(&mut self, callee: &Expr, callee_ty: &Ty, arg_types: &[Ty], span: Span) -> Ty {
        match callee_ty {
            Ty::Function(params, ret) => {
                if params.len() != arg_types.len() {
                    self.error(
                        span,
                        format!(
                            "wrong argument count: expected {}, got {}",
                            params.len(),
                            arg_types.len()
                        ),
                    );
                }
                for (index, (expected, actual)) in params.iter().zip(arg_types.iter()).enumerate() {
                    if !self.ty_compatible(expected, actual) {
                        self.error(
                            span,
                            format!(
                                "argument {} mismatch: expected {}, got {}",
                                index,
                                display_ty(expected),
                                display_ty(actual)
                            ),
                        );
                    }
                }
                ret.as_ref().clone()
            }
            Ty::NativeFunction(kind) => self.check_native_call(*kind, arg_types, span),
            Ty::Error => Ty::Error,
            other => {
                let native_hint = if let ExprKind::Variable(name) = &callee.kind {
                    native_by_name(name)
                        .map(|native| format!(" (did you mean native function '{}'? )", native.name))
                        .unwrap_or_default()
                } else {
                    String::new()
                };
                self.error(
                    span,
                    format!(
                        "value of type {} is not callable{}",
                        display_ty(other),
                        native_hint
                    ),
                );
                Ty::Error
            }
        }
    }

    fn check_native_call(&mut self, kind: NativeFunctionKind, arg_types: &[Ty], span: Span) -> Ty {
        if arg_types.iter().any(|ty| matches!(ty, Ty::Error)) {
            return Ty::Error;
        }

        let native = native_by_kind(kind);
        if let Some(signature) = native
            .signatures
            .iter()
            .find(|signature| native_signature_matches(signature, arg_types))
        {
            return ty_from_native_type(signature.return_type);
        }

        let matching_arity = native
            .signatures
            .iter()
            .filter(|signature| signature.params.len() == arg_types.len())
            .collect::<Vec<_>>();

        if matching_arity.is_empty() {
            let mut counts = native
                .signatures
                .iter()
                .map(|signature| signature.params.len())
                .collect::<Vec<_>>();
            counts.sort_unstable();
            counts.dedup();
            let expected = counts
                .iter()
                .map(|count| count.to_string())
                .collect::<Vec<_>>()
                .join(" or ");
            self.error(
                span,
                format!(
                    "{} expects {} argument(s), got {}",
                    native.name,
                    expected,
                    arg_types.len()
                ),
            );
            return Ty::Error;
        }

        for (index, actual) in arg_types.iter().enumerate() {
            let expected = expected_native_types(&matching_arity, index);
            if !expected.iter().any(|ty| native_type_matches_ty(*ty, actual)) {
                let expected = expected
                    .iter()
                    .map(|ty| native_type_name(*ty).to_string())
                    .collect::<Vec<_>>()
                    .join(" | ");
                self.error(
                    span,
                    format!(
                        "{} argument {} expects {}, got {}",
                        native.name,
                        index,
                        expected,
                        display_ty(actual)
                    ),
                );
                return Ty::Error;
            }
        }

        self.error(
            span,
            format!(
                "no matching overload for {}",
                format_native_overload(
                    native.name,
                    &arg_types
                        .iter()
                        .map(native_type_from_ty)
                        .collect::<Vec<_>>()
                )
            ),
        );
        Ty::Error
    }

    fn block_guarantees_return(&self, block: &Block) -> bool {
        for statement in &block.statements {
            if self.stmt_guarantees_return(statement) {
                return true;
            }
        }
        false
    }

    fn stmt_guarantees_return(&self, stmt: &Stmt) -> bool {
        match &stmt.kind {
            StmtKind::Return(_) => true,
            StmtKind::If {
                then_branch,
                else_branch,
                ..
            } => else_branch.as_ref().is_some_and(|else_branch| {
                self.block_guarantees_return(then_branch)
                    && self.block_guarantees_return(else_branch)
            }),
            _ => false,
        }
    }

    fn scalar_equality_compatible(&self, left: &Ty, right: &Ty) -> bool {
        matches!(
            (left, right),
            (Ty::Int, Ty::Int)
                | (Ty::Float, Ty::Float)
                | (Ty::Bool, Ty::Bool)
                | (Ty::String, Ty::String)
        )
    }

    fn is_tensor_numeric_pair(&self, left: &Ty, right: &Ty) -> bool {
        matches!(
            (left, right),
            (Ty::Tensor, Ty::Tensor)
                | (Ty::Tensor, Ty::Float)
                | (Ty::Float, Ty::Tensor)
        )
    }

    fn ty_compatible(&self, expected: &Ty, actual: &Ty) -> bool {
        expected == actual || matches!(expected, Ty::Error) || matches!(actual, Ty::Error)
    }

    fn define_global(&mut self, symbol: Symbol) {
        if self.scopes[0].contains_key(&symbol.name) {
            self.error(symbol.span, format!("'{}' is already defined", symbol.name));
            return;
        }
        let id = symbol.id;
        self.scopes[0].insert(symbol.name.clone(), id);
        self.model.symbols.push(symbol);
    }

    fn define_symbol(
        &mut self,
        name: String,
        kind: SymbolKind,
        span: Span,
        detail: String,
        ty: Ty,
        mutable: bool,
    ) {
        if self
            .scopes
            .last()
            .is_some_and(|scope| scope.contains_key(&name))
        {
            self.error(span, format!("'{}' is already defined in this scope", name));
            return;
        }
        let id = self.model.symbols.len();
        self.scopes
            .last_mut()
            .expect("scope")
            .insert(name.clone(), id);
        self.model.symbols.push(Symbol {
            id,
            name,
            kind,
            span,
            detail,
            ty,
            mutable,
        });
    }

    fn lookup_symbol(&self, name: &str) -> Option<usize> {
        self.scopes
            .iter()
            .rev()
            .find_map(|scope| scope.get(name).copied())
    }

    fn enter_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    fn exit_scope(&mut self) {
        self.scopes.pop();
    }

    fn error(&mut self, span: Span, message: String) {
        self.model
            .diagnostics
            .push(MuninnError::new("typecheck", message, span));
    }
}

fn format_function_signature(function: &FunctionDecl) -> String {
    let params = function
        .params
        .iter()
        .map(|param| {
            format!(
                "{}: {}",
                param.name,
                display_ty(&ty_from_type_expr(param.ty))
            )
        })
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "fn {}({}) -> {}",
        function.name,
        params,
        display_ty(&ty_from_type_expr(function.return_type))
    )
}

pub fn display_ty(ty: &Ty) -> String {
    match ty {
        Ty::Int => "Int".to_string(),
        Ty::Float => "Float".to_string(),
        Ty::Bool => "Bool".to_string(),
        Ty::String => "String".to_string(),
        Ty::Tensor => "Tensor".to_string(),
        Ty::Void => "Void".to_string(),
        Ty::Function(params, ret) => format!(
            "fn({}) -> {}",
            params.iter().map(display_ty).collect::<Vec<_>>().join(", "),
            display_ty(ret)
        ),
        Ty::NativeFunction(kind) => {
            format!("native fn {}", native_by_kind(*kind).name)
        }
        Ty::Error => "<error>".to_string(),
    }
}

pub fn ty_from_type_expr(ty: TypeExpr) -> Ty {
    match ty {
        TypeExpr::Int => Ty::Int,
        TypeExpr::Float => Ty::Float,
        TypeExpr::Bool => Ty::Bool,
        TypeExpr::String => Ty::String,
        TypeExpr::Tensor => Ty::Tensor,
        TypeExpr::Void => Ty::Void,
    }
}

fn ty_from_native_type(ty: NativeType) -> Ty {
    match ty {
        NativeType::Int => Ty::Int,
        NativeType::Float => Ty::Float,
        NativeType::Bool => Ty::Bool,
        NativeType::String => Ty::String,
        NativeType::Tensor => Ty::Tensor,
        NativeType::Void => Ty::Void,
    }
}

fn native_type_from_ty(ty: &Ty) -> NativeType {
    match ty {
        Ty::Int => NativeType::Int,
        Ty::Float => NativeType::Float,
        Ty::Bool => NativeType::Bool,
        Ty::String => NativeType::String,
        Ty::Tensor => NativeType::Tensor,
        Ty::Void | Ty::Function(_, _) | Ty::NativeFunction(_) | Ty::Error => NativeType::Void,
    }
}

fn native_signature_matches(signature: &NativeSignature, arg_types: &[Ty]) -> bool {
    signature.params.len() == arg_types.len()
        && signature
            .params
            .iter()
            .zip(arg_types.iter())
            .all(|(expected, actual)| native_type_matches_ty(*expected, actual))
}

fn native_type_matches_ty(expected: NativeType, actual: &Ty) -> bool {
    match expected {
        NativeType::Int => actual == &Ty::Int,
        NativeType::Float => actual == &Ty::Float,
        NativeType::Bool => actual == &Ty::Bool,
        NativeType::String => actual == &Ty::String,
        NativeType::Tensor => actual == &Ty::Tensor,
        NativeType::Void => actual == &Ty::Void,
    }
}

fn expected_native_types(signatures: &[&NativeSignature], index: usize) -> Vec<NativeType> {
    let mut types = BTreeSet::new();
    for signature in signatures {
        if let Some(ty) = signature.params.get(index) {
            types.insert(*ty);
        }
    }
    types.into_iter().collect()
}

fn native_type_name(ty: NativeType) -> &'static str {
    match ty {
        NativeType::Int => "Int",
        NativeType::Float => "Float",
        NativeType::Bool => "Bool",
        NativeType::String => "String",
        NativeType::Void => "Void",
        NativeType::Tensor => "Tensor",
    }
}

#[cfg(test)]
mod tests {
    use crate::frontend::parse_document;

    use super::{Ty, analyze_program};

    #[test]
    fn records_expression_types_by_node_id() {
        let program = parse_document("let x: Int = 1 + 2;").expect("program");
        let model = analyze_program(&program);
        let expr = match &program.statements[0].kind {
            crate::ast::StmtKind::Let { initializer, .. } => initializer,
            _ => panic!("expected let"),
        };
        assert_eq!(model.ty_for_expr(expr.id), Some(&Ty::Int));
    }

    #[test]
    fn reports_mismatched_assignment() {
        let program = parse_document("let mut x: Int = 1; x = true;").expect("program");
        let model = analyze_program(&program);
        assert!(!model.diagnostics.is_empty());
    }

    #[test]
    fn reports_missing_non_void_return() {
        let program = parse_document(
            "fn maybe(flag: Bool) -> Int { if (flag) { return 1; } } let out: Int = maybe(false);",
        )
        .expect("program");
        let model = analyze_program(&program);
        assert!(model
            .diagnostics
            .iter()
            .any(|error| error.message.contains("may fall through without returning Int")));
    }

    #[test]
    fn validates_assert_native_argument_type() {
        let program = parse_document("assert(1);").expect("program");
        let model = analyze_program(&program);
        assert!(model
            .diagnostics
            .iter()
            .any(|error| error.message.contains("assert argument 0 expects Bool")));
    }

    #[test]
    fn flags_unreachable_statement_after_return() {
        let program =
            parse_document("fn main() -> Int { return 1; let x: Int = 2; }").expect("program");
        let model = analyze_program(&program);
        assert!(model
            .diagnostics
            .iter()
            .any(|error| error.message == "unreachable statement"));
    }

    #[test]
    fn checks_if_expression_branch_types() {
        let program = parse_document("let x = if (true) { 1 } else { 2.0 };").expect("program");
        let model = analyze_program(&program);
        assert!(model
            .diagnostics
            .iter()
            .any(|error| error.message.contains("if expression branches must return the same type")));
    }
}
