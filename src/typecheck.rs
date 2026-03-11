use std::collections::{HashMap, HashSet};

use crate::ast::{
    AssignTarget, BinaryOp, BlockExpr, ClassDecl, Expr, FunctionDecl, Program, Stmt, TypeExpr,
    UnaryOp,
};
use crate::error::MuninnError;
use crate::span::Span;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Ty {
    Int,
    Float,
    Bool,
    String,
    Void,
    Enum(String),
    Option(Box<Ty>),
    Array(Box<Ty>, usize),
    Function(Vec<Ty>, Box<Ty>),
    Class(String),
    Instance(String),
    Unknown,
}

#[derive(Debug, Clone, Default)]
pub struct TypeContext {
    expr_types: HashMap<usize, Ty>,
}

impl TypeContext {
    fn expr_key(expr: &Expr) -> usize {
        expr as *const Expr as usize
    }

    pub fn ty_for_expr(&self, expr: &Expr) -> Option<&Ty> {
        self.expr_types.get(&Self::expr_key(expr))
    }

    fn set_expr_ty(&mut self, expr: &Expr, ty: Ty) {
        self.expr_types.insert(Self::expr_key(expr), ty);
    }
}

#[derive(Debug, Clone)]
struct Symbol {
    ty: Ty,
    mutable: bool,
}

#[derive(Debug, Clone)]
struct MethodSig {
    params: Vec<Ty>,
    ret: Ty,
}

#[derive(Debug, Clone)]
struct ClassInfo {
    fields: HashMap<String, Ty>,
    methods: HashMap<String, MethodSig>,
    init: Option<MethodSig>,
}

pub fn check_program(program: &Program) -> Result<TypeContext, Vec<MuninnError>> {
    let mut checker = TypeChecker::new();
    checker.collect_top_level(program);
    checker.check_program(program);
    if checker.errors.is_empty() {
        Ok(checker.type_context)
    } else {
        Err(checker.errors)
    }
}

struct TypeChecker {
    scopes: Vec<HashMap<String, Symbol>>,
    classes: HashMap<String, ClassInfo>,
    enums: HashMap<String, HashSet<String>>,
    errors: Vec<MuninnError>,
    current_return: Ty,
    current_class: Option<String>,
    loop_depth: usize,
    generic_params: Vec<HashSet<String>>,
    type_context: TypeContext,
}

impl TypeChecker {
    fn new() -> Self {
        let mut checker = Self {
            scopes: vec![HashMap::new()],
            classes: HashMap::new(),
            enums: HashMap::new(),
            errors: Vec::new(),
            current_return: Ty::Void,
            current_class: None,
            loop_depth: 0,
            generic_params: vec![HashSet::new()],
            type_context: TypeContext::default(),
        };

        checker.define_builtin(
            "to_string".to_string(),
            Symbol {
                ty: Ty::Function(vec![Ty::Unknown], Box::new(Ty::String)),
                mutable: false,
            },
        );
        checker.define_builtin(
            "print".to_string(),
            Symbol {
                ty: Ty::Function(vec![Ty::Unknown], Box::new(Ty::Void)),
                mutable: false,
            },
        );
        checker.define_builtin(
            "len".to_string(),
            Symbol {
                ty: Ty::Function(vec![Ty::Unknown], Box::new(Ty::Int)),
                mutable: false,
            },
        );
        checker.define_builtin(
            "sum".to_string(),
            Symbol {
                ty: Ty::Function(vec![Ty::Unknown], Box::new(Ty::Unknown)),
                mutable: false,
            },
        );
        checker.define_builtin(
            "dot".to_string(),
            Symbol {
                ty: Ty::Function(vec![Ty::Unknown, Ty::Unknown], Box::new(Ty::Unknown)),
                mutable: false,
            },
        );
        checker.define_builtin(
            "zeros".to_string(),
            Symbol {
                ty: Ty::Function(vec![Ty::Int], Box::new(Ty::Unknown)),
                mutable: false,
            },
        );
        checker.define_builtin(
            "ones".to_string(),
            Symbol {
                ty: Ty::Function(vec![Ty::Int], Box::new(Ty::Unknown)),
                mutable: false,
            },
        );
        checker.define_builtin(
            "none".to_string(),
            Symbol {
                ty: Ty::Option(Box::new(Ty::Unknown)),
                mutable: false,
            },
        );
        checker.define_builtin(
            "some".to_string(),
            Symbol {
                ty: Ty::Function(
                    vec![Ty::Unknown],
                    Box::new(Ty::Option(Box::new(Ty::Unknown))),
                ),
                mutable: false,
            },
        );
        checker.define_builtin(
            "is_none".to_string(),
            Symbol {
                ty: Ty::Function(vec![Ty::Option(Box::new(Ty::Unknown))], Box::new(Ty::Bool)),
                mutable: false,
            },
        );
        checker.define_builtin(
            "unwrap".to_string(),
            Symbol {
                ty: Ty::Function(
                    vec![Ty::Option(Box::new(Ty::Unknown))],
                    Box::new(Ty::Unknown),
                ),
                mutable: false,
            },
        );
        checker.define_builtin(
            "__none".to_string(),
            Symbol {
                ty: Ty::Option(Box::new(Ty::Unknown)),
                mutable: false,
            },
        );
        checker.define_builtin(
            "__some".to_string(),
            Symbol {
                ty: Ty::Function(
                    vec![Ty::Unknown],
                    Box::new(Ty::Option(Box::new(Ty::Unknown))),
                ),
                mutable: false,
            },
        );
        checker.define_builtin(
            "__is_none".to_string(),
            Symbol {
                ty: Ty::Function(vec![Ty::Option(Box::new(Ty::Unknown))], Box::new(Ty::Bool)),
                mutable: false,
            },
        );
        checker.define_builtin(
            "__unwrap".to_string(),
            Symbol {
                ty: Ty::Function(
                    vec![Ty::Option(Box::new(Ty::Unknown))],
                    Box::new(Ty::Unknown),
                ),
                mutable: false,
            },
        );

        checker
    }

    fn collect_top_level(&mut self, program: &Program) {
        for stmt in &program.statements {
            match stmt {
                Stmt::Function(function) => {
                    let sig = self.function_sig(function);
                    self.define(
                        function.name.clone(),
                        Symbol {
                            ty: sig,
                            mutable: false,
                        },
                        function.span,
                    );
                }
                Stmt::Class(class) => self.collect_class(class),
                Stmt::Enum(decl) => {
                    let variants: HashSet<String> = decl.variants.iter().cloned().collect();
                    self.enums.insert(decl.name.clone(), variants);
                    self.define(
                        decl.name.clone(),
                        Symbol {
                            ty: Ty::Enum(decl.name.clone()),
                            mutable: false,
                        },
                        decl.span,
                    );
                }
                _ => {}
            }
        }
    }

    fn collect_class(&mut self, class: &ClassDecl) {
        if self.classes.contains_key(&class.name) {
            self.error(
                class.span,
                format!("class '{}' already defined", class.name),
            );
            return;
        }

        let mut fields = HashMap::new();
        for field in &class.fields {
            let ty = self.resolve_type(&field.ty, field.span);
            fields.insert(field.name.clone(), ty);
        }

        let mut methods = HashMap::new();
        for method in &class.methods {
            let sig = MethodSig {
                params: method
                    .params
                    .iter()
                    .map(|param| self.resolve_type(&param.ty, param.span))
                    .collect(),
                ret: method
                    .return_type
                    .as_ref()
                    .map(|ty| self.resolve_type(ty, method.span))
                    .unwrap_or(Ty::Unknown),
            };
            methods.insert(method.name.clone(), sig);
        }

        let init = class.init.as_ref().map(|init| MethodSig {
            params: init
                .params
                .iter()
                .map(|param| self.resolve_type(&param.ty, param.span))
                .collect(),
            ret: Ty::Void,
        });

        self.classes.insert(
            class.name.clone(),
            ClassInfo {
                fields,
                methods,
                init,
            },
        );

        self.define(
            class.name.clone(),
            Symbol {
                ty: Ty::Class(class.name.clone()),
                mutable: false,
            },
            class.span,
        );
    }

    fn check_program(&mut self, program: &Program) {
        for stmt in &program.statements {
            self.check_stmt(stmt);
        }
    }

    fn check_stmt(&mut self, stmt: &Stmt) {
        match stmt {
            Stmt::Let {
                name,
                mutable,
                ty,
                initializer,
                span,
            } => {
                let actual = self.check_expr(initializer);
                let declared_ty = if let Some(ty) = ty {
                    let expected = self.resolve_type(ty, *span);
                    if !self.ty_compatible(&expected, &actual) {
                        self.error(
                            *span,
                            format!(
                                "type mismatch in declaration '{}': expected {:?}, got {:?}",
                                name, expected, actual
                            ),
                        );
                    }
                    expected
                } else if actual == Ty::Unknown {
                    self.error(
                        *span,
                        format!(
                            "cannot infer type for '{}' from initializer; add an explicit annotation",
                            name
                        ),
                    );
                    Ty::Unknown
                } else {
                    actual.clone()
                };
                self.define(
                    name.clone(),
                    Symbol {
                        ty: declared_ty,
                        mutable: *mutable,
                    },
                    *span,
                );
            }
            Stmt::Function(function) => self.check_function(function, None),
            Stmt::Class(class) => self.check_class(class),
            Stmt::Enum(_) => {}
            Stmt::Return { value, span } => {
                let actual = value
                    .as_ref()
                    .map(|expr| self.check_expr(expr))
                    .unwrap_or(Ty::Void);
                if !self.ty_compatible(&self.current_return, &actual) {
                    self.error(
                        *span,
                        format!(
                            "return type mismatch: expected {:?}, got {:?}",
                            self.current_return, actual
                        ),
                    );
                }
            }
            Stmt::While {
                condition,
                body,
                span,
            } => {
                let cond_ty = self.check_expr(condition);
                if cond_ty != Ty::Bool {
                    self.error(*span, "while condition must be Bool".to_string());
                }
                self.loop_depth += 1;
                self.check_block(body);
                self.loop_depth = self.loop_depth.saturating_sub(1);
            }
            Stmt::Break { span } => {
                if self.loop_depth == 0 {
                    self.error(*span, "break can only be used inside loops".to_string());
                }
            }
            Stmt::Continue { span } => {
                if self.loop_depth == 0 {
                    self.error(*span, "continue can only be used inside loops".to_string());
                }
            }
            Stmt::ForRange { span, .. } => {
                self.error(
                    *span,
                    "for-range should be desugared before type checking".to_string(),
                );
            }
            Stmt::Expression { expr, .. } => {
                self.check_expr(expr);
            }
        }
    }

    fn check_function(&mut self, function: &FunctionDecl, receiver: Option<&str>) {
        self.enter_scope();
        let previous_return = self.current_return.clone();
        let previous_class = self.current_class.clone();

        let params_scope: HashSet<String> = function.type_params.iter().cloned().collect();
        self.generic_params.push(params_scope);

        self.current_return = function
            .return_type
            .as_ref()
            .map(|ty| self.resolve_type(ty, function.span))
            .unwrap_or_else(|| self.infer_function_return(function));
        if let Some(class_name) = receiver {
            self.current_class = Some(class_name.to_string());
            self.define(
                "self".to_string(),
                Symbol {
                    ty: Ty::Instance(class_name.to_string()),
                    mutable: false,
                },
                function.span,
            );
        }

        for param in &function.params {
            let ty = self.resolve_type(&param.ty, param.span);
            self.define(
                param.name.clone(),
                Symbol { ty, mutable: false },
                param.span,
            );
        }

        self.check_block(&function.body);

        self.current_return = previous_return;
        self.current_class = previous_class;
        self.generic_params.pop();
        self.exit_scope();
    }

    fn check_class(&mut self, class: &ClassDecl) {
        for method in &class.methods {
            self.check_function(method, Some(&class.name));
        }
        if let Some(init) = &class.init {
            self.check_function(init, Some(&class.name));
        }
    }

    fn check_block(&mut self, block: &BlockExpr) -> Ty {
        self.enter_scope();
        for stmt in &block.statements {
            self.check_stmt(stmt);
        }
        let tail_ty = block
            .tail
            .as_ref()
            .map(|expr| self.check_expr(expr))
            .unwrap_or(Ty::Void);
        self.exit_scope();
        tail_ty
    }

    fn check_expr(&mut self, expr: &Expr) -> Ty {
        let ty = match expr {
            Expr::Int(..) => Ty::Int,
            Expr::Float(..) => Ty::Float,
            Expr::Bool(..) => Ty::Bool,
            Expr::String(..) => Ty::String,
            Expr::Variable(name, span) => self.lookup(name, *span).unwrap_or(Ty::Unknown),
            Expr::SelfRef(span) => self.lookup("self", *span).unwrap_or(Ty::Unknown),
            Expr::ArrayLiteral(items, span) => {
                if items.is_empty() {
                    self.error(*span, "array literal cannot be empty".to_string());
                    Ty::Unknown
                } else {
                    let first_ty = self.check_expr(&items[0]);
                    for item in items.iter().skip(1) {
                        let ty = self.check_expr(item);
                        if !self.ty_compatible(&first_ty, &ty) {
                            self.error(*span, "array literal must be homogeneous".to_string());
                        }
                    }
                    Ty::Array(Box::new(first_ty), items.len())
                }
            }
            Expr::Block(block) => self.check_block(block),
            Expr::Grouping(inner, _) => self.check_expr(inner),
            Expr::Unary { op, expr, span } => {
                let ty = self.check_expr(expr);
                match op {
                    UnaryOp::Negate => {
                        if ty != Ty::Int && ty != Ty::Float {
                            self.error(*span, "negation expects Int or Float".to_string());
                            Ty::Unknown
                        } else {
                            ty
                        }
                    }
                    UnaryOp::Not => {
                        if ty != Ty::Bool {
                            self.error(*span, "logical not expects Bool".to_string());
                            Ty::Unknown
                        } else {
                            Ty::Bool
                        }
                    }
                }
            }
            Expr::Binary {
                left,
                op,
                right,
                span,
            } => {
                let left_ty = self.check_expr(left);
                let right_ty = self.check_expr(right);
                self.check_binary(*op, left_ty, right_ty, *span)
            }
            Expr::VecBinary {
                left,
                op,
                right,
                len,
                span,
                ..
            } => {
                let left_ty = self.check_expr(left);
                let right_ty = self.check_expr(right);
                let vec_ty = self.check_binary(*op, left_ty, right_ty, *span);
                match vec_ty {
                    Ty::Array(_, actual_len) if actual_len == *len => vec_ty,
                    Ty::Array(_, actual_len) => {
                        self.error(
                            *span,
                            format!(
                                "vectorized op length mismatch in lowered node: expected {}, got {}",
                                len, actual_len
                            ),
                        );
                        Ty::Unknown
                    }
                    _ => Ty::Unknown,
                }
            }
            Expr::If {
                condition,
                then_branch,
                else_branch,
                span,
            } => {
                let cond_ty = self.check_expr(condition);
                if cond_ty != Ty::Bool {
                    self.error(*span, "if condition must be Bool".to_string());
                }
                let then_ty = self.check_block(then_branch);
                let else_ty = self.check_block(else_branch);
                if !self.ty_compatible(&then_ty, &else_ty) {
                    self.error(
                        *span,
                        format!(
                            "if branches must have same type, got {:?} and {:?}",
                            then_ty, else_ty
                        ),
                    );
                    Ty::Unknown
                } else {
                    then_ty
                }
            }
            Expr::Unless { span, .. } => {
                self.error(
                    *span,
                    "unless should be desugared before type checking".to_string(),
                );
                Ty::Unknown
            }
            Expr::Match { span, .. } => {
                self.error(
                    *span,
                    "match should be desugared before type checking".to_string(),
                );
                Ty::Unknown
            }
            Expr::Call { callee, args, span } => {
                let arg_types = args
                    .iter()
                    .map(|arg| self.check_expr(arg))
                    .collect::<Vec<_>>();

                if let Expr::Variable(name, _) = callee.as_ref() {
                    match name.as_str() {
                        "len" => {
                            if arg_types.len() != 1 {
                                self.error(*span, "len expects 1 argument".to_string());
                                Ty::Unknown
                            } else {
                                match &arg_types[0] {
                                    Ty::Array(_, _) | Ty::String | Ty::Unknown => Ty::Int,
                                    other => {
                                        self.error(
                                            *span,
                                            format!(
                                                "len expects an array or string, got {:?}",
                                                other
                                            ),
                                        );
                                        Ty::Unknown
                                    }
                                }
                            }
                        }
                        "sum" => {
                            if arg_types.len() != 1 {
                                self.error(*span, "sum expects 1 argument".to_string());
                                Ty::Unknown
                            } else {
                                match &arg_types[0] {
                                    Ty::Array(element, _) => {
                                        if matches!(element.as_ref(), Ty::Int | Ty::Float) {
                                            (*element.as_ref()).clone()
                                        } else {
                                            self.error(
                                                *span,
                                                format!(
                                                    "sum expects Int[] or Float[], got {:?}",
                                                    element
                                                ),
                                            );
                                            Ty::Unknown
                                        }
                                    }
                                    Ty::Unknown => Ty::Unknown,
                                    other => {
                                        self.error(
                                            *span,
                                            format!("sum expects an array, got {:?}", other),
                                        );
                                        Ty::Unknown
                                    }
                                }
                            }
                        }
                        "dot" => {
                            if arg_types.len() != 2 {
                                self.error(*span, "dot expects 2 arguments".to_string());
                                Ty::Unknown
                            } else {
                                match (&arg_types[0], &arg_types[1]) {
                                    (
                                        Ty::Array(left_elem, left_len),
                                        Ty::Array(right_elem, right_len),
                                    ) => {
                                        if left_len != right_len {
                                            self.error(
                                                *span,
                                                format!(
                                                    "dot expects same-length arrays, got {} and {}",
                                                    left_len, right_len
                                                ),
                                            );
                                            Ty::Unknown
                                        } else if !self.ty_compatible(left_elem, right_elem) {
                                            self.error(
                                                *span,
                                                format!(
                                                    "dot expects same element type arrays, got {:?} and {:?}",
                                                    left_elem, right_elem
                                                ),
                                            );
                                            Ty::Unknown
                                        } else if matches!(left_elem.as_ref(), Ty::Int | Ty::Float)
                                        {
                                            (*left_elem.as_ref()).clone()
                                        } else {
                                            self.error(
                                                *span,
                                                format!(
                                                    "dot expects Int[] or Float[] arrays, got {:?}",
                                                    left_elem
                                                ),
                                            );
                                            Ty::Unknown
                                        }
                                    }
                                    _ => {
                                        self.error(*span, "dot expects two arrays".to_string());
                                        Ty::Unknown
                                    }
                                }
                            }
                        }
                        "zeros" | "ones" => {
                            if arg_types.len() != 1 {
                                self.error(*span, format!("{} expects 1 argument", name));
                                Ty::Unknown
                            } else if arg_types[0] != Ty::Int && arg_types[0] != Ty::Unknown {
                                self.error(
                                    *span,
                                    format!("{} expects an Int length argument", name),
                                );
                                Ty::Unknown
                            } else {
                                match &args[0] {
                                    Expr::Int(len, _) if *len >= 0 => {
                                        Ty::Array(Box::new(Ty::Float), *len as usize)
                                    }
                                    _ => {
                                        self.error(
                                            *span,
                                            format!(
                                                "{} currently requires a compile-time non-negative Int literal length",
                                                name
                                            ),
                                        );
                                        Ty::Unknown
                                    }
                                }
                            }
                        }
                        "is_none" | "__is_none" => {
                            if arg_types.len() != 1 {
                                self.error(*span, "is_none expects 1 argument".to_string());
                                Ty::Unknown
                            } else {
                                match &arg_types[0] {
                                    Ty::Option(_) | Ty::Unknown => Ty::Bool,
                                    other => {
                                        self.error(
                                            *span,
                                            format!("is_none expects Option[T], got {:?}", other),
                                        );
                                        Ty::Unknown
                                    }
                                }
                            }
                        }
                        "unwrap" | "__unwrap" => {
                            if arg_types.len() != 1 {
                                self.error(*span, "unwrap expects 1 argument".to_string());
                                Ty::Unknown
                            } else {
                                match &arg_types[0] {
                                    Ty::Option(inner) => (*inner.as_ref()).clone(),
                                    Ty::Unknown => Ty::Unknown,
                                    other => {
                                        self.error(
                                            *span,
                                            format!("unwrap expects Option[T], got {:?}", other),
                                        );
                                        Ty::Unknown
                                    }
                                }
                            }
                        }
                        "some" | "__some" => {
                            if arg_types.len() != 1 {
                                self.error(*span, "some expects 1 argument".to_string());
                                Ty::Unknown
                            } else {
                                Ty::Option(Box::new(arg_types[0].clone()))
                            }
                        }
                        _ => {
                            let callee_ty = self.check_expr(callee);
                            self.check_call(callee_ty, &arg_types, *span)
                        }
                    }
                } else {
                    let callee_ty = self.check_expr(callee);
                    self.check_call(callee_ty, &arg_types, *span)
                }
            }
            Expr::Pipeline { span, .. } => {
                self.error(
                    *span,
                    "pipeline should be desugared before type checking".to_string(),
                );
                Ty::Unknown
            }
            Expr::EnumVariant {
                enum_name,
                variant_name,
                span,
            } => {
                if let Some(variants) = self.enums.get(enum_name) {
                    if !variants.contains(variant_name) {
                        self.error(
                            *span,
                            format!("enum '{}' has no variant '{}'", enum_name, variant_name),
                        );
                    }
                    Ty::Enum(enum_name.clone())
                } else {
                    self.error(*span, format!("unknown enum '{}'", enum_name));
                    Ty::Unknown
                }
            }
            Expr::Property { object, name, span } => {
                let object_ty = self.check_expr(object);
                self.check_property(object_ty, name, *span)
            }
            Expr::Index {
                target,
                index,
                span,
            } => {
                let target_ty = self.check_expr(target);
                let index_ty = self.check_expr(index);
                if index_ty != Ty::Int {
                    self.error(*span, "array index must be Int".to_string());
                }
                match target_ty {
                    Ty::Array(element, _) => *element,
                    _ => {
                        self.error(*span, "index target must be array".to_string());
                        Ty::Unknown
                    }
                }
            }
            Expr::GridIndex { span, .. } => {
                self.error(
                    *span,
                    "2D grid index should be desugared before type checking".to_string(),
                );
                Ty::Unknown
            }
            Expr::Assign {
                target,
                value,
                span,
            } => self.check_assign(target, value, *span),
            Expr::Try { span, .. } => {
                self.error(
                    *span,
                    "'?' should be desugared before type checking".to_string(),
                );
                Ty::Unknown
            }
            Expr::StringInterpolation { span, .. } => {
                self.error(
                    *span,
                    "string interpolation should be desugared before type checking".to_string(),
                );
                Ty::Unknown
            }
        };

        self.type_context.set_expr_ty(expr, ty.clone());
        ty
    }

    fn check_assign(&mut self, target: &AssignTarget, value: &Expr, span: Span) -> Ty {
        let value_ty = self.check_expr(value);
        match target {
            AssignTarget::Variable(name, target_span) => {
                if let Some(symbol) = self.lookup_symbol(name) {
                    if !symbol.mutable {
                        self.error(
                            *target_span,
                            format!("cannot assign to immutable '{}'", name),
                        );
                    }
                    if !self.ty_compatible(&symbol.ty, &value_ty) {
                        self.error(
                            span,
                            format!(
                                "assignment type mismatch for '{}': expected {:?}, got {:?}",
                                name, symbol.ty, value_ty
                            ),
                        );
                    }
                    symbol.ty.clone()
                } else {
                    self.error(*target_span, format!("undefined variable '{}'", name));
                    Ty::Unknown
                }
            }
            AssignTarget::Property { object, name, span } => {
                let object_ty = self.check_expr(object);
                match object_ty {
                    Ty::Instance(class_name) => {
                        if let Some(class) = self.classes.get(&class_name).cloned() {
                            if let Some(field_ty) = class.fields.get(name) {
                                if !self.ty_compatible(field_ty, &value_ty) {
                                    self.error(
                                        *span,
                                        format!(
                                            "field '{}' expects {:?}, got {:?}",
                                            name, field_ty, value_ty
                                        ),
                                    );
                                }
                                field_ty.clone()
                            } else {
                                self.error(
                                    *span,
                                    format!("class '{}' has no field '{}'", class_name, name),
                                );
                                Ty::Unknown
                            }
                        } else {
                            self.error(*span, format!("unknown class '{}'", class_name));
                            Ty::Unknown
                        }
                    }
                    _ => {
                        self.error(
                            *span,
                            "property assignment requires class instance".to_string(),
                        );
                        Ty::Unknown
                    }
                }
            }
            AssignTarget::Index {
                target,
                index,
                span,
            } => {
                let target_ty = self.check_expr(target);
                let index_ty = self.check_expr(index);
                if index_ty != Ty::Int {
                    self.error(*span, "array index must be Int".to_string());
                }
                match target_ty {
                    Ty::Array(element_ty, _) => {
                        if !self.ty_compatible(&element_ty, &value_ty) {
                            self.error(
                                *span,
                                format!(
                                    "array assignment mismatch: expected {:?}, got {:?}",
                                    element_ty, value_ty
                                ),
                            );
                        }
                        *element_ty
                    }
                    _ => {
                        self.error(*span, "index assignment requires array".to_string());
                        Ty::Unknown
                    }
                }
            }
            AssignTarget::GridIndex { span, .. } => {
                self.error(
                    *span,
                    "grid assignment should be desugared before type checking".to_string(),
                );
                Ty::Unknown
            }
        }
    }

    fn check_call(&mut self, callee_ty: Ty, arg_types: &[Ty], span: Span) -> Ty {
        match callee_ty {
            Ty::Function(params, ret) => {
                if params.len() != arg_types.len() && !self.any_unknown(&params) {
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
                                "argument {} type mismatch: expected {:?}, got {:?}",
                                index, expected, actual
                            ),
                        );
                    }
                }
                *ret
            }
            Ty::Class(name) => {
                if let Some(class) = self.classes.get(&name).cloned() {
                    if let Some(init) = &class.init {
                        if init.params.len() != arg_types.len() {
                            self.error(
                                span,
                                format!(
                                    "constructor '{}' expects {} args, got {}",
                                    name,
                                    init.params.len(),
                                    arg_types.len()
                                ),
                            );
                        }
                        for (index, (expected, actual)) in
                            init.params.iter().zip(arg_types.iter()).enumerate()
                        {
                            if !self.ty_compatible(expected, actual) {
                                self.error(
                                    span,
                                    format!(
                                        "constructor arg {} mismatch: expected {:?}, got {:?}",
                                        index, expected, actual
                                    ),
                                );
                            }
                        }
                    } else if !arg_types.is_empty() {
                        self.error(span, format!("constructor '{}' takes no arguments", name));
                    }
                }
                Ty::Instance(name)
            }
            Ty::Unknown => Ty::Unknown,
            other => {
                self.error(span, format!("value of type {:?} is not callable", other));
                Ty::Unknown
            }
        }
    }

    fn check_property(&mut self, object_ty: Ty, name: &str, span: Span) -> Ty {
        match object_ty {
            Ty::Instance(class_name) => {
                if let Some(class) = self.classes.get(&class_name) {
                    if let Some(field_ty) = class.fields.get(name) {
                        return field_ty.clone();
                    }
                    if let Some(method) = class.methods.get(name) {
                        return Ty::Function(method.params.clone(), Box::new(method.ret.clone()));
                    }
                    self.error(
                        span,
                        format!("class '{}' has no member '{}'", class_name, name),
                    );
                    Ty::Unknown
                } else {
                    self.error(span, format!("unknown class '{}'", class_name));
                    Ty::Unknown
                }
            }
            other => {
                self.error(
                    span,
                    format!("property access on non-instance type {:?}", other),
                );
                Ty::Unknown
            }
        }
    }

    fn check_binary(&mut self, op: BinaryOp, left: Ty, right: Ty, span: Span) -> Ty {
        match op {
            BinaryOp::And | BinaryOp::Or => {
                if left == Ty::Bool && right == Ty::Bool {
                    Ty::Bool
                } else {
                    self.error(
                        span,
                        format!(
                            "logical operator expects Bool/Bool operands, got {:?} and {:?}",
                            left, right
                        ),
                    );
                    Ty::Unknown
                }
            }
            BinaryOp::Add => {
                if left == Ty::String && right == Ty::String {
                    return Ty::String;
                }

                if let Some(array_ty) = self.check_vectorized_numeric(op, &left, &right, span) {
                    return array_ty;
                }

                if left == Ty::Int && right == Ty::Int {
                    Ty::Int
                } else if left == Ty::Float && right == Ty::Float {
                    Ty::Float
                } else {
                    self.error(
                        span,
                        format!(
                            "'+' expects Int/Int, Float/Float, String/String, or vectorized numeric operands; got {:?} and {:?}",
                            left, right
                        ),
                    );
                    Ty::Unknown
                }
            }
            BinaryOp::Subtract | BinaryOp::Multiply | BinaryOp::Divide => {
                if let Some(array_ty) = self.check_vectorized_numeric(op, &left, &right, span) {
                    return array_ty;
                }

                if left == Ty::Int && right == Ty::Int {
                    Ty::Int
                } else if left == Ty::Float && right == Ty::Float {
                    Ty::Float
                } else {
                    self.error(
                        span,
                        format!(
                            "numeric operator expects matching numeric types, got {:?} and {:?}",
                            left, right
                        ),
                    );
                    Ty::Unknown
                }
            }
            BinaryOp::Equal | BinaryOp::NotEqual => {
                if !self.ty_compatible(&left, &right) {
                    self.error(
                        span,
                        format!(
                            "equality operands must match, got {:?} and {:?}",
                            left, right
                        ),
                    );
                }
                Ty::Bool
            }
            BinaryOp::Greater | BinaryOp::GreaterEqual | BinaryOp::Less | BinaryOp::LessEqual => {
                if (left == Ty::Int && right == Ty::Int)
                    || (left == Ty::Float && right == Ty::Float)
                    || (left == Ty::String && right == Ty::String)
                {
                    Ty::Bool
                } else {
                    self.error(
                        span,
                        format!("comparison expects matching Int/Float/String operands, got {:?} and {:?}", left, right),
                    );
                    Ty::Unknown
                }
            }
        }
    }

    fn check_vectorized_numeric(
        &mut self,
        op: BinaryOp,
        left: &Ty,
        right: &Ty,
        span: Span,
    ) -> Option<Ty> {
        if !matches!(
            op,
            BinaryOp::Add | BinaryOp::Subtract | BinaryOp::Multiply | BinaryOp::Divide
        ) {
            return None;
        }

        match (left, right) {
            (Ty::Array(left_element, left_len), Ty::Array(right_element, right_len)) => {
                if left_len != right_len {
                    self.error(
                        span,
                        format!(
                            "vectorized binary op requires same-length arrays, got {} and {}",
                            left_len, right_len
                        ),
                    );
                    return Some(Ty::Unknown);
                }

                if !self.ty_compatible(left_element, right_element) {
                    self.error(
                        span,
                        format!(
                            "vectorized array op requires identical element types, got {:?} and {:?}",
                            left_element, right_element
                        ),
                    );
                    return Some(Ty::Unknown);
                }

                if !matches!(left_element.as_ref(), Ty::Int | Ty::Float | Ty::Unknown) {
                    self.error(
                        span,
                        format!(
                            "vectorized array op only supports numeric element types, got {:?}",
                            left_element
                        ),
                    );
                    return Some(Ty::Unknown);
                }

                Some(Ty::Array(left_element.clone(), *left_len))
            }
            (Ty::Array(element, len), scalar) => {
                if !matches!(element.as_ref(), Ty::Int | Ty::Float | Ty::Unknown) {
                    self.error(
                        span,
                        format!(
                            "vectorized array-scalar op only supports numeric element types, got {:?}",
                            element
                        ),
                    );
                    return Some(Ty::Unknown);
                }

                if !self.ty_compatible(element, scalar) {
                    self.error(
                        span,
                        format!(
                            "array-scalar promotion is strict; expected scalar {:?}, got {:?}",
                            element, scalar
                        ),
                    );
                    return Some(Ty::Unknown);
                }

                Some(Ty::Array(element.clone(), *len))
            }
            (scalar, Ty::Array(element, len)) => {
                if !matches!(element.as_ref(), Ty::Int | Ty::Float | Ty::Unknown) {
                    self.error(
                        span,
                        format!(
                            "vectorized scalar-array op only supports numeric element types, got {:?}",
                            element
                        ),
                    );
                    return Some(Ty::Unknown);
                }

                if !self.ty_compatible(element, scalar) {
                    self.error(
                        span,
                        format!(
                            "scalar-array promotion is strict; expected scalar {:?}, got {:?}",
                            element, scalar
                        ),
                    );
                    return Some(Ty::Unknown);
                }

                Some(Ty::Array(element.clone(), *len))
            }
            _ => None,
        }
    }

    fn resolve_type(&mut self, ty: &TypeExpr, span: Span) -> Ty {
        match ty {
            TypeExpr::Int => Ty::Int,
            TypeExpr::Float => Ty::Float,
            TypeExpr::String => Ty::String,
            TypeExpr::Bool => Ty::Bool,
            TypeExpr::Void => Ty::Void,
            TypeExpr::Applied { name, args } => {
                if name == "Option" {
                    if args.len() != 1 {
                        self.error(span, "Option expects exactly one type argument".to_string());
                        Ty::Unknown
                    } else {
                        Ty::Option(Box::new(self.resolve_type(&args[0], span)))
                    }
                } else if self.classes.contains_key(name) {
                    Ty::Instance(name.clone())
                } else {
                    Ty::Unknown
                }
            }
            TypeExpr::Option(inner) => Ty::Option(Box::new(self.resolve_type(inner, span))),
            TypeExpr::Array { element, len } => {
                Ty::Array(Box::new(self.resolve_type(element, span)), *len)
            }
            TypeExpr::Grid { .. } => {
                self.error(
                    span,
                    "grid type should be desugared before type checking".to_string(),
                );
                Ty::Unknown
            }
            TypeExpr::Named(name) => {
                if self.classes.contains_key(name) {
                    Ty::Instance(name.clone())
                } else if self.enums.contains_key(name) {
                    Ty::Enum(name.clone())
                } else if self
                    .generic_params
                    .iter()
                    .rev()
                    .any(|params| params.contains(name))
                {
                    Ty::Unknown
                } else {
                    self.error(span, format!("unknown type '{}'", name));
                    Ty::Unknown
                }
            }
        }
    }

    fn function_sig(&mut self, function: &FunctionDecl) -> Ty {
        let params_scope: HashSet<String> = function.type_params.iter().cloned().collect();
        self.generic_params.push(params_scope);
        let params = function
            .params
            .iter()
            .map(|param| self.resolve_type(&param.ty, param.span))
            .collect::<Vec<_>>();
        let ret = function
            .return_type
            .as_ref()
            .map(|ty| self.resolve_type(ty, function.span))
            .unwrap_or_else(|| self.infer_function_return(function));
        self.generic_params.pop();
        Ty::Function(params, Box::new(ret))
    }

    fn infer_function_return(&mut self, function: &FunctionDecl) -> Ty {
        let mut candidates = Vec::new();
        self.collect_return_types_from_block(&function.body, &mut candidates);
        if let Some(tail) = &function.body.tail {
            candidates.push(self.infer_expr_type_static(tail));
        }

        if candidates.is_empty() {
            return Ty::Void;
        }

        let mut inferred = candidates[0].clone();
        for ty in candidates.into_iter().skip(1) {
            if !self.ty_compatible(&inferred, &ty) {
                return Ty::Unknown;
            }
            if inferred == Ty::Unknown {
                inferred = ty;
            }
        }
        inferred
    }

    fn collect_return_types_from_block(&mut self, block: &BlockExpr, out: &mut Vec<Ty>) {
        for stmt in &block.statements {
            self.collect_return_types_from_stmt(stmt, out);
        }
    }

    fn collect_return_types_from_stmt(&mut self, stmt: &Stmt, out: &mut Vec<Ty>) {
        match stmt {
            Stmt::Return { value, .. } => {
                let ty = value
                    .as_ref()
                    .map(|expr| self.infer_expr_type_static(expr))
                    .unwrap_or(Ty::Void);
                out.push(ty);
            }
            Stmt::While { body, .. } => self.collect_return_types_from_block(body, out),
            Stmt::Expression {
                expr: Expr::Block(block),
                ..
            } => self.collect_return_types_from_block(block, out),
            Stmt::Expression {
                expr:
                    Expr::If {
                        then_branch,
                        else_branch,
                        ..
                    },
                ..
            } => {
                self.collect_return_types_from_block(then_branch, out);
                self.collect_return_types_from_block(else_branch, out);
            }
            _ => {}
        }
    }

    fn infer_expr_type_static(&mut self, expr: &Expr) -> Ty {
        match expr {
            Expr::Int(..) => Ty::Int,
            Expr::Float(..) => Ty::Float,
            Expr::Bool(..) => Ty::Bool,
            Expr::String(..) | Expr::StringInterpolation { .. } => Ty::String,
            Expr::EnumVariant { enum_name, .. } => Ty::Enum(enum_name.clone()),
            Expr::If {
                then_branch,
                else_branch,
                ..
            } => {
                let then_ty = then_branch
                    .tail
                    .as_ref()
                    .map(|expr| self.infer_expr_type_static(expr))
                    .unwrap_or(Ty::Void);
                let else_ty = else_branch
                    .tail
                    .as_ref()
                    .map(|expr| self.infer_expr_type_static(expr))
                    .unwrap_or(Ty::Void);
                if self.ty_compatible(&then_ty, &else_ty) {
                    then_ty
                } else {
                    Ty::Unknown
                }
            }
            Expr::Block(block) => block
                .tail
                .as_ref()
                .map(|expr| self.infer_expr_type_static(expr))
                .unwrap_or(Ty::Void),
            Expr::Call { callee, .. } => {
                if let Expr::Variable(name, _) = callee.as_ref() {
                    if let Some(Symbol {
                        ty: Ty::Function(_, ret),
                        ..
                    }) = self.lookup_symbol(name)
                    {
                        return (*ret).clone();
                    }
                }
                Ty::Unknown
            }
            _ => Ty::Unknown,
        }
    }

    fn define(&mut self, name: String, symbol: Symbol, span: Span) {
        if is_reserved_intrinsic(&name) {
            self.error(
                span,
                format!(
                    "'{}' is reserved for compiler/runtime intrinsics and cannot be redefined",
                    name
                ),
            );
            return;
        }

        let exists = self
            .scopes
            .last()
            .map(|scope| scope.contains_key(&name))
            .unwrap_or(false);
        if exists {
            self.error(span, format!("symbol '{}' already defined in scope", name));
        }
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(name, symbol);
        }
    }

    fn define_builtin(&mut self, name: String, symbol: Symbol) {
        if let Some(global) = self.scopes.first_mut() {
            global.insert(name, symbol);
        }
    }

    fn lookup_symbol(&self, name: &str) -> Option<Symbol> {
        for scope in self.scopes.iter().rev() {
            if let Some(symbol) = scope.get(name) {
                return Some(symbol.clone());
            }
        }
        None
    }

    fn lookup(&mut self, name: &str, span: Span) -> Option<Ty> {
        for scope in self.scopes.iter().rev() {
            if let Some(symbol) = scope.get(name) {
                return Some(symbol.ty.clone());
            }
        }
        self.error(span, format!("undefined symbol '{}'", name));
        None
    }

    fn ty_compatible(&self, expected: &Ty, actual: &Ty) -> bool {
        if *expected == Ty::Unknown || *actual == Ty::Unknown {
            return true;
        }

        match (expected, actual) {
            (Ty::Option(expected_inner), Ty::Option(actual_inner)) => {
                self.ty_compatible(expected_inner, actual_inner)
            }
            (Ty::Array(expected_inner, expected_len), Ty::Array(actual_inner, actual_len)) => {
                expected_len == actual_len && self.ty_compatible(expected_inner, actual_inner)
            }
            (
                Ty::Function(expected_params, expected_ret),
                Ty::Function(actual_params, actual_ret),
            ) => {
                expected_params.len() == actual_params.len()
                    && expected_params
                        .iter()
                        .zip(actual_params.iter())
                        .all(|(expected, actual)| self.ty_compatible(expected, actual))
                    && self.ty_compatible(expected_ret, actual_ret)
            }
            _ => expected == actual,
        }
    }

    fn any_unknown(&self, list: &[Ty]) -> bool {
        list.iter().any(|ty| *ty == Ty::Unknown)
    }

    fn enter_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    fn exit_scope(&mut self) {
        self.scopes.pop();
    }

    fn error(&mut self, span: Span, message: String) {
        self.errors
            .push(MuninnError::new("typecheck", message, span));
    }
}

fn is_reserved_intrinsic(name: &str) -> bool {
    matches!(
        name,
        "none" | "some" | "is_none" | "unwrap" | "__none" | "__some" | "__is_none" | "__unwrap"
    )
}
