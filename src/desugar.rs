use std::collections::HashMap;

use crate::ast::{
    AssignTarget, BinaryOp, BlockExpr, Expr, InterpolationPart, Param, Program, Stmt, TypeExpr,
    UnaryOp,
};
use crate::error::MuninnError;
use crate::span::Span;

pub fn desugar_program(program: Program) -> Result<Program, MuninnError> {
    let mut desugarer = Desugarer::new();
    desugarer.desugar_program(program)
}

struct Desugarer {
    temp_counter: usize,
    grid_scopes: Vec<HashMap<String, usize>>,
    value_scopes: Vec<HashMap<String, TypeExpr>>,
    class_field_types: HashMap<String, HashMap<String, TypeExpr>>,
    class_grid_widths: HashMap<String, HashMap<String, usize>>,
    current_class: Vec<String>,
    function_returns: Vec<TypeExpr>,
}

impl Desugarer {
    fn new() -> Self {
        Self {
            temp_counter: 0,
            grid_scopes: vec![HashMap::new()],
            value_scopes: vec![HashMap::new()],
            class_field_types: HashMap::new(),
            class_grid_widths: HashMap::new(),
            current_class: Vec::new(),
            function_returns: Vec::new(),
        }
    }

    fn desugar_program(&mut self, program: Program) -> Result<Program, MuninnError> {
        let mut statements = Vec::with_capacity(program.statements.len());
        for stmt in program.statements {
            statements.push(self.desugar_stmt(stmt)?);
        }
        Ok(Program { statements })
    }

    fn desugar_stmt(&mut self, stmt: Stmt) -> Result<Stmt, MuninnError> {
        match stmt {
            Stmt::Let {
                name,
                mutable,
                ty,
                initializer,
                span,
            } => {
                let lowered_ty = ty.clone().map(|declared| self.desugar_type(declared));
                if let Some(TypeExpr::Grid { width, .. }) = ty {
                    self.grid_scopes
                        .last_mut()
                        .expect("scope")
                        .insert(name.clone(), width);
                }
                let initializer = self.desugar_expr(initializer)?;
                let inferred_ty = lowered_ty
                    .clone()
                    .or_else(|| self.infer_type_from_expr(&initializer));
                if let Some(value_ty) = inferred_ty {
                    self.value_scopes
                        .last_mut()
                        .expect("scope")
                        .insert(name.clone(), value_ty);
                }
                Ok(Stmt::Let {
                    name,
                    mutable,
                    ty: lowered_ty,
                    initializer,
                    span,
                })
            }
            Stmt::Function(mut function) => {
                self.enter_scope();
                for Param { name, ty, .. } in &mut function.params {
                    if let TypeExpr::Grid { width, .. } = ty {
                        self.grid_scopes
                            .last_mut()
                            .expect("scope")
                            .insert(name.clone(), *width);
                    }
                    *ty = self.desugar_type(ty.clone());
                    self.value_scopes
                        .last_mut()
                        .expect("scope")
                        .insert(name.clone(), ty.clone());
                }
                function.return_type = self.desugar_type(function.return_type);
                self.function_returns.push(function.return_type.clone());
                function.body = self.desugar_block(function.body)?;
                self.function_returns.pop();
                self.exit_scope();
                Ok(Stmt::Function(function))
            }
            Stmt::Class(mut class) => {
                let mut class_fields = HashMap::new();
                let mut class_grids = HashMap::new();
                for field in &class.fields {
                    if let TypeExpr::Grid { width, .. } = &field.ty {
                        class_grids.insert(field.name.clone(), *width);
                    }
                    class_fields.insert(field.name.clone(), self.desugar_type(field.ty.clone()));
                }
                self.class_field_types
                    .insert(class.name.clone(), class_fields);
                self.class_grid_widths
                    .insert(class.name.clone(), class_grids);
                self.current_class.push(class.name.clone());

                self.enter_scope();
                for field in &mut class.fields {
                    if let TypeExpr::Grid { width, .. } = &field.ty {
                        self.grid_scopes
                            .last_mut()
                            .expect("scope")
                            .insert(field.name.clone(), *width);
                    }
                    field.ty = self.desugar_type(field.ty.clone());
                }
                for method in &mut class.methods {
                    self.enter_scope();
                    self.value_scopes
                        .last_mut()
                        .expect("scope")
                        .insert("self".to_string(), TypeExpr::Named(class.name.clone()));
                    for param in &mut method.params {
                        if let TypeExpr::Grid { width, .. } = &param.ty {
                            self.grid_scopes
                                .last_mut()
                                .expect("scope")
                                .insert(param.name.clone(), *width);
                        }
                        param.ty = self.desugar_type(param.ty.clone());
                        self.value_scopes
                            .last_mut()
                            .expect("scope")
                            .insert(param.name.clone(), param.ty.clone());
                    }
                    method.return_type = self.desugar_type(method.return_type.clone());
                    self.function_returns.push(method.return_type.clone());
                    method.body = self.desugar_block(method.body.clone())?;
                    self.function_returns.pop();
                    self.exit_scope();
                }
                if let Some(init) = &mut class.init {
                    self.enter_scope();
                    self.value_scopes
                        .last_mut()
                        .expect("scope")
                        .insert("self".to_string(), TypeExpr::Named(class.name.clone()));
                    for param in &mut init.params {
                        if let TypeExpr::Grid { width, .. } = &param.ty {
                            self.grid_scopes
                                .last_mut()
                                .expect("scope")
                                .insert(param.name.clone(), *width);
                        }
                        param.ty = self.desugar_type(param.ty.clone());
                        self.value_scopes
                            .last_mut()
                            .expect("scope")
                            .insert(param.name.clone(), param.ty.clone());
                    }
                    init.return_type = self.desugar_type(init.return_type.clone());
                    self.function_returns.push(init.return_type.clone());
                    init.body = self.desugar_block(init.body.clone())?;
                    self.function_returns.pop();
                    self.exit_scope();
                }
                self.exit_scope();
                self.current_class.pop();
                Ok(Stmt::Class(class))
            }
            Stmt::Return { value, span } => {
                let value = value.map(|expr| self.desugar_expr(expr)).transpose()?;
                Ok(Stmt::Return { value, span })
            }
            Stmt::While {
                condition,
                body,
                span,
            } => Ok(Stmt::While {
                condition: self.desugar_expr(condition)?,
                body: self.desugar_block(body)?,
                span,
            }),
            Stmt::ForRange {
                var_name,
                start,
                end,
                body,
                span,
            } => self.desugar_for_range(var_name, start, end, body, span),
            Stmt::Expression { expr, span } => Ok(Stmt::Expression {
                expr: self.desugar_expr(expr)?,
                span,
            }),
        }
    }

    fn desugar_for_range(
        &mut self,
        var_name: String,
        start: Expr,
        end: Expr,
        body: BlockExpr,
        span: Span,
    ) -> Result<Stmt, MuninnError> {
        let range_id = self.next_temp_id();
        let start_name = format!("__range_{}_start", range_id);
        let end_name = format!("__range_{}_end", range_id);

        let start_expr = self.desugar_expr(start)?;
        let end_expr = self.desugar_expr(end)?;
        let mut body_block = self.desugar_block(body)?;

        let increment = Expr::Assign {
            target: AssignTarget::Variable(var_name.clone(), span),
            value: Box::new(Expr::Binary {
                left: Box::new(Expr::Variable(var_name.clone(), span)),
                op: BinaryOp::Add,
                right: Box::new(Expr::Int(1, span)),
                span,
            }),
            span,
        };
        body_block.statements.push(Stmt::Expression {
            expr: increment,
            span,
        });

        let while_stmt = Stmt::While {
            condition: Expr::Binary {
                left: Box::new(Expr::Variable(var_name.clone(), span)),
                op: BinaryOp::Less,
                right: Box::new(Expr::Variable(end_name.clone(), span)),
                span,
            },
            body: body_block,
            span,
        };

        let generated_block = BlockExpr {
            statements: vec![
                Stmt::Let {
                    name: start_name.clone(),
                    mutable: false,
                    ty: Some(TypeExpr::Int),
                    initializer: start_expr,
                    span,
                },
                Stmt::Let {
                    name: end_name,
                    mutable: false,
                    ty: Some(TypeExpr::Int),
                    initializer: end_expr,
                    span,
                },
                Stmt::Let {
                    name: var_name,
                    mutable: true,
                    ty: Some(TypeExpr::Int),
                    initializer: Expr::Variable(start_name, span),
                    span,
                },
                while_stmt,
            ],
            tail: None,
            span,
        };

        Ok(Stmt::Expression {
            expr: Expr::Block(generated_block),
            span,
        })
    }

    fn desugar_block(&mut self, block: BlockExpr) -> Result<BlockExpr, MuninnError> {
        self.enter_scope();
        let mut statements = Vec::with_capacity(block.statements.len());
        for stmt in block.statements {
            statements.push(self.desugar_stmt(stmt)?);
        }
        let tail = block
            .tail
            .map(|expr| self.desugar_expr(*expr))
            .transpose()?;
        self.exit_scope();

        Ok(BlockExpr {
            statements,
            tail: tail.map(Box::new),
            span: block.span,
        })
    }

    fn desugar_expr(&mut self, expr: Expr) -> Result<Expr, MuninnError> {
        match expr {
            Expr::Int(_, _)
            | Expr::Float(_, _)
            | Expr::Bool(_, _)
            | Expr::String(_, _)
            | Expr::Variable(_, _)
            | Expr::SelfRef(_)
            | Expr::ArrayLiteral(_, _) => self.desugar_leaf_expr(expr),
            Expr::Block(block) => Ok(Expr::Block(self.desugar_block(block)?)),
            Expr::Grouping(inner, span) => {
                Ok(Expr::Grouping(Box::new(self.desugar_expr(*inner)?), span))
            }
            Expr::Unary { op, expr, span } => Ok(Expr::Unary {
                op,
                expr: Box::new(self.desugar_expr(*expr)?),
                span,
            }),
            Expr::Binary {
                left,
                op,
                right,
                span,
            } => Ok(Expr::Binary {
                left: Box::new(self.desugar_expr(*left)?),
                op,
                right: Box::new(self.desugar_expr(*right)?),
                span,
            }),
            Expr::VecBinary {
                left,
                op,
                right,
                len,
                mode,
                span,
            } => Ok(Expr::VecBinary {
                left: Box::new(self.desugar_expr(*left)?),
                op,
                right: Box::new(self.desugar_expr(*right)?),
                len,
                mode,
                span,
            }),
            Expr::If {
                condition,
                then_branch,
                else_branch,
                span,
            } => Ok(Expr::If {
                condition: Box::new(self.desugar_expr(*condition)?),
                then_branch: self.desugar_block(then_branch)?,
                else_branch: self.desugar_block(else_branch)?,
                span,
            }),
            Expr::Unless {
                condition,
                then_branch,
                else_branch,
                span,
            } => {
                let condition = Expr::Unary {
                    op: UnaryOp::Not,
                    expr: Box::new(self.desugar_expr(*condition)?),
                    span,
                };
                let then_branch = self.desugar_block(then_branch)?;
                let else_branch = match else_branch {
                    Some(branch) => self.desugar_block(branch)?,
                    None => BlockExpr {
                        statements: Vec::new(),
                        tail: None,
                        span,
                    },
                };
                Ok(Expr::If {
                    condition: Box::new(condition),
                    then_branch,
                    else_branch,
                    span,
                })
            }
            Expr::Call { callee, args, span } => {
                let callee = self.desugar_expr(*callee)?;
                let args = args
                    .into_iter()
                    .map(|arg| self.desugar_expr(arg))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(Expr::Call {
                    callee: Box::new(callee),
                    args,
                    span,
                })
            }
            Expr::Pipeline {
                lhs,
                callee,
                args,
                span,
            } => {
                let mut lowered_args = Vec::with_capacity(args.len() + 1);
                lowered_args.push(self.desugar_expr(*lhs)?);
                lowered_args.extend(
                    args.into_iter()
                        .map(|arg| self.desugar_expr(arg))
                        .collect::<Result<Vec<_>, _>>()?,
                );
                Ok(Expr::Call {
                    callee: Box::new(self.desugar_expr(*callee)?),
                    args: lowered_args,
                    span,
                })
            }
            Expr::Property { object, name, span } => Ok(Expr::Property {
                object: Box::new(self.desugar_expr(*object)?),
                name,
                span,
            }),
            Expr::Index {
                target,
                index,
                span,
            } => Ok(Expr::Index {
                target: Box::new(self.desugar_expr(*target)?),
                index: Box::new(self.desugar_expr(*index)?),
                span,
            }),
            Expr::GridIndex { target, x, y, span } => {
                let lowered_target = self.desugar_expr(*target)?;
                let width = self.lookup_grid_width(&lowered_target).ok_or_else(|| {
                    MuninnError::new(
                        "desugar",
                        "cannot resolve grid width for 2D index expression",
                        span,
                    )
                })?;
                let lowered_x = self.desugar_expr(*x)?;
                let lowered_y = self.desugar_expr(*y)?;
                let x_span = lowered_x.span();
                let y_span = lowered_y.span();
                let lowered_index = Expr::Binary {
                    left: Box::new(Expr::Binary {
                        left: Box::new(lowered_y),
                        op: BinaryOp::Multiply,
                        right: Box::new(Expr::Int(width as i64, y_span)),
                        span: y_span,
                    }),
                    op: BinaryOp::Add,
                    right: Box::new(lowered_x),
                    span: x_span,
                };
                Ok(Expr::Index {
                    target: Box::new(lowered_target),
                    index: Box::new(lowered_index),
                    span,
                })
            }
            Expr::Assign {
                target,
                value,
                span,
            } => {
                let value = Box::new(self.desugar_expr(*value)?);
                let target = match target {
                    AssignTarget::Variable(name, target_span) => {
                        AssignTarget::Variable(name, target_span)
                    }
                    AssignTarget::Property { object, name, span } => AssignTarget::Property {
                        object: Box::new(self.desugar_expr(*object)?),
                        name,
                        span,
                    },
                    AssignTarget::Index {
                        target,
                        index,
                        span,
                    } => AssignTarget::Index {
                        target: Box::new(self.desugar_expr(*target)?),
                        index: Box::new(self.desugar_expr(*index)?),
                        span,
                    },
                    AssignTarget::GridIndex { target, x, y, span } => {
                        let lowered_target = self.desugar_expr(*target)?;
                        let width = self.lookup_grid_width(&lowered_target).ok_or_else(|| {
                            MuninnError::new(
                                "desugar",
                                "cannot resolve grid width for 2D assignment",
                                span,
                            )
                        })?;
                        let lowered_x = self.desugar_expr(*x)?;
                        let lowered_y = self.desugar_expr(*y)?;
                        let x_span = lowered_x.span();
                        let y_span = lowered_y.span();
                        let lowered_index = Expr::Binary {
                            left: Box::new(Expr::Binary {
                                left: Box::new(lowered_y),
                                op: BinaryOp::Multiply,
                                right: Box::new(Expr::Int(width as i64, y_span)),
                                span: y_span,
                            }),
                            op: BinaryOp::Add,
                            right: Box::new(lowered_x),
                            span: x_span,
                        };
                        AssignTarget::Index {
                            target: Box::new(lowered_target),
                            index: Box::new(lowered_index),
                            span,
                        }
                    }
                };
                Ok(Expr::Assign {
                    target,
                    value,
                    span,
                })
            }
            Expr::Try { expr, span } => self.desugar_try(*expr, span),
            Expr::StringInterpolation { parts, span } => self.desugar_interpolation(parts, span),
        }
    }

    fn desugar_try(&mut self, expr: Expr, span: Span) -> Result<Expr, MuninnError> {
        let Some(current_return) = self.function_returns.last().cloned() else {
            return Err(MuninnError::new(
                "desugar",
                "'?' can only be used inside a function returning Option[T]",
                span,
            ));
        };

        let TypeExpr::Option(inner_ty) = current_return else {
            return Err(MuninnError::new(
                "desugar",
                "'?' requires the enclosing function to return Option[T]",
                span,
            ));
        };

        let temp_name = format!("__try_{}", self.next_temp_id());
        let temp_var = Expr::Variable(temp_name.clone(), span);
        let lowered_expr = self.desugar_expr(expr)?;

        let is_none = Expr::Call {
            callee: Box::new(Expr::Variable("is_none".to_string(), span)),
            args: vec![temp_var.clone()],
            span,
        };

        let guard = Expr::If {
            condition: Box::new(is_none),
            then_branch: BlockExpr {
                statements: vec![Stmt::Return {
                    value: Some(Expr::Variable("none".to_string(), span)),
                    span,
                }],
                tail: None,
                span,
            },
            else_branch: BlockExpr {
                statements: Vec::new(),
                tail: None,
                span,
            },
            span,
        };

        let unwrapped = Expr::Call {
            callee: Box::new(Expr::Variable("unwrap".to_string(), span)),
            args: vec![temp_var],
            span,
        };

        Ok(Expr::Block(BlockExpr {
            statements: vec![
                Stmt::Let {
                    name: temp_name,
                    mutable: false,
                    ty: Some(TypeExpr::Option(inner_ty)),
                    initializer: lowered_expr,
                    span,
                },
                Stmt::Expression { expr: guard, span },
            ],
            tail: Some(Box::new(unwrapped)),
            span,
        }))
    }

    fn desugar_leaf_expr(&mut self, expr: Expr) -> Result<Expr, MuninnError> {
        match expr {
            Expr::ArrayLiteral(items, span) => Ok(Expr::ArrayLiteral(
                items
                    .into_iter()
                    .map(|item| self.desugar_expr(item))
                    .collect::<Result<Vec<_>, _>>()?,
                span,
            )),
            _ => Ok(expr),
        }
    }

    fn desugar_interpolation(
        &mut self,
        parts: Vec<InterpolationPart>,
        span: Span,
    ) -> Result<Expr, MuninnError> {
        let mut pieces = Vec::new();
        for part in parts {
            match part {
                InterpolationPart::Text(text) => pieces.push(Expr::String(text, span)),
                InterpolationPart::Expr(expr) => {
                    let call = Expr::Call {
                        callee: Box::new(Expr::Variable("to_string".to_string(), span)),
                        args: vec![self.desugar_expr(expr)?],
                        span,
                    };
                    pieces.push(call);
                }
            }
        }

        if pieces.is_empty() {
            return Ok(Expr::String(String::new(), span));
        }

        let mut iter = pieces.into_iter();
        let mut expr = iter.next().expect("first piece");
        for piece in iter {
            expr = Expr::Binary {
                left: Box::new(expr),
                op: BinaryOp::Add,
                right: Box::new(piece),
                span,
            };
        }

        Ok(expr)
    }

    fn desugar_type(&self, ty: TypeExpr) -> TypeExpr {
        match ty {
            TypeExpr::Array { element, len } => TypeExpr::Array {
                element: Box::new(self.desugar_type(*element)),
                len,
            },
            TypeExpr::Option(inner) => TypeExpr::Option(Box::new(self.desugar_type(*inner))),
            TypeExpr::Grid {
                element,
                width,
                height,
            } => TypeExpr::Array {
                element: Box::new(self.desugar_type(*element)),
                len: width * height,
            },
            other => other,
        }
    }

    fn lookup_grid_width(&self, target: &Expr) -> Option<usize> {
        match target {
            Expr::Variable(name, _) => {
                for scope in self.grid_scopes.iter().rev() {
                    if let Some(width) = scope.get(name) {
                        return Some(*width);
                    }
                }
                None
            }
            Expr::Property { object, name, .. } => {
                if matches!(object.as_ref(), Expr::SelfRef(_))
                    || matches!(object.as_ref(), Expr::Variable(self_name, _) if self_name == "self")
                {
                    for scope in self.grid_scopes.iter().rev() {
                        if let Some(width) = scope.get(name) {
                            return Some(*width);
                        }
                    }
                }

                let owner_class = self.resolve_expr_class(object)?;
                self.class_grid_widths
                    .get(&owner_class)
                    .and_then(|fields| fields.get(name))
                    .copied()
            }
            _ => None,
        }
    }

    fn resolve_expr_class(&self, expr: &Expr) -> Option<String> {
        match expr {
            Expr::SelfRef(_) => self
                .current_class
                .last()
                .cloned()
                .or_else(|| self.resolve_variable_class("self")),
            Expr::Variable(name, _) => self.resolve_variable_class(name),
            Expr::Property { object, name, .. } => {
                let owner_class = self.resolve_expr_class(object)?;
                let fields = self.class_field_types.get(&owner_class)?;
                match fields.get(name) {
                    Some(TypeExpr::Named(class_name)) => Some(class_name.clone()),
                    _ => None,
                }
            }
            _ => None,
        }
    }

    fn resolve_variable_class(&self, name: &str) -> Option<String> {
        match self.lookup_value_type(name) {
            Some(TypeExpr::Named(class_name)) => Some(class_name.clone()),
            _ => None,
        }
    }

    fn lookup_value_type(&self, name: &str) -> Option<&TypeExpr> {
        for scope in self.value_scopes.iter().rev() {
            if let Some(ty) = scope.get(name) {
                return Some(ty);
            }
        }
        None
    }

    fn infer_type_from_expr(&self, expr: &Expr) -> Option<TypeExpr> {
        match expr {
            Expr::Int(..) => Some(TypeExpr::Int),
            Expr::Float(..) => Some(TypeExpr::Float),
            Expr::Bool(..) => Some(TypeExpr::Bool),
            Expr::String(..) => Some(TypeExpr::String),
            Expr::Variable(name, _) => self.lookup_value_type(name).cloned(),
            Expr::Call { callee, .. } => {
                if let Expr::Variable(name, _) = callee.as_ref() {
                    if self.class_field_types.contains_key(name) {
                        return Some(TypeExpr::Named(name.clone()));
                    }
                }
                None
            }
            Expr::ArrayLiteral(items, _) => {
                let first = items.first()?;
                let element = self.infer_type_from_expr(first)?;
                if items
                    .iter()
                    .skip(1)
                    .all(|item| self.infer_type_from_expr(item).as_ref() == Some(&element))
                {
                    Some(TypeExpr::Array {
                        element: Box::new(element),
                        len: items.len(),
                    })
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    fn enter_scope(&mut self) {
        self.grid_scopes.push(HashMap::new());
        self.value_scopes.push(HashMap::new());
    }

    fn exit_scope(&mut self) {
        self.grid_scopes.pop();
        self.value_scopes.pop();
    }

    fn next_temp_id(&mut self) -> usize {
        let id = self.temp_counter;
        self.temp_counter += 1;
        id
    }
}

#[cfg(test)]
mod tests {
    use crate::ast::{AssignTarget, BinaryOp, Expr, Stmt};
    use crate::desugar::desugar_program;
    use crate::lexer::Lexer;
    use crate::parser::Parser;

    #[test]
    fn lowers_pipeline_to_call() {
        let src = "let x: Int = 1; let y: Int = x |> add(2);";
        let tokens = Lexer::new(src).lex().expect("tokens");
        let mut parser = Parser::new(tokens);
        let program = parser.parse_program().expect("program");
        let lowered = desugar_program(program).expect("desugar");

        let Stmt::Let { initializer, .. } = &lowered.statements[1] else {
            panic!("expected let statement")
        };
        let Expr::Call { args, .. } = initializer else {
            panic!("pipeline should lower to call")
        };
        assert_eq!(args.len(), 2);
    }

    #[test]
    fn lowers_grid_index_to_flat_offset() {
        let src = "let grid: Int[5, 5] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]; grid[2, 3] = 7;";
        let tokens = Lexer::new(src).lex().expect("tokens");
        let mut parser = Parser::new(tokens);
        let program = parser.parse_program().expect("program");
        let lowered = desugar_program(program).expect("desugar");

        let Stmt::Expression {
            expr: Expr::Assign { target, .. },
            ..
        } = &lowered.statements[1]
        else {
            panic!("expected assignment expression")
        };

        let AssignTarget::Index { index, .. } = target else {
            panic!("grid assignment should lower to linear index")
        };

        let Expr::Binary {
            left,
            op: BinaryOp::Add,
            right,
            ..
        } = index.as_ref()
        else {
            panic!("expected y*width + x offset")
        };

        let Expr::Binary {
            op: BinaryOp::Multiply,
            ..
        } = left.as_ref()
        else {
            panic!("expected y * width")
        };

        let Expr::Int(2, _) = right.as_ref() else {
            panic!("expected x coordinate to remain right operand")
        };
    }

    #[test]
    fn lowers_range_for_to_while() {
        let src = "for i in 0..3 { print(i); }";
        let tokens = Lexer::new(src).lex().expect("tokens");
        let mut parser = Parser::new(tokens);
        let program = parser.parse_program().expect("program");
        let lowered = desugar_program(program).expect("desugar");

        let Stmt::Expression {
            expr: Expr::Block(block),
            ..
        } = &lowered.statements[0]
        else {
            panic!("for loop should lower into block expression")
        };

        assert!(block
            .statements
            .iter()
            .any(|stmt| matches!(stmt, Stmt::While { .. })));
    }

    #[test]
    fn lowers_try_operator() {
        let src = r#"
fn checked(v: Float) -> Option[Float] {
    if (v > 0.0) { __some(v) } else { __none }
}

fn run(v: Float) -> Option[Float] {
    let x: Float = checked(v)?;
    __some(x)
}
"#;

        let tokens = Lexer::new(src).lex().expect("tokens");
        let mut parser = Parser::new(tokens);
        let program = parser.parse_program().expect("program");
        let lowered = desugar_program(program).expect("desugar");

        let Stmt::Function(function) = &lowered.statements[1] else {
            panic!("expected second statement to be function")
        };

        let Stmt::Let { initializer, .. } = &function.body.statements[0] else {
            panic!("expected first statement to be let")
        };

        assert!(matches!(initializer, Expr::Block(_)));
    }
}
