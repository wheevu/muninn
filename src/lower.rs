use crate::ast::{
    AssignTarget, BinaryOp, BlockExpr, Expr, FunctionDecl, Program, Stmt, VecBinaryMode,
};
use crate::typecheck::{Ty, TypeContext};

pub fn lower_program(program: Program, type_context: &TypeContext) -> Program {
    let mut lowerer = Lowerer { type_context };
    lowerer.lower_program(program)
}

struct Lowerer<'a> {
    type_context: &'a TypeContext,
}

impl<'a> Lowerer<'a> {
    fn lower_program(&mut self, program: Program) -> Program {
        Program {
            statements: program
                .statements
                .into_iter()
                .map(|stmt| self.lower_stmt(stmt))
                .collect(),
        }
    }

    fn lower_stmt(&mut self, stmt: Stmt) -> Stmt {
        match stmt {
            Stmt::Let {
                name,
                mutable,
                ty,
                initializer,
                span,
            } => Stmt::Let {
                name,
                mutable,
                ty,
                initializer: self.lower_expr(initializer),
                span,
            },
            Stmt::Function(mut function) => {
                function.body = self.lower_block(function.body);
                Stmt::Function(function)
            }
            Stmt::Class(mut class) => {
                class.methods = class
                    .methods
                    .into_iter()
                    .map(|method| self.lower_function(method))
                    .collect();
                class.init = class.init.map(|init| self.lower_function(init));
                Stmt::Class(class)
            }
            Stmt::Enum(decl) => Stmt::Enum(decl),
            Stmt::Return { value, span } => Stmt::Return {
                value: value.map(|expr| self.lower_expr(expr)),
                span,
            },
            Stmt::Break { span } => Stmt::Break { span },
            Stmt::Continue { span } => Stmt::Continue { span },
            Stmt::While {
                condition,
                body,
                span,
            } => Stmt::While {
                condition: self.lower_expr(condition),
                body: self.lower_block(body),
                span,
            },
            Stmt::If {
                condition,
                then_branch,
                else_branch,
                span,
            } => Stmt::If {
                condition: self.lower_expr(condition),
                then_branch: self.lower_block(then_branch),
                else_branch: else_branch.map(|branch| self.lower_block(branch)),
                span,
            },
            Stmt::ForRange {
                var_name,
                start,
                end,
                body,
                span,
            } => Stmt::ForRange {
                var_name,
                start: self.lower_expr(start),
                end: self.lower_expr(end),
                body: self.lower_block(body),
                span,
            },
            Stmt::Expression { expr, span } => Stmt::Expression {
                expr: self.lower_expr(expr),
                span,
            },
        }
    }

    fn lower_function(&mut self, mut function: FunctionDecl) -> FunctionDecl {
        function.body = self.lower_block(function.body);
        function
    }

    fn lower_block(&mut self, block: BlockExpr) -> BlockExpr {
        BlockExpr {
            statements: block
                .statements
                .into_iter()
                .map(|stmt| self.lower_stmt(stmt))
                .collect(),
            tail: block.tail.map(|expr| Box::new(self.lower_expr(*expr))),
            span: block.span,
        }
    }

    fn lower_expr(&mut self, expr: Expr) -> Expr {
        match expr {
            Expr::Int(..)
            | Expr::Float(..)
            | Expr::Bool(..)
            | Expr::String(..)
            | Expr::Variable(..)
            | Expr::SelfRef(..)
            | Expr::ArrayLiteral(..) => self.lower_leaf_expr(expr),
            Expr::Block(block) => Expr::Block(self.lower_block(block)),
            Expr::Grouping(inner, span) => Expr::Grouping(Box::new(self.lower_expr(*inner)), span),
            Expr::Unary { op, expr, span } => Expr::Unary {
                op,
                expr: Box::new(self.lower_expr(*expr)),
                span,
            },
            Expr::Binary {
                left,
                op,
                right,
                span,
            } => {
                let mode = self.vector_mode(op, left.as_ref(), right.as_ref());
                let lowered_left = self.lower_expr(*left);
                let lowered_right = self.lower_expr(*right);
                if let Some((mode, len)) = mode {
                    Expr::VecBinary {
                        left: Box::new(lowered_left),
                        op,
                        right: Box::new(lowered_right),
                        len,
                        mode,
                        span,
                    }
                } else {
                    Expr::Binary {
                        left: Box::new(lowered_left),
                        op,
                        right: Box::new(lowered_right),
                        span,
                    }
                }
            }
            Expr::VecBinary {
                left,
                op,
                right,
                len,
                mode,
                span,
            } => Expr::VecBinary {
                left: Box::new(self.lower_expr(*left)),
                op,
                right: Box::new(self.lower_expr(*right)),
                len,
                mode,
                span,
            },
            Expr::If {
                condition,
                then_branch,
                else_branch,
                span,
            } => Expr::If {
                condition: Box::new(self.lower_expr(*condition)),
                then_branch: self.lower_block(then_branch),
                else_branch: self.lower_block(else_branch),
                span,
            },
            Expr::Unless {
                condition,
                then_branch,
                else_branch,
                span,
            } => Expr::Unless {
                condition: Box::new(self.lower_expr(*condition)),
                then_branch: self.lower_block(then_branch),
                else_branch: else_branch.map(|branch| self.lower_block(branch)),
                span,
            },
            Expr::Match {
                scrutinee,
                arms,
                span,
            } => Expr::Match {
                scrutinee: Box::new(self.lower_expr(*scrutinee)),
                arms: arms
                    .into_iter()
                    .map(|mut arm| {
                        arm.expr = self.lower_expr(arm.expr);
                        arm
                    })
                    .collect(),
                span,
            },
            Expr::Call { callee, args, span } => Expr::Call {
                callee: Box::new(self.lower_expr(*callee)),
                args: args.into_iter().map(|arg| self.lower_expr(arg)).collect(),
                span,
            },
            Expr::EnumVariant {
                enum_name,
                variant_name,
                span,
            } => Expr::EnumVariant {
                enum_name,
                variant_name,
                span,
            },
            Expr::Pipeline {
                lhs,
                callee,
                args,
                span,
            } => Expr::Pipeline {
                lhs: Box::new(self.lower_expr(*lhs)),
                callee: Box::new(self.lower_expr(*callee)),
                args: args.into_iter().map(|arg| self.lower_expr(arg)).collect(),
                span,
            },
            Expr::Property { object, name, span } => Expr::Property {
                object: Box::new(self.lower_expr(*object)),
                name,
                span,
            },
            Expr::Index {
                target,
                index,
                span,
            } => Expr::Index {
                target: Box::new(self.lower_expr(*target)),
                index: Box::new(self.lower_expr(*index)),
                span,
            },
            Expr::GridIndex { target, x, y, span } => Expr::GridIndex {
                target: Box::new(self.lower_expr(*target)),
                x: Box::new(self.lower_expr(*x)),
                y: Box::new(self.lower_expr(*y)),
                span,
            },
            Expr::Assign {
                target,
                value,
                span,
            } => Expr::Assign {
                target: self.lower_assign_target(target),
                value: Box::new(self.lower_expr(*value)),
                span,
            },
            Expr::Try { expr, span } => Expr::Try {
                expr: Box::new(self.lower_expr(*expr)),
                span,
            },
            Expr::StringInterpolation { parts, span } => Expr::StringInterpolation { parts, span },
        }
    }

    fn lower_leaf_expr(&mut self, expr: Expr) -> Expr {
        match expr {
            Expr::ArrayLiteral(items, span) => Expr::ArrayLiteral(
                items
                    .into_iter()
                    .map(|item| self.lower_expr(item))
                    .collect(),
                span,
            ),
            other => other,
        }
    }

    fn lower_assign_target(&mut self, target: AssignTarget) -> AssignTarget {
        match target {
            AssignTarget::Variable(..) => target,
            AssignTarget::Property { object, name, span } => AssignTarget::Property {
                object: Box::new(self.lower_expr(*object)),
                name,
                span,
            },
            AssignTarget::Index {
                target,
                index,
                span,
            } => AssignTarget::Index {
                target: Box::new(self.lower_expr(*target)),
                index: Box::new(self.lower_expr(*index)),
                span,
            },
            AssignTarget::GridIndex { target, x, y, span } => AssignTarget::GridIndex {
                target: Box::new(self.lower_expr(*target)),
                x: Box::new(self.lower_expr(*x)),
                y: Box::new(self.lower_expr(*y)),
                span,
            },
        }
    }

    fn vector_mode(
        &self,
        op: BinaryOp,
        left: &Expr,
        right: &Expr,
    ) -> Option<(VecBinaryMode, usize)> {
        if !matches!(
            op,
            BinaryOp::Add | BinaryOp::Subtract | BinaryOp::Multiply | BinaryOp::Divide
        ) {
            return None;
        }

        let left_ty = self.type_context.ty_for_expr(left)?;
        let right_ty = self.type_context.ty_for_expr(right)?;

        match (left_ty, right_ty) {
            (Ty::Array(left_element, left_len), Ty::Array(right_element, right_len)) => {
                if left_len == right_len
                    && left_element.as_ref() == right_element.as_ref()
                    && matches!(left_element.as_ref(), Ty::Int | Ty::Float)
                {
                    Some((VecBinaryMode::ArrayArray, *left_len))
                } else {
                    None
                }
            }
            (Ty::Array(element, len), scalar) => {
                if element.as_ref() == scalar && matches!(element.as_ref(), Ty::Int | Ty::Float) {
                    Some((VecBinaryMode::ArrayScalarRight, *len))
                } else {
                    None
                }
            }
            (scalar, Ty::Array(element, len)) => {
                if element.as_ref() == scalar && matches!(element.as_ref(), Ty::Int | Ty::Float) {
                    Some((VecBinaryMode::ScalarArrayLeft, *len))
                } else {
                    None
                }
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::ast::{Expr, Stmt};
    use crate::desugar::desugar_program;
    use crate::lexer::Lexer;
    use crate::lower::lower_program;
    use crate::parser::Parser;
    use crate::typecheck::check_program;

    #[test]
    fn lowers_array_binary_to_vec_binary() {
        let src = r#"
let a: Float[3] = [1.0, 2.0, 3.0];
let b: Float[3] = [4.0, 5.0, 6.0];
let c: Float[3] = a + b;
"#;

        let tokens = Lexer::new(src).lex().expect("tokens");
        let mut parser = Parser::new(tokens);
        let parsed = parser.parse_program().expect("program");
        let desugared = desugar_program(parsed).expect("desugar");
        let type_context = check_program(&desugared).expect("typecheck");
        let lowered = lower_program(desugared, &type_context);

        let Stmt::Let { initializer, .. } = &lowered.statements[2] else {
            panic!("expected third statement to be let");
        };

        assert!(matches!(initializer, Expr::VecBinary { .. }));
    }
}
