use crate::ast::{
    BinaryOp, Block, Expr, ExprKind, FunctionDecl, NodeId, Param, Program, Stmt, StmtKind,
    TypeExpr, UnaryOp,
};
use crate::error::MuninnError;
use crate::token::{Token, TokenKind};

pub struct Parser {
    tokens: Vec<Token>,
    current: usize,
    next_node_id: u32,
}

impl Parser {
    pub fn new(tokens: Vec<Token>) -> Self {
        Self {
            tokens,
            current: 0,
            next_node_id: 1,
        }
    }

    pub fn parse_program(&mut self) -> Result<Program, Vec<MuninnError>> {
        let mut statements = Vec::new();
        let mut errors = Vec::new();

        while !self.is_at_end() {
            match self.parse_top_level_statement() {
                Ok(stmt) => statements.push(stmt),
                Err(error) => {
                    errors.push(error);
                    self.synchronize();
                }
            }
        }

        if errors.is_empty() {
            Ok(Program { statements })
        } else {
            Err(errors)
        }
    }

    fn parse_top_level_statement(&mut self) -> Result<Stmt, MuninnError> {
        if self.match_simple(TokenKind::Fn) {
            return self.parse_function_statement();
        }
        self.parse_statement(false)
    }

    fn parse_statement(&mut self, inside_block: bool) -> Result<Stmt, MuninnError> {
        if self.match_simple(TokenKind::Let) {
            return self.parse_let_statement();
        }
        if self.match_simple(TokenKind::Return) {
            return self.parse_return_statement();
        }
        if self.match_simple(TokenKind::While) {
            return self.parse_while_statement();
        }
        if self.match_simple(TokenKind::If) {
            return self.parse_if_statement();
        }
        if self.match_simple(TokenKind::Fn) {
            return Err(MuninnError::new(
                "parser",
                "nested functions are not supported",
                self.previous().span,
            ));
        }
        if !inside_block && self.check_simple(TokenKind::Eof) {
            return Err(MuninnError::new(
                "parser",
                "unexpected end of input",
                self.peek().span,
            ));
        }
        if self.is_assignment_statement() {
            return self.parse_assignment_statement();
        }
        self.parse_expression_statement()
    }

    fn parse_function_statement(&mut self) -> Result<Stmt, MuninnError> {
        let start = self.previous().span;
        let (name, name_span) = self.consume_identifier_with_span("expected function name")?;
        self.consume_simple(TokenKind::LeftParen, "expected '(' after function name")?;

        let mut params = Vec::new();
        if !self.check_simple(TokenKind::RightParen) {
            loop {
                let param_span = self.peek().span;
                let param_name = self.consume_identifier("expected parameter name")?;
                self.consume_simple(TokenKind::Colon, "expected ':' after parameter name")?;
                let param_ty = self.parse_type_expr()?;
                params.push(Param {
                    id: self.alloc_id(),
                    name: param_name,
                    ty: param_ty,
                    span: param_span.merge(self.previous().span),
                });
                if !self.match_simple(TokenKind::Comma) {
                    break;
                }
            }
        }
        self.consume_simple(TokenKind::RightParen, "expected ')' after parameters")?;
        self.consume_simple(TokenKind::Arrow, "expected '->' before return type")?;
        let return_type = self.parse_type_expr()?;
        let body = self.parse_block()?;
        let span = start.merge(body.span);
        let function = FunctionDecl {
            id: self.alloc_id(),
            name,
            name_span,
            params,
            return_type,
            body,
            span,
        };
        Ok(Stmt {
            id: self.alloc_id(),
            kind: StmtKind::Function(function),
            span,
        })
    }

    fn parse_let_statement(&mut self) -> Result<Stmt, MuninnError> {
        let start = self.previous().span;
        let mutable = self.match_simple(TokenKind::Mut);
        let (name, name_span) = self.consume_identifier_with_span("expected variable name")?;
        let ty = if self.match_simple(TokenKind::Colon) {
            Some(self.parse_type_expr()?)
        } else {
            None
        };
        self.consume_simple(TokenKind::Equal, "expected '=' after variable name")?;
        let initializer = self.parse_expression()?;
        let end_span = self
            .consume_simple(TokenKind::Semicolon, "expected ';' after let binding")?
            .span;
        Ok(Stmt {
            id: self.alloc_id(),
            kind: StmtKind::Let {
                name,
                name_span,
                mutable,
                ty,
                initializer,
            },
            span: start.merge(end_span),
        })
    }

    fn parse_return_statement(&mut self) -> Result<Stmt, MuninnError> {
        let start = self.previous().span;
        let value = if self.check_simple(TokenKind::Semicolon) {
            None
        } else {
            Some(self.parse_expression()?)
        };
        let end_span = self
            .consume_simple(TokenKind::Semicolon, "expected ';' after return")?
            .span;
        Ok(Stmt {
            id: self.alloc_id(),
            kind: StmtKind::Return(value),
            span: start.merge(end_span),
        })
    }

    fn parse_while_statement(&mut self) -> Result<Stmt, MuninnError> {
        let start = self.previous().span;
        self.consume_simple(TokenKind::LeftParen, "expected '(' after 'while'")?;
        let condition = self.parse_expression()?;
        self.consume_simple(TokenKind::RightParen, "expected ')' after while condition")?;
        let body = self.parse_block()?;
        let body_span = body.span;
        Ok(Stmt {
            id: self.alloc_id(),
            kind: StmtKind::While { condition, body },
            span: start.merge(body_span),
        })
    }

    fn parse_if_statement(&mut self) -> Result<Stmt, MuninnError> {
        let start = self.previous().span;
        self.consume_simple(TokenKind::LeftParen, "expected '(' after 'if'")?;
        let condition = self.parse_expression()?;
        self.consume_simple(TokenKind::RightParen, "expected ')' after if condition")?;
        let then_branch = self.parse_block()?;
        let else_branch = if self.match_simple(TokenKind::Else) {
            Some(self.parse_block()?)
        } else {
            None
        };
        let span = start.merge(else_branch.as_ref().map(|block| block.span).unwrap_or(then_branch.span));
        Ok(Stmt {
            id: self.alloc_id(),
            kind: StmtKind::If {
                condition,
                then_branch,
                else_branch,
            },
            span,
        })
    }

    fn parse_assignment_statement(&mut self) -> Result<Stmt, MuninnError> {
        let start = self.peek().span;
        let (name, name_span) = self.consume_identifier_with_span("expected assignment target")?;
        self.consume_simple(TokenKind::Equal, "expected '=' in assignment")?;
        let value = self.parse_expression()?;
        let end_span = self
            .consume_simple(TokenKind::Semicolon, "expected ';' after assignment")?
            .span;
        Ok(Stmt {
            id: self.alloc_id(),
            kind: StmtKind::Assign {
                name,
                name_span,
                value,
            },
            span: start.merge(end_span),
        })
    }

    fn parse_expression_statement(&mut self) -> Result<Stmt, MuninnError> {
        let expr = self.parse_expression()?;
        let end_span = self
            .consume_simple(TokenKind::Semicolon, "expected ';' after expression")?
            .span;
        let span = expr.span.merge(end_span);
        Ok(Stmt {
            id: self.alloc_id(),
            kind: StmtKind::Expr(expr),
            span,
        })
    }

    fn parse_block(&mut self) -> Result<Block, MuninnError> {
        let start = self.consume_simple(TokenKind::LeftBrace, "expected '{'")?.span;
        let mut statements = Vec::new();
        while !self.check_simple(TokenKind::RightBrace) && !self.is_at_end() {
            statements.push(self.parse_statement(true)?);
        }
        let end_span = self
            .consume_simple(TokenKind::RightBrace, "expected '}' after block")?
            .span;
        Ok(Block {
            id: self.alloc_id(),
            statements,
            span: start.merge(end_span),
        })
    }

    fn parse_expression(&mut self) -> Result<Expr, MuninnError> {
        self.parse_or()
    }

    fn parse_or(&mut self) -> Result<Expr, MuninnError> {
        let mut expr = self.parse_and()?;
        while self.match_simple(TokenKind::OrOr) {
            let op_span = self.previous().span;
            let right = self.parse_and()?;
            let span = expr.span.merge(right.span);
            expr = Expr {
                id: self.alloc_id(),
                kind: ExprKind::Binary {
                    left: Box::new(expr),
                    op: BinaryOp::Or,
                    right: Box::new(right),
                },
                span: op_span.merge(span),
            };
        }
        Ok(expr)
    }

    fn parse_and(&mut self) -> Result<Expr, MuninnError> {
        let mut expr = self.parse_equality()?;
        while self.match_simple(TokenKind::AndAnd) {
            let op_span = self.previous().span;
            let right = self.parse_equality()?;
            let span = expr.span.merge(right.span);
            expr = Expr {
                id: self.alloc_id(),
                kind: ExprKind::Binary {
                    left: Box::new(expr),
                    op: BinaryOp::And,
                    right: Box::new(right),
                },
                span: op_span.merge(span),
            };
        }
        Ok(expr)
    }

    fn parse_equality(&mut self) -> Result<Expr, MuninnError> {
        let mut expr = self.parse_comparison()?;
        while self.match_any(&[TokenKind::EqualEqual, TokenKind::BangEqual]) {
            let op_token = self.previous().clone();
            let right = self.parse_comparison()?;
            let op = match op_token.kind {
                TokenKind::EqualEqual => BinaryOp::Equal,
                TokenKind::BangEqual => BinaryOp::NotEqual,
                _ => unreachable!(),
            };
            let span = expr.span.merge(right.span);
            expr = Expr {
                id: self.alloc_id(),
                kind: ExprKind::Binary {
                    left: Box::new(expr),
                    op,
                    right: Box::new(right),
                },
                span,
            };
        }
        Ok(expr)
    }

    fn parse_comparison(&mut self) -> Result<Expr, MuninnError> {
        let mut expr = self.parse_term()?;
        while self.match_any(&[
            TokenKind::Greater,
            TokenKind::GreaterEqual,
            TokenKind::Less,
            TokenKind::LessEqual,
        ]) {
            let op_token = self.previous().clone();
            let right = self.parse_term()?;
            let op = match op_token.kind {
                TokenKind::Greater => BinaryOp::Greater,
                TokenKind::GreaterEqual => BinaryOp::GreaterEqual,
                TokenKind::Less => BinaryOp::Less,
                TokenKind::LessEqual => BinaryOp::LessEqual,
                _ => unreachable!(),
            };
            let span = expr.span.merge(right.span);
            expr = Expr {
                id: self.alloc_id(),
                kind: ExprKind::Binary {
                    left: Box::new(expr),
                    op,
                    right: Box::new(right),
                },
                span,
            };
        }
        Ok(expr)
    }

    fn parse_term(&mut self) -> Result<Expr, MuninnError> {
        let mut expr = self.parse_factor()?;
        while self.match_any(&[TokenKind::Plus, TokenKind::Minus]) {
            let op_token = self.previous().clone();
            let right = self.parse_factor()?;
            let op = match op_token.kind {
                TokenKind::Plus => BinaryOp::Add,
                TokenKind::Minus => BinaryOp::Subtract,
                _ => unreachable!(),
            };
            let span = expr.span.merge(right.span);
            expr = Expr {
                id: self.alloc_id(),
                kind: ExprKind::Binary {
                    left: Box::new(expr),
                    op,
                    right: Box::new(right),
                },
                span,
            };
        }
        Ok(expr)
    }

    fn parse_factor(&mut self) -> Result<Expr, MuninnError> {
        let mut expr = self.parse_unary()?;
        while self.match_any(&[TokenKind::Star, TokenKind::Slash]) {
            let op_token = self.previous().clone();
            let right = self.parse_unary()?;
            let op = match op_token.kind {
                TokenKind::Star => BinaryOp::Multiply,
                TokenKind::Slash => BinaryOp::Divide,
                _ => unreachable!(),
            };
            let span = expr.span.merge(right.span);
            expr = Expr {
                id: self.alloc_id(),
                kind: ExprKind::Binary {
                    left: Box::new(expr),
                    op,
                    right: Box::new(right),
                },
                span,
            };
        }
        Ok(expr)
    }

    fn parse_unary(&mut self) -> Result<Expr, MuninnError> {
        if self.match_any(&[TokenKind::Bang, TokenKind::Minus]) {
            let op_token = self.previous().clone();
            let expr = self.parse_unary()?;
            let op = match op_token.kind {
                TokenKind::Bang => UnaryOp::Not,
                TokenKind::Minus => UnaryOp::Negate,
                _ => unreachable!(),
            };
            let span = op_token.span.merge(expr.span);
            return Ok(Expr {
                id: self.alloc_id(),
                kind: ExprKind::Unary {
                    op,
                    expr: Box::new(expr),
                },
                span,
            });
        }
        self.parse_call()
    }

    fn parse_call(&mut self) -> Result<Expr, MuninnError> {
        let mut expr = self.parse_primary()?;
        loop {
            if self.match_simple(TokenKind::LeftParen) {
                let mut args = Vec::new();
                if !self.check_simple(TokenKind::RightParen) {
                    loop {
                        args.push(self.parse_expression()?);
                        if !self.match_simple(TokenKind::Comma) {
                            break;
                        }
                    }
                }
                let end = self.consume_simple(TokenKind::RightParen, "expected ')' after arguments")?;
                let span = expr.span.merge(end.span);
                expr = Expr {
                    id: self.alloc_id(),
                    kind: ExprKind::Call {
                        callee: Box::new(expr),
                        args,
                    },
                    span,
                };
                continue;
            }
            break;
        }
        Ok(expr)
    }

    fn parse_primary(&mut self) -> Result<Expr, MuninnError> {
        let token = self.advance().clone();
        let kind = match token.kind {
            TokenKind::IntLiteral(value) => ExprKind::Int(value),
            TokenKind::FloatLiteral(value) => ExprKind::Float(value),
            TokenKind::True => ExprKind::Bool(true),
            TokenKind::False => ExprKind::Bool(false),
            TokenKind::StringLiteral(value) => ExprKind::String(value),
            TokenKind::Identifier(name) => ExprKind::Variable(name),
            TokenKind::LeftParen => {
                let expr = self.parse_expression()?;
                let end_span = self
                    .consume_simple(TokenKind::RightParen, "expected ')' after expression")?
                    .span;
                return Ok(Expr {
                    id: self.alloc_id(),
                    kind: ExprKind::Grouping(Box::new(expr)),
                    span: token.span.merge(end_span),
                });
            }
            _ => {
                return Err(MuninnError::new(
                    "parser",
                    "expected expression",
                    token.span,
                ));
            }
        };

        Ok(Expr {
            id: self.alloc_id(),
            kind,
            span: token.span,
        })
    }

    fn parse_type_expr(&mut self) -> Result<TypeExpr, MuninnError> {
        let token = self.advance().clone();
        match token.kind {
            TokenKind::TypeInt => Ok(TypeExpr::Int),
            TokenKind::TypeFloat => Ok(TypeExpr::Float),
            TokenKind::TypeBool => Ok(TypeExpr::Bool),
            TokenKind::TypeString => Ok(TypeExpr::String),
            TokenKind::TypeVoid => Ok(TypeExpr::Void),
            _ => Err(MuninnError::new(
                "parser",
                "expected type name",
                token.span,
            )),
        }
    }

    fn is_assignment_statement(&self) -> bool {
        matches!(self.peek().kind, TokenKind::Identifier(_))
            && self
                .tokens
                .get(self.current + 1)
                .is_some_and(|token| matches!(token.kind, TokenKind::Equal))
    }

    fn synchronize(&mut self) {
        while !self.is_at_end() {
            if matches!(self.previous().kind, TokenKind::Semicolon) {
                return;
            }
            match self.peek().kind {
                TokenKind::Fn
                | TokenKind::Let
                | TokenKind::Return
                | TokenKind::If
                | TokenKind::While => return,
                _ => {
                    self.current += 1;
                }
            }
        }
    }

    fn consume_identifier(&mut self, message: &'static str) -> Result<String, MuninnError> {
        self.consume_identifier_with_span(message).map(|(name, _)| name)
    }

    fn consume_identifier_with_span(
        &mut self,
        message: &'static str,
    ) -> Result<(String, crate::span::Span), MuninnError> {
        let token = self.advance().clone();
        if let TokenKind::Identifier(name) = token.kind {
            Ok((name, token.span))
        } else {
            Err(MuninnError::new("parser", message, token.span))
        }
    }

    fn consume_simple(
        &mut self,
        expected: TokenKind,
        message: &'static str,
    ) -> Result<&Token, MuninnError> {
        if self.check_simple(expected.clone()) {
            Ok(self.advance())
        } else {
            Err(MuninnError::new("parser", message, self.peek().span))
        }
    }

    fn match_simple(&mut self, expected: TokenKind) -> bool {
        if self.check_simple(expected) {
            self.current += 1;
            true
        } else {
            false
        }
    }

    fn match_any(&mut self, expected: &[TokenKind]) -> bool {
        for kind in expected {
            if self.check_simple(kind.clone()) {
                self.current += 1;
                return true;
            }
        }
        false
    }

    fn check_simple(&self, expected: TokenKind) -> bool {
        same_variant(&self.peek().kind, &expected)
    }

    fn advance(&mut self) -> &Token {
        if !self.is_at_end() {
            self.current += 1;
        }
        self.previous()
    }

    fn is_at_end(&self) -> bool {
        matches!(self.peek().kind, TokenKind::Eof)
    }

    fn peek(&self) -> &Token {
        &self.tokens[self.current]
    }

    fn previous(&self) -> &Token {
        &self.tokens[self.current.saturating_sub(1)]
    }

    fn alloc_id(&mut self) -> NodeId {
        let id = self.next_node_id;
        self.next_node_id += 1;
        NodeId(id)
    }
}

fn same_variant(left: &TokenKind, right: &TokenKind) -> bool {
    std::mem::discriminant(left) == std::mem::discriminant(right)
}

#[cfg(test)]
mod tests {
    use crate::ast::{ExprKind, StmtKind};
    use crate::lexer::Lexer;

    use super::Parser;

    #[test]
    fn parses_function_and_loop() {
        let src = r#"
fn add(a: Int, b: Int) -> Int {
    return a + b;
}

let mut total: Int = 0;
while (total < 3) {
    total = add(total, 1);
}
"#;
        let tokens = Lexer::new(src).lex().expect("tokens");
        let mut parser = Parser::new(tokens);
        let program = parser.parse_program().expect("program");
        assert_eq!(program.statements.len(), 3);
        assert!(matches!(program.statements[0].kind, StmtKind::Function(_)));
    }

    #[test]
    fn parses_call_expression_statement() {
        let src = "print(1 + 2);";
        let tokens = Lexer::new(src).lex().expect("tokens");
        let mut parser = Parser::new(tokens);
        let program = parser.parse_program().expect("program");
        let StmtKind::Expr(expr) = &program.statements[0].kind else {
            panic!("expected expr stmt");
        };
        assert!(matches!(expr.kind, ExprKind::Call { .. }));
    }

    #[test]
    fn recovers_multiple_errors() {
        let src = "let x: Int = ; let y: String = ;";
        let tokens = Lexer::new(src).lex().expect("tokens");
        let mut parser = Parser::new(tokens);
        let errors = parser.parse_program().expect_err("errors");
        assert!(errors.len() >= 2);
    }
}
