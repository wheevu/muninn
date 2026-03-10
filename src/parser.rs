use crate::ast::{
    AssignTarget, BinaryOp, BlockExpr, ClassDecl, Expr, FieldDecl, FunctionDecl, InterpolationPart,
    Param, Program, Stmt, TypeExpr, UnaryOp,
};
use crate::error::MuninnError;
use crate::lexer::Lexer;
use crate::span::Span;
use crate::token::{Token, TokenKind};

pub struct Parser {
    tokens: Vec<Token>,
    current: usize,
}

impl Parser {
    pub fn new(tokens: Vec<Token>) -> Self {
        Self { tokens, current: 0 }
    }

    pub fn parse_program(&mut self) -> Result<Program, Vec<MuninnError>> {
        let mut statements = Vec::new();
        let mut errors = Vec::new();
        while !self.is_at_end() {
            match self.parse_declaration() {
                Ok(stmt) => statements.push(stmt),
                Err(err) => {
                    errors.push(err);
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

    fn parse_declaration(&mut self) -> Result<Stmt, MuninnError> {
        if self.match_where(|k| matches!(k, TokenKind::Let)) {
            return self.parse_let();
        }
        if self.match_where(|k| matches!(k, TokenKind::Fn)) {
            return Ok(Stmt::Function(self.parse_function_decl(false)?));
        }
        if self.match_where(|k| matches!(k, TokenKind::Class)) {
            return Ok(Stmt::Class(self.parse_class_decl()?));
        }
        self.parse_statement()
    }

    fn parse_let(&mut self) -> Result<Stmt, MuninnError> {
        let mutable = self.match_where(|k| matches!(k, TokenKind::Mut));
        let name = self.consume_identifier("expected variable name after let")?;
        let span = self.previous().span;
        let ty = if self.match_where(|k| matches!(k, TokenKind::Colon)) {
            Some(self.parse_type_expr()?)
        } else {
            None
        };
        self.consume_where(
            |k| matches!(k, TokenKind::Equal),
            "expected '=' after variable declaration",
        )?;
        let initializer = self.parse_expression()?;
        self.consume_where(
            |k| matches!(k, TokenKind::Semicolon),
            "expected ';' after variable declaration",
        )?;

        Ok(Stmt::Let {
            name,
            mutable,
            ty,
            initializer,
            span,
        })
    }

    fn parse_function_decl(&mut self, is_method: bool) -> Result<FunctionDecl, MuninnError> {
        let name = if is_method {
            if self.match_where(|k| matches!(k, TokenKind::Init)) {
                "init".to_string()
            } else {
                self.consume_identifier("expected method name")?
            }
        } else {
            self.consume_identifier("expected function name")?
        };

        let span = self.previous().span;
        self.consume_where(
            |k| matches!(k, TokenKind::LeftParen),
            "expected '(' after function name",
        )?;

        let mut params = Vec::new();
        if !self.check_where(|k| matches!(k, TokenKind::RightParen)) {
            loop {
                let param_name = self.consume_identifier("expected parameter name")?;
                let param_span = self.previous().span;
                self.consume_where(
                    |k| matches!(k, TokenKind::Colon),
                    "expected ':' after parameter name",
                )?;
                let param_ty = self.parse_type_expr()?;
                params.push(Param {
                    name: param_name,
                    ty: param_ty,
                    span: param_span,
                });

                if !self.match_where(|k| matches!(k, TokenKind::Comma)) {
                    break;
                }
            }
        }

        self.consume_where(
            |k| matches!(k, TokenKind::RightParen),
            "expected ')' after parameters",
        )?;

        let return_type = if self.match_where(|k| matches!(k, TokenKind::Arrow)) {
            self.parse_type_expr()?
        } else if name == "init" {
            TypeExpr::Void
        } else {
            TypeExpr::Void
        };

        let body = self.parse_block_expr()?;

        Ok(FunctionDecl {
            name,
            params,
            return_type,
            body,
            span,
        })
    }

    fn parse_class_decl(&mut self) -> Result<ClassDecl, MuninnError> {
        let name = self.consume_identifier("expected class name")?;
        let span = self.previous().span;
        self.consume_where(
            |k| matches!(k, TokenKind::LeftBrace),
            "expected '{' after class name",
        )?;

        let mut fields = Vec::new();
        let mut methods = Vec::new();
        let mut init = None;

        while !self.check_where(|k| matches!(k, TokenKind::RightBrace)) && !self.is_at_end() {
            if self.match_where(|k| matches!(k, TokenKind::Let)) {
                let field_name = self.consume_identifier("expected field name")?;
                let field_span = self.previous().span;
                self.consume_where(
                    |k| matches!(k, TokenKind::Colon),
                    "expected ':' after field name",
                )?;
                let field_type = self.parse_type_expr()?;
                self.consume_where(
                    |k| matches!(k, TokenKind::Semicolon),
                    "expected ';' after field declaration",
                )?;
                fields.push(FieldDecl {
                    name: field_name,
                    ty: field_type,
                    span: field_span,
                });
                continue;
            }

            if self.match_where(|k| matches!(k, TokenKind::Fn)) {
                let method = self.parse_function_decl(true)?;
                if method.name == "init" {
                    init = Some(method);
                } else {
                    methods.push(method);
                }
                continue;
            }

            return Err(MuninnError::new(
                "parser",
                "expected class member declaration",
                self.peek().span,
            ));
        }

        self.consume_where(
            |k| matches!(k, TokenKind::RightBrace),
            "expected '}' after class body",
        )?;

        Ok(ClassDecl {
            name,
            fields,
            methods,
            init,
            span,
        })
    }

    fn parse_statement(&mut self) -> Result<Stmt, MuninnError> {
        if self.match_where(|k| matches!(k, TokenKind::Return)) {
            let span = self.previous().span;
            let value = if self.check_where(|k| matches!(k, TokenKind::Semicolon)) {
                None
            } else {
                Some(self.parse_expression()?)
            };
            self.consume_where(
                |k| matches!(k, TokenKind::Semicolon),
                "expected ';' after return value",
            )?;
            return Ok(Stmt::Return { value, span });
        }

        if self.match_where(|k| matches!(k, TokenKind::While)) {
            let span = self.previous().span;
            self.consume_where(
                |k| matches!(k, TokenKind::LeftParen),
                "expected '(' after while",
            )?;
            let condition = self.parse_expression()?;
            self.consume_where(
                |k| matches!(k, TokenKind::RightParen),
                "expected ')' after while condition",
            )?;
            let body = self.parse_block_expr()?;
            return Ok(Stmt::While {
                condition,
                body,
                span,
            });
        }

        if self.match_where(|k| matches!(k, TokenKind::For)) {
            return self.parse_for_range_stmt();
        }

        let expr = self.parse_expression()?;
        let span = expr.span();
        self.consume_where(
            |k| matches!(k, TokenKind::Semicolon),
            "expected ';' after expression statement",
        )?;
        Ok(Stmt::Expression { expr, span })
    }

    fn parse_for_range_stmt(&mut self) -> Result<Stmt, MuninnError> {
        let span = self.previous().span;
        let var_name = self.consume_identifier("expected loop variable after 'for'")?;
        self.consume_where(
            |k| matches!(k, TokenKind::In),
            "expected 'in' after loop variable",
        )?;
        let start = self.parse_expression()?;
        self.consume_where(
            |k| matches!(k, TokenKind::DotDot),
            "expected '..' in for-range loop",
        )?;
        let end = self.parse_expression()?;
        let body = self.parse_block_expr()?;
        Ok(Stmt::ForRange {
            var_name,
            start,
            end,
            body,
            span,
        })
    }

    fn parse_expression(&mut self) -> Result<Expr, MuninnError> {
        self.parse_assignment()
    }

    fn parse_assignment(&mut self) -> Result<Expr, MuninnError> {
        let expr = self.parse_pipeline()?;
        if self.match_where(|k| matches!(k, TokenKind::Equal)) {
            let value = self.parse_assignment()?;
            let span = expr.span();
            let target = match expr {
                Expr::Variable(name, target_span) => AssignTarget::Variable(name, target_span),
                Expr::Property { object, name, span } => {
                    AssignTarget::Property { object, name, span }
                }
                Expr::Index {
                    target,
                    index,
                    span,
                } => AssignTarget::Index {
                    target,
                    index,
                    span,
                },
                Expr::GridIndex { target, x, y, span } => {
                    AssignTarget::GridIndex { target, x, y, span }
                }
                _ => {
                    return Err(MuninnError::new(
                        "parser",
                        "invalid assignment target",
                        span,
                    ));
                }
            };
            return Ok(Expr::Assign {
                target,
                value: Box::new(value),
                span,
            });
        }
        Ok(expr)
    }

    fn parse_pipeline(&mut self) -> Result<Expr, MuninnError> {
        let mut expr = self.parse_equality()?;
        while self.match_where(|k| matches!(k, TokenKind::PipeGreater)) {
            let span = expr.span();
            let rhs = self.parse_call()?;
            let (callee, args) = match rhs {
                Expr::Call { callee, args, .. } => (callee, args),
                other => (Box::new(other), Vec::new()),
            };
            expr = Expr::Pipeline {
                lhs: Box::new(expr),
                callee,
                args,
                span,
            };
        }
        Ok(expr)
    }

    fn parse_equality(&mut self) -> Result<Expr, MuninnError> {
        let mut expr = self.parse_comparison()?;
        while self.match_where(|k| matches!(k, TokenKind::EqualEqual | TokenKind::BangEqual)) {
            let operator = match &self.previous().kind {
                TokenKind::EqualEqual => BinaryOp::Equal,
                TokenKind::BangEqual => BinaryOp::NotEqual,
                _ => unreachable!(),
            };
            let right = self.parse_comparison()?;
            let span = expr.span();
            expr = Expr::Binary {
                left: Box::new(expr),
                op: operator,
                right: Box::new(right),
                span,
            }
        }
        Ok(expr)
    }

    fn parse_comparison(&mut self) -> Result<Expr, MuninnError> {
        let mut expr = self.parse_term()?;
        while self.match_where(|k| {
            matches!(
                k,
                TokenKind::Greater
                    | TokenKind::GreaterEqual
                    | TokenKind::Less
                    | TokenKind::LessEqual
            )
        }) {
            let operator = match &self.previous().kind {
                TokenKind::Greater => BinaryOp::Greater,
                TokenKind::GreaterEqual => BinaryOp::GreaterEqual,
                TokenKind::Less => BinaryOp::Less,
                TokenKind::LessEqual => BinaryOp::LessEqual,
                _ => unreachable!(),
            };
            let right = self.parse_term()?;
            let span = expr.span();
            expr = Expr::Binary {
                left: Box::new(expr),
                op: operator,
                right: Box::new(right),
                span,
            };
        }
        Ok(expr)
    }

    fn parse_term(&mut self) -> Result<Expr, MuninnError> {
        let mut expr = self.parse_factor()?;
        while self.match_where(|k| matches!(k, TokenKind::Plus | TokenKind::Minus)) {
            let operator = match &self.previous().kind {
                TokenKind::Plus => BinaryOp::Add,
                TokenKind::Minus => BinaryOp::Subtract,
                _ => unreachable!(),
            };
            let right = self.parse_factor()?;
            let span = expr.span();
            expr = Expr::Binary {
                left: Box::new(expr),
                op: operator,
                right: Box::new(right),
                span,
            };
        }
        Ok(expr)
    }

    fn parse_factor(&mut self) -> Result<Expr, MuninnError> {
        let mut expr = self.parse_unary()?;
        while self.match_where(|k| matches!(k, TokenKind::Star | TokenKind::Slash)) {
            let operator = match &self.previous().kind {
                TokenKind::Star => BinaryOp::Multiply,
                TokenKind::Slash => BinaryOp::Divide,
                _ => unreachable!(),
            };
            let right = self.parse_unary()?;
            let span = expr.span();
            expr = Expr::Binary {
                left: Box::new(expr),
                op: operator,
                right: Box::new(right),
                span,
            };
        }
        Ok(expr)
    }

    fn parse_unary(&mut self) -> Result<Expr, MuninnError> {
        if self.match_where(|k| matches!(k, TokenKind::Bang | TokenKind::Minus)) {
            let op = match &self.previous().kind {
                TokenKind::Bang => UnaryOp::Not,
                TokenKind::Minus => UnaryOp::Negate,
                _ => unreachable!(),
            };
            let span = self.previous().span;
            let right = self.parse_unary()?;
            return Ok(Expr::Unary {
                op,
                expr: Box::new(right),
                span,
            });
        }
        self.parse_call()
    }

    fn parse_call(&mut self) -> Result<Expr, MuninnError> {
        let mut expr = self.parse_primary()?;

        loop {
            if self.match_where(|k| matches!(k, TokenKind::LeftParen)) {
                let span = expr.span();
                let mut args = Vec::new();
                if !self.check_where(|k| matches!(k, TokenKind::RightParen)) {
                    loop {
                        args.push(self.parse_expression()?);
                        if !self.match_where(|k| matches!(k, TokenKind::Comma)) {
                            break;
                        }
                    }
                }
                self.consume_where(
                    |k| matches!(k, TokenKind::RightParen),
                    "expected ')' after call arguments",
                )?;
                expr = Expr::Call {
                    callee: Box::new(expr),
                    args,
                    span,
                };
                continue;
            }

            if self.match_where(|k| matches!(k, TokenKind::Dot)) {
                let name = self.consume_identifier("expected property name after '.'")?;
                let span = self.previous().span;
                expr = Expr::Property {
                    object: Box::new(expr),
                    name,
                    span,
                };
                continue;
            }

            if self.match_where(|k| matches!(k, TokenKind::LeftBracket)) {
                let span = expr.span();
                let first = self.parse_expression()?;
                if self.match_where(|k| matches!(k, TokenKind::Comma)) {
                    let second = self.parse_expression()?;
                    self.consume_where(
                        |k| matches!(k, TokenKind::RightBracket),
                        "expected ']' after grid indices",
                    )?;
                    expr = Expr::GridIndex {
                        target: Box::new(expr),
                        x: Box::new(first),
                        y: Box::new(second),
                        span,
                    };
                } else {
                    self.consume_where(
                        |k| matches!(k, TokenKind::RightBracket),
                        "expected ']' after index expression",
                    )?;
                    expr = Expr::Index {
                        target: Box::new(expr),
                        index: Box::new(first),
                        span,
                    };
                }
                continue;
            }

            if self.match_where(|k| matches!(k, TokenKind::Question)) {
                let span = expr.span();
                expr = Expr::Try {
                    expr: Box::new(expr),
                    span,
                };
                continue;
            }

            break;
        }

        Ok(expr)
    }

    fn parse_primary(&mut self) -> Result<Expr, MuninnError> {
        if self.is_at_end() {
            return Err(MuninnError::new(
                "parser",
                "expected expression",
                self.peek().span,
            ));
        }

        let token = self.advance().clone();
        match token.kind {
            TokenKind::False => Ok(Expr::Bool(false, token.span)),
            TokenKind::True => Ok(Expr::Bool(true, token.span)),
            TokenKind::IntLiteral(v) => Ok(Expr::Int(v, token.span)),
            TokenKind::FloatLiteral(v) => Ok(Expr::Float(v, token.span)),
            TokenKind::StringLiteral(text) => self.parse_string_primary(text, token.span),
            TokenKind::SelfKw => Ok(Expr::SelfRef(token.span)),
            TokenKind::Identifier(name) => Ok(Expr::Variable(name, token.span)),
            TokenKind::LeftParen => {
                let expr = self.parse_expression()?;
                self.consume_where(
                    |k| matches!(k, TokenKind::RightParen),
                    "expected ')' after expression",
                )?;
                Ok(Expr::Grouping(Box::new(expr), token.span))
            }
            TokenKind::LeftBrace => {
                self.current = self.current.saturating_sub(1);
                let block = self.parse_block_expr()?;
                Ok(Expr::Block(block))
            }
            TokenKind::If => self.parse_if_expression(token.span),
            TokenKind::Unless => self.parse_unless_expression(token.span),
            TokenKind::LeftBracket => {
                let mut items = Vec::new();
                if !self.check_where(|k| matches!(k, TokenKind::RightBracket)) {
                    loop {
                        items.push(self.parse_expression()?);
                        if !self.match_where(|k| matches!(k, TokenKind::Comma)) {
                            break;
                        }
                    }
                }
                self.consume_where(
                    |k| matches!(k, TokenKind::RightBracket),
                    "expected ']' after array literal",
                )?;
                Ok(Expr::ArrayLiteral(items, token.span))
            }
            _ => Err(MuninnError::new(
                "parser",
                "expected expression",
                token.span,
            )),
        }
    }

    fn parse_if_expression(&mut self, span: Span) -> Result<Expr, MuninnError> {
        self.consume_where(
            |k| matches!(k, TokenKind::LeftParen),
            "expected '(' after if",
        )?;
        let condition = self.parse_expression()?;
        self.consume_where(
            |k| matches!(k, TokenKind::RightParen),
            "expected ')' after if condition",
        )?;
        let then_branch = self.parse_block_expr()?;
        self.consume_where(
            |k| matches!(k, TokenKind::Else),
            "if expression requires an else branch",
        )?;
        let else_branch = self.parse_block_expr()?;
        Ok(Expr::If {
            condition: Box::new(condition),
            then_branch,
            else_branch,
            span,
        })
    }

    fn parse_unless_expression(&mut self, span: Span) -> Result<Expr, MuninnError> {
        self.consume_where(
            |k| matches!(k, TokenKind::LeftParen),
            "expected '(' after unless",
        )?;
        let condition = self.parse_expression()?;
        self.consume_where(
            |k| matches!(k, TokenKind::RightParen),
            "expected ')' after unless condition",
        )?;
        let then_branch = self.parse_block_expr()?;
        let else_branch = if self.match_where(|k| matches!(k, TokenKind::Else)) {
            Some(self.parse_block_expr()?)
        } else {
            None
        };

        Ok(Expr::Unless {
            condition: Box::new(condition),
            then_branch,
            else_branch,
            span,
        })
    }

    fn parse_block_expr(&mut self) -> Result<BlockExpr, MuninnError> {
        let open = self.consume_where(
            |k| matches!(k, TokenKind::LeftBrace),
            "expected '{' to start block",
        )?;
        let mut statements = Vec::new();
        let mut tail = None;

        while !self.check_where(|k| matches!(k, TokenKind::RightBrace)) && !self.is_at_end() {
            if self.check_where(|k| matches!(k, TokenKind::Let | TokenKind::Fn | TokenKind::Class))
                || self.check_where(|k| {
                    matches!(k, TokenKind::Return | TokenKind::While | TokenKind::For)
                })
            {
                statements.push(self.parse_declaration()?);
                continue;
            }

            let expr = self.parse_expression()?;
            if self.match_where(|k| matches!(k, TokenKind::Semicolon)) {
                let span = expr.span();
                statements.push(Stmt::Expression { expr, span });
                continue;
            }

            tail = Some(Box::new(expr));
            break;
        }

        self.consume_where(
            |k| matches!(k, TokenKind::RightBrace),
            "expected '}' to close block",
        )?;

        Ok(BlockExpr {
            statements,
            tail,
            span: open.span,
        })
    }

    fn parse_type_expr(&mut self) -> Result<TypeExpr, MuninnError> {
        if let TokenKind::Identifier(name) = &self.peek().kind {
            if name == "Option" {
                self.advance();
                self.consume_where(
                    |k| matches!(k, TokenKind::LeftBracket),
                    "expected '[' after Option",
                )?;
                let inner = self.parse_type_expr()?;
                self.consume_where(
                    |k| matches!(k, TokenKind::RightBracket),
                    "expected ']' after Option inner type",
                )?;
                return Ok(TypeExpr::Option(Box::new(inner)));
            }
        }

        let base = match &self.peek().kind {
            TokenKind::TypeInt => {
                self.advance();
                TypeExpr::Int
            }
            TokenKind::TypeFloat => {
                self.advance();
                TypeExpr::Float
            }
            TokenKind::TypeString => {
                self.advance();
                TypeExpr::String
            }
            TokenKind::TypeBool => {
                self.advance();
                TypeExpr::Bool
            }
            TokenKind::TypeVoid => {
                self.advance();
                TypeExpr::Void
            }
            TokenKind::Identifier(name) => {
                let name = name.clone();
                self.advance();
                TypeExpr::Named(name)
            }
            _ => {
                return Err(MuninnError::new(
                    "parser",
                    "expected type annotation",
                    self.peek().span,
                ));
            }
        };

        if !self.match_where(|k| matches!(k, TokenKind::LeftBracket)) {
            return Ok(base);
        }

        let first = self.consume_int_literal("expected array/grid size")? as usize;
        if self.match_where(|k| matches!(k, TokenKind::Comma)) {
            let second = self.consume_int_literal("expected second grid size")? as usize;
            self.consume_where(
                |k| matches!(k, TokenKind::RightBracket),
                "expected ']' after grid dimensions",
            )?;
            Ok(TypeExpr::Grid {
                element: Box::new(base),
                width: first,
                height: second,
            })
        } else {
            self.consume_where(
                |k| matches!(k, TokenKind::RightBracket),
                "expected ']' after array size",
            )?;
            Ok(TypeExpr::Array {
                element: Box::new(base),
                len: first,
            })
        }
    }

    fn parse_string_primary(&self, text: String, span: Span) -> Result<Expr, MuninnError> {
        if !text.contains('{') && !text.contains('}') {
            return Ok(Expr::String(text, span));
        }

        let mut parts = Vec::new();
        let mut chars = text.chars().peekable();
        let mut current_text = String::new();

        while let Some(ch) = chars.next() {
            if ch == '{' {
                if chars.peek().is_some_and(|c| *c == '{') {
                    chars.next();
                    current_text.push('{');
                    continue;
                }

                if !current_text.is_empty() {
                    parts.push(InterpolationPart::Text(std::mem::take(&mut current_text)));
                }

                let mut expr_source = String::new();
                let mut depth = 0usize;
                loop {
                    let next = chars.next().ok_or_else(|| {
                        MuninnError::new("parser", "unterminated interpolation segment", span)
                    })?;
                    if next == '{' {
                        depth += 1;
                        expr_source.push(next);
                        continue;
                    }
                    if next == '}' {
                        if depth == 0 {
                            break;
                        }
                        depth -= 1;
                        expr_source.push(next);
                        continue;
                    }
                    expr_source.push(next);
                }

                if expr_source.trim().is_empty() {
                    return Err(MuninnError::new(
                        "parser",
                        "empty interpolation expression",
                        span,
                    ));
                }

                let lexer = Lexer::new(expr_source.trim());
                let tokens = lexer.lex().map_err(|mut errs| {
                    errs.pop().unwrap_or_else(|| {
                        MuninnError::new("parser", "invalid interpolation expression", span)
                    })
                })?;
                let mut parser = Parser::new(tokens);
                let expr = parser.parse_expression()?;
                if !parser.is_at_end() {
                    return Err(MuninnError::new(
                        "parser",
                        "unexpected tokens in interpolation expression",
                        span,
                    ));
                }
                parts.push(InterpolationPart::Expr(expr));
                continue;
            }

            if ch == '}' {
                if chars.peek().is_some_and(|c| *c == '}') {
                    chars.next();
                    current_text.push('}');
                    continue;
                }
                return Err(MuninnError::new(
                    "parser",
                    "unmatched '}' in string literal",
                    span,
                ));
            }

            current_text.push(ch);
        }

        if !current_text.is_empty() {
            parts.push(InterpolationPart::Text(current_text));
        }

        Ok(Expr::StringInterpolation { parts, span })
    }

    fn consume_identifier(&mut self, message: &str) -> Result<String, MuninnError> {
        let token = self.consume_where(|k| matches!(k, TokenKind::Identifier(_)), message)?;
        match token.kind {
            TokenKind::Identifier(name) => Ok(name),
            _ => unreachable!(),
        }
    }

    fn consume_int_literal(&mut self, message: &str) -> Result<i64, MuninnError> {
        let token = self.consume_where(|k| matches!(k, TokenKind::IntLiteral(_)), message)?;
        match token.kind {
            TokenKind::IntLiteral(v) => Ok(v),
            _ => unreachable!(),
        }
    }

    fn consume_where<F>(&mut self, predicate: F, message: &str) -> Result<Token, MuninnError>
    where
        F: Fn(&TokenKind) -> bool,
    {
        if self.check_where(predicate) {
            Ok(self.advance().clone())
        } else {
            Err(MuninnError::new("parser", message, self.peek().span))
        }
    }

    fn match_where<F>(&mut self, predicate: F) -> bool
    where
        F: Fn(&TokenKind) -> bool,
    {
        if self.check_where(predicate) {
            self.advance();
            true
        } else {
            false
        }
    }

    fn check_where<F>(&self, predicate: F) -> bool
    where
        F: Fn(&TokenKind) -> bool,
    {
        if self.is_at_end() {
            return false;
        }
        predicate(&self.peek().kind)
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
        if self.current == 0 {
            &self.tokens[0]
        } else {
            &self.tokens[self.current - 1]
        }
    }

    fn synchronize(&mut self) {
        while !self.is_at_end() {
            if matches!(self.peek().kind, TokenKind::Semicolon) {
                self.advance();
                return;
            }

            if matches!(
                self.peek().kind,
                TokenKind::Class
                    | TokenKind::Fn
                    | TokenKind::Let
                    | TokenKind::Return
                    | TokenKind::While
                    | TokenKind::For
                    | TokenKind::RightBrace
            ) {
                return;
            }

            self.advance();
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::lexer::Lexer;

    use super::Parser;

    #[test]
    fn parses_range_for() {
        let src = "for i in 0..10 { i; }";
        let tokens = Lexer::new(src).lex().expect("tokens");
        let mut parser = Parser::new(tokens);
        let program = parser.parse_program().expect("program");
        assert_eq!(program.statements.len(), 1);
    }

    #[test]
    fn parses_interpolation() {
        let src = "let x: Int = 10; let y: String = \"Value: {x}\";";
        let tokens = Lexer::new(src).lex().expect("tokens");
        let mut parser = Parser::new(tokens);
        let program = parser.parse_program().expect("program");
        assert_eq!(program.statements.len(), 2);
    }

    #[test]
    fn parses_option_type_and_try() {
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
        assert_eq!(program.statements.len(), 2);
    }

    #[test]
    fn rejects_empty_interpolation_expression() {
        let src = "let x: String = \"{}\";";
        let tokens = Lexer::new(src).lex().expect("tokens");
        let mut parser = Parser::new(tokens);
        assert!(parser.parse_program().is_err());
    }

    #[test]
    fn parses_inferred_let_binding() {
        let src = "let x = 1; let y = x + 2;";
        let tokens = Lexer::new(src).lex().expect("tokens");
        let mut parser = Parser::new(tokens);
        let program = parser.parse_program().expect("program");
        assert_eq!(program.statements.len(), 2);
    }

    #[test]
    fn recovers_and_reports_multiple_errors() {
        let src = "let x: Int = ; let y: Float = ;";
        let tokens = Lexer::new(src).lex().expect("tokens");
        let mut parser = Parser::new(tokens);
        let errors = parser.parse_program().expect_err("expected parser errors");
        assert!(errors.len() >= 2);
    }
}
