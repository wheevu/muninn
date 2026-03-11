use std::collections::HashMap;

use crate::ast::{
    BinaryOp, Expr, ExprKind, FunctionDecl, Program, Stmt, StmtKind, TypeExpr, UnaryOp,
};
use crate::bytecode::{BytecodeModule, Chunk, Constant, FunctionBytecode, OpCode};
use crate::error::MuninnError;
use crate::span::Span;

pub fn compile_program(program: &Program) -> Result<BytecodeModule, Vec<MuninnError>> {
    let mut compiler = ModuleCompiler::new();
    compiler.compile(program);
    compiler.finish()
}

struct ModuleCompiler {
    module: BytecodeModule,
    function_ids: HashMap<String, usize>,
    errors: Vec<MuninnError>,
}

impl ModuleCompiler {
    fn new() -> Self {
        Self {
            module: BytecodeModule::new(),
            function_ids: HashMap::new(),
            errors: Vec::new(),
        }
    }

    fn finish(self) -> Result<BytecodeModule, Vec<MuninnError>> {
        if self.errors.is_empty() {
            Ok(self.module)
        } else {
            Err(self.errors)
        }
    }

    fn compile(&mut self, program: &Program) {
        for statement in &program.statements {
            if let StmtKind::Function(function) = &statement.kind {
                let id = self.compile_function(function);
                self.function_ids.insert(function.name.clone(), id);
            }
        }

        let mut entry = FunctionCompiler::new("<entry>".to_string(), 0, false);
        for statement in &program.statements {
            if let StmtKind::Function(function) = &statement.kind {
                let Some(function_id) = self.function_ids.get(&function.name).copied() else {
                    continue;
                };
                if let Err(error) = entry.emit_constant(Constant::Function(function_id), function.span) {
                    self.errors.push(error);
                }
                if let Err(error) = entry.emit_named_op(OpCode::DefineGlobal, &function.name, function.span) {
                    self.errors.push(error);
                }
            }
        }

        let runtime_statements = program
            .statements
            .iter()
            .filter(|statement| !matches!(statement.kind, StmtKind::Function(_)))
            .collect::<Vec<_>>();
        for (index, statement) in runtime_statements.iter().enumerate() {
            let is_last = index + 1 == runtime_statements.len();
            if is_last && let StmtKind::Expr(expr) = &statement.kind {
                self.compile_expr(&mut entry, expr);
                entry.emit_op(OpCode::Return, statement.span);
                self.module.entry_function = self.push_function(entry.finish());
                return;
            }
            self.compile_stmt(&mut entry, statement);
        }
        entry.emit_op(OpCode::Nil, Span::default());
        entry.emit_op(OpCode::Return, Span::default());
        self.module.entry_function = self.push_function(entry.finish());
    }

    fn compile_function(&mut self, function: &FunctionDecl) -> usize {
        let expects_return_value = function.return_type != TypeExpr::Void;
        let mut compiler = FunctionCompiler::new(
            function.name.clone(),
            function.params.len(),
            expects_return_value,
        );
        for param in &function.params {
            compiler.define_parameter(param.name.clone());
        }
        for statement in &function.body.statements {
            self.compile_stmt(&mut compiler, statement);
        }
        compiler.emit_op(OpCode::Nil, function.span);
        compiler.emit_op(OpCode::Return, function.span);
        self.push_function(compiler.finish())
    }

    fn push_function(&mut self, function: FunctionBytecode) -> usize {
        self.module.functions.push(function);
        self.module.functions.len() - 1
    }

    fn compile_stmt(&mut self, compiler: &mut FunctionCompiler, stmt: &Stmt) {
        match &stmt.kind {
            StmtKind::Let {
                name,
                initializer,
                ..
            } => {
                self.compile_expr(compiler, initializer);
                if compiler.scope_depth == 0 {
                    if let Err(error) = compiler.emit_named_op(OpCode::DefineGlobal, name, stmt.span) {
                        self.errors.push(error);
                    }
                } else {
                    let slot = compiler.define_local(name.clone());
                    compiler.emit_slot_op(OpCode::SetLocal, slot, stmt.span);
                }
            }
            StmtKind::Function(_) => {}
            StmtKind::Return(value) => {
                if let Some(value) = value {
                    self.compile_expr(compiler, value);
                } else {
                    compiler.emit_op(OpCode::Nil, stmt.span);
                }
                compiler.emit_op(OpCode::Return, stmt.span);
            }
            StmtKind::While { condition, body } => {
                let loop_start = compiler.current_offset();
                self.compile_expr(compiler, condition);
                let exit_jump = compiler.emit_jump(OpCode::JumpIfFalse, stmt.span);
                compiler.emit_op(OpCode::Pop, stmt.span);
                compiler.enter_scope();
                for body_stmt in &body.statements {
                    self.compile_stmt(compiler, body_stmt);
                }
                compiler.exit_scope();
                compiler.emit_loop(loop_start, stmt.span);
                compiler.patch_jump(exit_jump, stmt.span, &mut self.errors);
                compiler.emit_op(OpCode::Pop, stmt.span);
            }
            StmtKind::If {
                condition,
                then_branch,
                else_branch,
            } => {
                self.compile_expr(compiler, condition);
                let else_jump = compiler.emit_jump(OpCode::JumpIfFalse, stmt.span);
                compiler.emit_op(OpCode::Pop, stmt.span);
                compiler.enter_scope();
                for then_stmt in &then_branch.statements {
                    self.compile_stmt(compiler, then_stmt);
                }
                compiler.exit_scope();
                let end_jump = else_branch.as_ref().map(|_| compiler.emit_jump(OpCode::Jump, stmt.span));
                compiler.patch_jump(else_jump, stmt.span, &mut self.errors);
                compiler.emit_op(OpCode::Pop, stmt.span);
                if let Some(else_branch) = else_branch {
                    compiler.enter_scope();
                    for else_stmt in &else_branch.statements {
                        self.compile_stmt(compiler, else_stmt);
                    }
                    compiler.exit_scope();
                }
                if let Some(end_jump) = end_jump {
                    compiler.patch_jump(end_jump, stmt.span, &mut self.errors);
                }
            }
            StmtKind::Assign { name, value, .. } => {
                self.compile_expr(compiler, value);
                if let Some(slot) = compiler.resolve_local(name) {
                    compiler.emit_slot_op(OpCode::SetLocal, slot, stmt.span);
                } else if let Err(error) = compiler.emit_named_op(OpCode::SetGlobal, name, stmt.span) {
                    self.errors.push(error);
                }
            }
            StmtKind::Expr(expr) => {
                self.compile_expr(compiler, expr);
                compiler.emit_op(OpCode::Pop, stmt.span);
            }
        }
    }

    fn compile_expr(&mut self, compiler: &mut FunctionCompiler, expr: &Expr) {
        match &expr.kind {
            ExprKind::Int(value) => {
                if let Err(error) = compiler.emit_constant(Constant::Int(*value), expr.span) {
                    self.errors.push(error);
                }
            }
            ExprKind::Float(value) => {
                if let Err(error) = compiler.emit_constant(Constant::Float(*value), expr.span) {
                    self.errors.push(error);
                }
            }
            ExprKind::Bool(value) => {
                compiler.emit_op(if *value { OpCode::True } else { OpCode::False }, expr.span);
            }
            ExprKind::String(value) => {
                if let Err(error) = compiler.emit_constant(Constant::String(value.clone()), expr.span) {
                    self.errors.push(error);
                }
            }
            ExprKind::Variable(name) => {
                if let Some(slot) = compiler.resolve_local(name) {
                    compiler.emit_slot_op(OpCode::GetLocal, slot, expr.span);
                } else if let Err(error) = compiler.emit_named_op(OpCode::GetGlobal, name, expr.span) {
                    self.errors.push(error);
                }
            }
            ExprKind::Grouping(inner) => self.compile_expr(compiler, inner),
            ExprKind::Unary { op, expr: inner } => {
                self.compile_expr(compiler, inner);
                match op {
                    UnaryOp::Negate => compiler.emit_op(OpCode::Negate, expr.span),
                    UnaryOp::Not => compiler.emit_op(OpCode::Not, expr.span),
                }
            }
            ExprKind::Binary { left, op, right } => match op {
                BinaryOp::And => {
                    self.compile_expr(compiler, left);
                    let end_jump = compiler.emit_jump(OpCode::JumpIfFalse, expr.span);
                    compiler.emit_op(OpCode::Pop, expr.span);
                    self.compile_expr(compiler, right);
                    compiler.patch_jump(end_jump, expr.span, &mut self.errors);
                }
                BinaryOp::Or => {
                    self.compile_expr(compiler, left);
                    let else_jump = compiler.emit_jump(OpCode::JumpIfFalse, expr.span);
                    let end_jump = compiler.emit_jump(OpCode::Jump, expr.span);
                    compiler.patch_jump(else_jump, expr.span, &mut self.errors);
                    compiler.emit_op(OpCode::Pop, expr.span);
                    self.compile_expr(compiler, right);
                    compiler.patch_jump(end_jump, expr.span, &mut self.errors);
                }
                BinaryOp::NotEqual => {
                    self.compile_expr(compiler, left);
                    self.compile_expr(compiler, right);
                    compiler.emit_op(OpCode::Equal, expr.span);
                    compiler.emit_op(OpCode::Not, expr.span);
                }
                BinaryOp::GreaterEqual => {
                    self.compile_expr(compiler, left);
                    self.compile_expr(compiler, right);
                    compiler.emit_op(OpCode::Less, expr.span);
                    compiler.emit_op(OpCode::Not, expr.span);
                }
                BinaryOp::LessEqual => {
                    self.compile_expr(compiler, left);
                    self.compile_expr(compiler, right);
                    compiler.emit_op(OpCode::Greater, expr.span);
                    compiler.emit_op(OpCode::Not, expr.span);
                }
                _ => {
                    self.compile_expr(compiler, left);
                    self.compile_expr(compiler, right);
                    let opcode = match op {
                        BinaryOp::Add => OpCode::Add,
                        BinaryOp::Subtract => OpCode::Subtract,
                        BinaryOp::Multiply => OpCode::Multiply,
                        BinaryOp::Divide => OpCode::Divide,
                        BinaryOp::Equal => OpCode::Equal,
                        BinaryOp::Greater => OpCode::Greater,
                        BinaryOp::Less => OpCode::Less,
                        BinaryOp::NotEqual
                        | BinaryOp::GreaterEqual
                        | BinaryOp::LessEqual
                        | BinaryOp::And
                        | BinaryOp::Or => unreachable!(),
                    };
                    compiler.emit_op(opcode, expr.span);
                }
            },
            ExprKind::Call { callee, args } => {
                self.compile_expr(compiler, callee);
                for arg in args {
                    self.compile_expr(compiler, arg);
                }
                compiler.emit_op(OpCode::Call, expr.span);
                compiler.emit_u8(args.len() as u8, expr.span);
            }
        }
    }
}

struct FunctionCompiler {
    name: String,
    arity: usize,
    expects_return_value: bool,
    chunk: Chunk,
    locals: Vec<Local>,
    next_slot: usize,
    max_slot: usize,
    scope_depth: usize,
}

#[derive(Debug, Clone)]
struct Local {
    name: String,
    depth: usize,
    slot: usize,
}

impl FunctionCompiler {
    fn new(name: String, arity: usize, expects_return_value: bool) -> Self {
        Self {
            name,
            arity,
            expects_return_value,
            chunk: Chunk::new(),
            locals: Vec::new(),
            next_slot: 0,
            max_slot: 0,
            scope_depth: 0,
        }
    }

    fn finish(self) -> FunctionBytecode {
        FunctionBytecode {
            name: self.name,
            arity: self.arity,
            local_count: self.max_slot,
            expects_return_value: self.expects_return_value,
            chunk: self.chunk,
        }
    }

    fn define_parameter(&mut self, name: String) {
        let slot = self.next_slot;
        self.next_slot += 1;
        self.max_slot = self.max_slot.max(self.next_slot);
        self.locals.push(Local {
            name,
            depth: 0,
            slot,
        });
    }

    fn define_local(&mut self, name: String) -> usize {
        let slot = self.next_slot;
        self.next_slot += 1;
        self.max_slot = self.max_slot.max(self.next_slot);
        self.locals.push(Local {
            name,
            depth: self.scope_depth,
            slot,
        });
        slot
    }

    fn resolve_local(&self, name: &str) -> Option<usize> {
        self.locals
            .iter()
            .rev()
            .find(|local| local.name == name)
            .map(|local| local.slot)
    }

    fn enter_scope(&mut self) {
        self.scope_depth += 1;
    }

    fn exit_scope(&mut self) {
        while self
            .locals
            .last()
            .is_some_and(|local| local.depth == self.scope_depth)
        {
            self.locals.pop();
        }
        self.scope_depth = self.scope_depth.saturating_sub(1);
    }

    fn current_offset(&self) -> usize {
        self.chunk.code.len()
    }

    fn emit_op(&mut self, op: OpCode, span: Span) {
        self.chunk.write_op(op, span);
    }

    fn emit_u8(&mut self, value: u8, span: Span) {
        self.chunk.write_u8(value, span);
    }

    fn emit_u16(&mut self, value: u16, span: Span) {
        self.chunk.write_u16(value, span);
    }

    fn emit_slot_op(&mut self, op: OpCode, slot: usize, span: Span) {
        self.emit_op(op, span);
        self.emit_u16(slot as u16, span);
    }

    fn emit_named_op(&mut self, op: OpCode, name: &str, span: Span) -> Result<(), MuninnError> {
        let index = self
            .chunk
            .add_constant(Constant::String(name.to_string()))
            .map_err(|msg| MuninnError::new("compiler", msg, span))?;
        self.emit_op(op, span);
        self.emit_u16(index, span);
        Ok(())
    }

    fn emit_constant(&mut self, constant: Constant, span: Span) -> Result<(), MuninnError> {
        let index = self
            .chunk
            .add_constant(constant)
            .map_err(|msg| MuninnError::new("compiler", msg, span))?;
        self.emit_op(OpCode::Constant, span);
        self.emit_u16(index, span);
        Ok(())
    }

    fn emit_jump(&mut self, op: OpCode, span: Span) -> usize {
        self.emit_op(op, span);
        let patch_at = self.current_offset();
        self.emit_u16(u16::MAX, span);
        patch_at
    }

    fn patch_jump(&mut self, patch_at: usize, span: Span, errors: &mut Vec<MuninnError>) {
        let jump = self.current_offset().saturating_sub(patch_at + 2);
        if jump > u16::MAX as usize {
            errors.push(MuninnError::new(
                "compiler",
                "jump offset overflow",
                span,
            ));
            return;
        }
        let bytes = (jump as u16).to_le_bytes();
        self.chunk.code[patch_at] = bytes[0];
        self.chunk.code[patch_at + 1] = bytes[1];
    }

    fn emit_loop(&mut self, loop_start: usize, span: Span) {
        self.emit_op(OpCode::Loop, span);
        let jump = self.current_offset().saturating_sub(loop_start) + 2;
        self.emit_u16(jump as u16, span);
    }
}
