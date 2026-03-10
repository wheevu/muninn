use std::collections::HashMap;
use std::rc::Rc;

use crate::ast::{
    AssignTarget, BinaryOp, BlockExpr, Expr, FunctionDecl, Program, Stmt, UnaryOp, VecBinaryMode,
};
use crate::bytecode::{BytecodeModule, Chunk, ClassBytecode, Constant, FunctionBytecode, OpCode};
use crate::error::MuninnError;
use crate::span::Span;

pub fn compile_program(program: &Program) -> Result<BytecodeModule, Vec<MuninnError>> {
    let mut compiler = ModuleCompiler::new();
    match compiler.compile(program) {
        Ok(module) => Ok(module),
        Err(errors) => Err(errors),
    }
}

struct ModuleCompiler {
    module: BytecodeModule,
    errors: Vec<MuninnError>,
    temp_counter: usize,
}

impl ModuleCompiler {
    fn new() -> Self {
        Self {
            module: BytecodeModule::new(),
            errors: Vec::new(),
            temp_counter: 0,
        }
    }

    fn compile(&mut self, program: &Program) -> Result<BytecodeModule, Vec<MuninnError>> {
        let mut main_compiler = FunctionCompiler::new("__main".to_string(), 0, false);
        for stmt in &program.statements {
            self.compile_stmt(&mut main_compiler, stmt);
        }
        main_compiler.emit_op(OpCode::Nil);
        main_compiler.emit_op(OpCode::Return);

        let main_id = self.push_function(main_compiler.finish());
        self.module.entry_function = main_id;

        if self.errors.is_empty() {
            Ok(self.module.clone())
        } else {
            Err(std::mem::take(&mut self.errors))
        }
    }

    fn push_function(&mut self, function: FunctionBytecode) -> usize {
        self.module.functions.push(Rc::new(function));
        self.module.functions.len() - 1
    }

    fn compile_stmt(&mut self, compiler: &mut FunctionCompiler, stmt: &Stmt) {
        match stmt {
            Stmt::Let {
                name,
                initializer,
                span,
                ..
            } => {
                self.compile_expr(compiler, initializer);
                if compiler.scope_depth == 0 {
                    let name_idx = compiler.make_constant(Constant::String(name.clone()));
                    compiler.emit_op(OpCode::DefineGlobal);
                    compiler.emit_u16(name_idx);
                } else {
                    let slot = compiler.declare_local(name.clone(), *span);
                    compiler.emit_op(OpCode::SetLocal);
                    compiler.emit_u16(slot as u16);
                    compiler.emit_op(OpCode::Pop);
                }
            }
            Stmt::Function(function) => {
                let function_id = self.compile_function(function, false);
                let const_idx = compiler.make_constant(Constant::Function(function_id));
                compiler.emit_op(OpCode::Constant);
                compiler.emit_u16(const_idx);

                if compiler.scope_depth == 0 {
                    let name_idx = compiler.make_constant(Constant::String(function.name.clone()));
                    compiler.emit_op(OpCode::DefineGlobal);
                    compiler.emit_u16(name_idx);
                } else {
                    let slot = compiler.declare_local(function.name.clone(), function.span);
                    compiler.emit_op(OpCode::SetLocal);
                    compiler.emit_u16(slot as u16);
                    compiler.emit_op(OpCode::Pop);
                }
            }
            Stmt::Class(class) => {
                let mut method_map = HashMap::new();
                let mut init = None;
                for method in &class.methods {
                    let id = self.compile_function(method, true);
                    method_map.insert(method.name.clone(), id);
                }
                if let Some(init_fn) = &class.init {
                    init = Some(self.compile_function(init_fn, true));
                }

                let class_id = self.module.classes.len();
                self.module.classes.push(Rc::new(ClassBytecode {
                    name: class.name.clone(),
                    fields: class.fields.iter().map(|f| f.name.clone()).collect(),
                    methods: method_map,
                    init,
                }));

                let class_const = compiler.make_constant(Constant::Class(class_id));
                compiler.emit_op(OpCode::Constant);
                compiler.emit_u16(class_const);

                if compiler.scope_depth == 0 {
                    let name_idx = compiler.make_constant(Constant::String(class.name.clone()));
                    compiler.emit_op(OpCode::DefineGlobal);
                    compiler.emit_u16(name_idx);
                } else {
                    let slot = compiler.declare_local(class.name.clone(), class.span);
                    compiler.emit_op(OpCode::SetLocal);
                    compiler.emit_u16(slot as u16);
                    compiler.emit_op(OpCode::Pop);
                }
            }
            Stmt::Return { value, .. } => {
                if let Some(value) = value {
                    self.compile_expr(compiler, value);
                } else {
                    compiler.emit_op(OpCode::Nil);
                }
                compiler.emit_op(OpCode::Return);
            }
            Stmt::While {
                condition, body, ..
            } => {
                let loop_start = compiler.current_offset();
                self.compile_expr(compiler, condition);
                let exit_jump = compiler.emit_jump(OpCode::JumpIfFalse);
                compiler.emit_op(OpCode::Pop);
                self.compile_block(compiler, body, false);
                compiler.emit_loop(loop_start);
                compiler.patch_jump(exit_jump);
                compiler.emit_op(OpCode::Pop);
            }
            Stmt::ForRange { span, .. } => {
                self.errors.push(MuninnError::new(
                    "compiler",
                    "for-range should be desugared before compilation",
                    *span,
                ));
            }
            Stmt::Expression { expr, .. } => {
                self.compile_expr(compiler, expr);
                compiler.emit_op(OpCode::Pop);
            }
        }
    }

    fn compile_function(&mut self, function: &FunctionDecl, is_method: bool) -> usize {
        let arity = function.params.len() + usize::from(is_method);
        let mut fn_compiler = FunctionCompiler::new(function.name.clone(), arity, is_method);
        if is_method {
            fn_compiler.declare_local("self".to_string(), function.span);
        }
        for param in &function.params {
            fn_compiler.declare_local(param.name.clone(), param.span);
        }

        self.compile_block(&mut fn_compiler, &function.body, true);
        fn_compiler.emit_op(OpCode::Return);

        fn_compiler.emit_op(OpCode::Nil);
        fn_compiler.emit_op(OpCode::Return);
        self.push_function(fn_compiler.finish())
    }

    fn compile_block(
        &mut self,
        compiler: &mut FunctionCompiler,
        block: &BlockExpr,
        keep_value: bool,
    ) {
        compiler.begin_scope();
        for stmt in &block.statements {
            self.compile_stmt(compiler, stmt);
        }
        if let Some(expr) = &block.tail {
            self.compile_expr(compiler, expr);
        } else {
            compiler.emit_op(OpCode::Nil);
        }
        compiler.end_scope();

        if !keep_value {
            compiler.emit_op(OpCode::Pop);
        }
    }

    fn compile_expr(&mut self, compiler: &mut FunctionCompiler, expr: &Expr) {
        match expr {
            Expr::Int(value, _) => {
                let idx = compiler.make_constant(Constant::Int(*value));
                compiler.emit_op(OpCode::Constant);
                compiler.emit_u16(idx);
            }
            Expr::Float(value, _) => {
                let idx = compiler.make_constant(Constant::Float(*value));
                compiler.emit_op(OpCode::Constant);
                compiler.emit_u16(idx);
            }
            Expr::Bool(true, _) => compiler.emit_op(OpCode::True),
            Expr::Bool(false, _) => compiler.emit_op(OpCode::False),
            Expr::String(value, _) => {
                let idx = compiler.make_constant(Constant::String(value.clone()));
                compiler.emit_op(OpCode::Constant);
                compiler.emit_u16(idx);
            }
            Expr::Variable(name, span) => {
                if let Some(slot) = compiler.resolve_local(name) {
                    compiler.emit_op(OpCode::GetLocal);
                    compiler.emit_u16(slot as u16);
                } else {
                    let idx = compiler.make_constant(Constant::String(name.clone()));
                    compiler.emit_op(OpCode::GetGlobal);
                    compiler.emit_u16(idx);
                }

                if name == "self" && compiler.resolve_local(name).is_none() {
                    self.errors.push(MuninnError::new(
                        "compiler",
                        "'self' is only available inside methods",
                        *span,
                    ));
                }
            }
            Expr::SelfRef(span) => {
                if let Some(slot) = compiler.resolve_local("self") {
                    compiler.emit_op(OpCode::GetLocal);
                    compiler.emit_u16(slot as u16);
                } else {
                    self.errors.push(MuninnError::new(
                        "compiler",
                        "'self' is only available inside methods",
                        *span,
                    ));
                }
            }
            Expr::ArrayLiteral(items, _) => {
                if items.len() > u16::MAX as usize {
                    self.errors.push(MuninnError::new(
                        "compiler",
                        format!(
                            "array literal has {} items, exceeding {}",
                            items.len(),
                            u16::MAX
                        ),
                        expr.span(),
                    ));
                    return;
                }
                for item in items {
                    self.compile_expr(compiler, item);
                }
                compiler.emit_op(OpCode::BuildArray);
                compiler.emit_u16(items.len() as u16);
            }
            Expr::Block(block) => self.compile_block(compiler, block, true),
            Expr::Grouping(inner, _) => self.compile_expr(compiler, inner),
            Expr::Unary { op, expr, .. } => {
                self.compile_expr(compiler, expr);
                match op {
                    UnaryOp::Negate => compiler.emit_op(OpCode::Negate),
                    UnaryOp::Not => compiler.emit_op(OpCode::Not),
                }
            }
            Expr::Binary {
                left, op, right, ..
            } => {
                self.compile_expr(compiler, left);
                self.compile_expr(compiler, right);
                match op {
                    BinaryOp::Add => compiler.emit_op(OpCode::Add),
                    BinaryOp::Subtract => compiler.emit_op(OpCode::Subtract),
                    BinaryOp::Multiply => compiler.emit_op(OpCode::Multiply),
                    BinaryOp::Divide => compiler.emit_op(OpCode::Divide),
                    BinaryOp::Equal => compiler.emit_op(OpCode::Equal),
                    BinaryOp::NotEqual => {
                        compiler.emit_op(OpCode::Equal);
                        compiler.emit_op(OpCode::Not);
                    }
                    BinaryOp::Greater => compiler.emit_op(OpCode::Greater),
                    BinaryOp::Less => compiler.emit_op(OpCode::Less),
                    BinaryOp::GreaterEqual => {
                        compiler.emit_op(OpCode::Less);
                        compiler.emit_op(OpCode::Not);
                    }
                    BinaryOp::LessEqual => {
                        compiler.emit_op(OpCode::Greater);
                        compiler.emit_op(OpCode::Not);
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
            } => self.compile_vec_binary(compiler, left, *op, right, *len, *mode, *span),
            Expr::If {
                condition,
                then_branch,
                else_branch,
                ..
            } => {
                self.compile_expr(compiler, condition);
                let else_jump = compiler.emit_jump(OpCode::JumpIfFalse);
                compiler.emit_op(OpCode::Pop);
                self.compile_block(compiler, then_branch, true);
                let end_jump = compiler.emit_jump(OpCode::Jump);
                compiler.patch_jump(else_jump);
                compiler.emit_op(OpCode::Pop);
                self.compile_block(compiler, else_branch, true);
                compiler.patch_jump(end_jump);
            }
            Expr::Unless { span, .. } => self.errors.push(MuninnError::new(
                "compiler",
                "unless should be desugared before compilation",
                *span,
            )),
            Expr::Call { callee, args, .. } => {
                if let Expr::Property { object, name, .. } = callee.as_ref() {
                    self.compile_expr(compiler, object);
                    for arg in args {
                        self.compile_expr(compiler, arg);
                    }
                    let name_idx = compiler.make_constant(Constant::String(name.clone()));
                    compiler.emit_op(OpCode::Invoke);
                    compiler.emit_u16(name_idx);
                    compiler.emit_u8(args.len() as u8);
                } else {
                    self.compile_expr(compiler, callee);
                    for arg in args {
                        self.compile_expr(compiler, arg);
                    }
                    compiler.emit_op(OpCode::Call);
                    compiler.emit_u8(args.len() as u8);
                }
            }
            Expr::Pipeline { span, .. } => self.errors.push(MuninnError::new(
                "compiler",
                "pipeline should be desugared before compilation",
                *span,
            )),
            Expr::Property { object, name, .. } => {
                self.compile_expr(compiler, object);
                let name_idx = compiler.make_constant(Constant::String(name.clone()));
                compiler.emit_op(OpCode::GetProperty);
                compiler.emit_u16(name_idx);
            }
            Expr::Index { target, index, .. } => {
                self.compile_expr(compiler, target);
                self.compile_expr(compiler, index);
                compiler.emit_op(OpCode::GetIndex);
            }
            Expr::GridIndex { span, .. } => self.errors.push(MuninnError::new(
                "compiler",
                "grid index should be desugared before compilation",
                *span,
            )),
            Expr::Assign {
                target,
                value,
                span,
            } => self.compile_assignment(compiler, target, value, *span),
            Expr::Try { span, .. } => self.errors.push(MuninnError::new(
                "compiler",
                "'?' should be desugared before compilation",
                *span,
            )),
            Expr::StringInterpolation { span, .. } => self.errors.push(MuninnError::new(
                "compiler",
                "string interpolation should be desugared before compilation",
                *span,
            )),
        }
    }

    fn compile_vec_binary(
        &mut self,
        compiler: &mut FunctionCompiler,
        left: &Expr,
        op: BinaryOp,
        right: &Expr,
        len: usize,
        mode: VecBinaryMode,
        span: Span,
    ) {
        if !matches!(
            op,
            BinaryOp::Add | BinaryOp::Subtract | BinaryOp::Multiply | BinaryOp::Divide
        ) {
            self.errors.push(MuninnError::new(
                "compiler",
                "vectorized op only supports +, -, *, /",
                span,
            ));
            return;
        }

        if len > u16::MAX as usize {
            self.errors.push(MuninnError::new(
                "compiler",
                format!(
                    "vectorized operation length {} exceeds maximum {}",
                    len,
                    u16::MAX
                ),
                span,
            ));
            return;
        }

        compiler.begin_scope();
        let left_slot = compiler.declare_local(self.next_temp_name("vec_left"), span);
        let right_slot = compiler.declare_local(self.next_temp_name("vec_right"), span);
        let result_slot = compiler.declare_local(self.next_temp_name("vec_out"), span);
        let index_slot = compiler.declare_local(self.next_temp_name("vec_i"), span);

        self.compile_expr(compiler, left);
        compiler.emit_op(OpCode::SetLocal);
        compiler.emit_u16(left_slot as u16);
        compiler.emit_op(OpCode::Pop);

        self.compile_expr(compiler, right);
        compiler.emit_op(OpCode::SetLocal);
        compiler.emit_u16(right_slot as u16);
        compiler.emit_op(OpCode::Pop);

        for _ in 0..len {
            compiler.emit_op(OpCode::Nil);
        }
        compiler.emit_op(OpCode::BuildArray);
        compiler.emit_u16(len as u16);
        compiler.emit_op(OpCode::SetLocal);
        compiler.emit_u16(result_slot as u16);
        compiler.emit_op(OpCode::Pop);

        let zero_idx = compiler.make_constant(Constant::Int(0));
        compiler.emit_op(OpCode::Constant);
        compiler.emit_u16(zero_idx);
        compiler.emit_op(OpCode::SetLocal);
        compiler.emit_u16(index_slot as u16);
        compiler.emit_op(OpCode::Pop);

        let loop_start = compiler.current_offset();
        compiler.emit_op(OpCode::GetLocal);
        compiler.emit_u16(index_slot as u16);
        let len_idx = compiler.make_constant(Constant::Int(len as i64));
        compiler.emit_op(OpCode::Constant);
        compiler.emit_u16(len_idx);
        compiler.emit_op(OpCode::Less);

        let exit_jump = compiler.emit_jump(OpCode::JumpIfFalse);
        compiler.emit_op(OpCode::Pop);

        compiler.emit_op(OpCode::GetLocal);
        compiler.emit_u16(result_slot as u16);
        compiler.emit_op(OpCode::GetLocal);
        compiler.emit_u16(index_slot as u16);

        match mode {
            VecBinaryMode::ArrayArray => {
                compiler.emit_op(OpCode::GetLocal);
                compiler.emit_u16(left_slot as u16);
                compiler.emit_op(OpCode::GetLocal);
                compiler.emit_u16(index_slot as u16);
                compiler.emit_op(OpCode::GetIndex);

                compiler.emit_op(OpCode::GetLocal);
                compiler.emit_u16(right_slot as u16);
                compiler.emit_op(OpCode::GetLocal);
                compiler.emit_u16(index_slot as u16);
                compiler.emit_op(OpCode::GetIndex);
            }
            VecBinaryMode::ArrayScalarRight => {
                compiler.emit_op(OpCode::GetLocal);
                compiler.emit_u16(left_slot as u16);
                compiler.emit_op(OpCode::GetLocal);
                compiler.emit_u16(index_slot as u16);
                compiler.emit_op(OpCode::GetIndex);

                compiler.emit_op(OpCode::GetLocal);
                compiler.emit_u16(right_slot as u16);
            }
            VecBinaryMode::ScalarArrayLeft => {
                compiler.emit_op(OpCode::GetLocal);
                compiler.emit_u16(left_slot as u16);

                compiler.emit_op(OpCode::GetLocal);
                compiler.emit_u16(right_slot as u16);
                compiler.emit_op(OpCode::GetLocal);
                compiler.emit_u16(index_slot as u16);
                compiler.emit_op(OpCode::GetIndex);
            }
        }

        match op {
            BinaryOp::Add => compiler.emit_op(OpCode::Add),
            BinaryOp::Subtract => compiler.emit_op(OpCode::Subtract),
            BinaryOp::Multiply => compiler.emit_op(OpCode::Multiply),
            BinaryOp::Divide => compiler.emit_op(OpCode::Divide),
            _ => unreachable!(),
        }

        compiler.emit_op(OpCode::SetIndex);
        compiler.emit_op(OpCode::Pop);

        compiler.emit_op(OpCode::GetLocal);
        compiler.emit_u16(index_slot as u16);
        let one_idx = compiler.make_constant(Constant::Int(1));
        compiler.emit_op(OpCode::Constant);
        compiler.emit_u16(one_idx);
        compiler.emit_op(OpCode::Add);
        compiler.emit_op(OpCode::SetLocal);
        compiler.emit_u16(index_slot as u16);
        compiler.emit_op(OpCode::Pop);

        compiler.emit_loop(loop_start);
        compiler.patch_jump(exit_jump);
        compiler.emit_op(OpCode::Pop);

        compiler.emit_op(OpCode::GetLocal);
        compiler.emit_u16(result_slot as u16);
        compiler.end_scope();
    }

    fn compile_assignment(
        &mut self,
        compiler: &mut FunctionCompiler,
        target: &AssignTarget,
        value: &Expr,
        span: Span,
    ) {
        match target {
            AssignTarget::Variable(name, _) => {
                self.compile_expr(compiler, value);
                if let Some(slot) = compiler.resolve_local(name) {
                    compiler.emit_op(OpCode::SetLocal);
                    compiler.emit_u16(slot as u16);
                } else {
                    let idx = compiler.make_constant(Constant::String(name.clone()));
                    compiler.emit_op(OpCode::SetGlobal);
                    compiler.emit_u16(idx);
                }
            }
            AssignTarget::Property { object, name, .. } => {
                self.compile_expr(compiler, object);
                self.compile_expr(compiler, value);
                let idx = compiler.make_constant(Constant::String(name.clone()));
                compiler.emit_op(OpCode::SetProperty);
                compiler.emit_u16(idx);
            }
            AssignTarget::Index { target, index, .. } => {
                self.compile_expr(compiler, target);
                self.compile_expr(compiler, index);
                self.compile_expr(compiler, value);
                compiler.emit_op(OpCode::SetIndex);
            }
            AssignTarget::GridIndex { .. } => self.errors.push(MuninnError::new(
                "compiler",
                "grid assignment should be desugared before compilation",
                span,
            )),
        }
    }

    fn next_temp_name(&mut self, prefix: &str) -> String {
        let id = self.temp_counter;
        self.temp_counter += 1;
        format!("__{}_{}", prefix, id)
    }
}

#[derive(Clone)]
struct FunctionCompiler {
    name: String,
    arity: usize,
    chunk: Chunk,
    scopes: Vec<HashMap<String, usize>>,
    scope_depth: usize,
    next_local: usize,
    max_local: usize,
}

impl FunctionCompiler {
    fn new(name: String, arity: usize, _is_method: bool) -> Self {
        Self {
            name,
            arity,
            chunk: Chunk::new(),
            scopes: vec![HashMap::new()],
            scope_depth: 0,
            next_local: 0,
            max_local: 0,
        }
    }

    fn finish(self) -> FunctionBytecode {
        FunctionBytecode {
            name: self.name,
            arity: self.arity,
            local_count: self.max_local.max(self.next_local),
            chunk: self.chunk,
        }
    }

    fn begin_scope(&mut self) {
        self.scope_depth += 1;
        self.scopes.push(HashMap::new());
    }

    fn end_scope(&mut self) {
        self.scope_depth = self.scope_depth.saturating_sub(1);
        self.scopes.pop();
    }

    fn declare_local(&mut self, name: String, _span: Span) -> usize {
        let slot = self.next_local;
        assert!(
            slot <= u16::MAX as usize,
            "local variable slots exceed u16 operand capacity"
        );
        self.next_local += 1;
        self.max_local = self.max_local.max(self.next_local);
        self.scopes.last_mut().expect("scope").insert(name, slot);
        slot
    }

    fn resolve_local(&self, name: &str) -> Option<usize> {
        for scope in self.scopes.iter().rev() {
            if let Some(slot) = scope.get(name) {
                return Some(*slot);
            }
        }
        None
    }

    fn make_constant(&mut self, constant: Constant) -> u16 {
        self.chunk.add_constant(constant)
    }

    fn emit_op(&mut self, op: OpCode) {
        self.chunk.write_op(op);
    }

    fn emit_u8(&mut self, value: u8) {
        self.chunk.write_u8(value);
    }

    fn emit_u16(&mut self, value: u16) {
        self.chunk.write_u16(value);
    }

    fn current_offset(&self) -> usize {
        self.chunk.code.len()
    }

    fn emit_jump(&mut self, op: OpCode) -> usize {
        self.emit_op(op);
        self.emit_u16(u16::MAX);
        self.chunk.code.len() - 2
    }

    fn patch_jump(&mut self, jump_operand_offset: usize) {
        let jump = self.chunk.code.len() - jump_operand_offset - 2;
        assert!(
            jump <= u16::MAX as usize,
            "jump offset exceeds u16 operand capacity"
        );
        let bytes = (jump as u16).to_le_bytes();
        self.chunk.code[jump_operand_offset] = bytes[0];
        self.chunk.code[jump_operand_offset + 1] = bytes[1];
    }

    fn emit_loop(&mut self, loop_start: usize) {
        self.emit_op(OpCode::Loop);
        let offset = self.chunk.code.len() - loop_start + 2;
        assert!(
            offset <= u16::MAX as usize,
            "loop offset exceeds u16 operand capacity"
        );
        self.emit_u16(offset as u16);
    }
}
