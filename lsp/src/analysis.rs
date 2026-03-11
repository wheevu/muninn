use std::collections::{HashMap, HashSet};

use muninn::ast::{
    AssignTarget, BlockExpr, ClassDecl, Expr, FunctionDecl, Program, Stmt, TypeExpr,
};
use muninn::span::Span;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DefKind {
    Variable,
    Parameter,
    Function,
    Class,
    Field,
    Method,
    Builtin,
}

#[derive(Debug, Clone)]
pub struct SymbolDef {
    pub id: usize,
    pub name: String,
    pub kind: DefKind,
    pub span: Span,
    pub detail: String,
    pub container: Option<usize>,
    pub class_hint: Option<String>,
}

#[derive(Debug, Clone)]
pub struct SymbolRef {
    pub span: Span,
    pub target: Option<usize>,
}

#[derive(Debug, Clone, Default)]
pub struct SymbolIndex {
    pub defs: Vec<SymbolDef>,
    pub refs: Vec<SymbolRef>,
    pub class_members: HashMap<String, HashMap<String, usize>>,
    def_order: Vec<usize>,
    ref_order: Vec<usize>,
}

#[derive(Debug, Clone)]
enum Binding {
    Def(usize),
    SelfType { class_name: String, class_id: usize },
}

struct Builder {
    defs: Vec<SymbolDef>,
    refs: Vec<SymbolRef>,
    scopes: Vec<HashMap<String, Binding>>,
    class_members: HashMap<String, HashMap<String, usize>>,
    class_ids: HashMap<String, usize>,
    known_classes: HashSet<String>,
    current_class: Option<String>,
}

impl SymbolIndex {
    pub fn build(program: &Program) -> Self {
        let mut builder = Builder::new(program);
        builder.index_program(program);
        builder.finish()
    }

    pub fn symbol_by_id(&self, id: usize) -> Option<&SymbolDef> {
        self.defs.get(id)
    }

    pub fn definition_at_offset(&self, offset: usize) -> Option<&SymbolDef> {
        if let Some(reference) = self.reference_at_offset(offset)
            && let Some(target) = reference.target
        {
            return self.defs.get(target);
        }

        self.symbol_at_offset(offset)
    }

    pub fn reference_at_offset(&self, offset: usize) -> Option<&SymbolRef> {
        self.best_match_ref(offset)
            .and_then(|index| self.refs.get(index))
    }

    pub fn symbol_at_offset(&self, offset: usize) -> Option<&SymbolDef> {
        self.best_match_def(offset)
            .and_then(|index| self.defs.get(index))
    }

    pub fn visible_symbols_before(&self, offset: usize) -> Vec<&SymbolDef> {
        let mut best_by_name = HashMap::<&str, &SymbolDef>::new();
        for symbol in &self.defs {
            if symbol.kind == DefKind::Field || symbol.kind == DefKind::Method {
                continue;
            }
            if symbol.span.offset > offset {
                continue;
            }

            match best_by_name.get(symbol.name.as_str()) {
                Some(existing) if existing.span.offset >= symbol.span.offset => {}
                _ => {
                    best_by_name.insert(symbol.name.as_str(), symbol);
                }
            }
        }

        let mut items = best_by_name.into_values().collect::<Vec<_>>();
        items.sort_by(|a, b| a.name.cmp(&b.name));
        items
    }

    pub fn resolve_member_chain(&self, chain: &str, offset: usize) -> Option<Vec<&SymbolDef>> {
        let mut parts = chain.split('.').collect::<Vec<_>>();
        if parts.is_empty() {
            return None;
        }

        let head = parts.remove(0);
        let mut class_name = self.class_for_name_before(head, offset)?;

        for segment in parts {
            let members = self.class_members.get(&class_name)?;
            let id = *members.get(segment)?;
            let symbol = self.defs.get(id)?;
            class_name = symbol.class_hint.clone()?;
        }

        let members = self.class_members.get(&class_name)?;
        let mut items = members
            .values()
            .filter_map(|id| self.defs.get(*id))
            .collect::<Vec<_>>();
        items.sort_by(|a, b| {
            (a.kind as u8)
                .cmp(&(b.kind as u8))
                .then_with(|| a.name.cmp(&b.name))
        });
        Some(items)
    }

    pub fn class_for_name_before(&self, name: &str, offset: usize) -> Option<String> {
        let symbol = self
            .defs
            .iter()
            .filter(|symbol| {
                symbol.name == name
                    && symbol.span.offset <= offset
                    && matches!(
                        symbol.kind,
                        DefKind::Variable
                            | DefKind::Parameter
                            | DefKind::Function
                            | DefKind::Class
                            | DefKind::Builtin
                    )
            })
            .max_by_key(|symbol| symbol.span.offset)?;

        if symbol.kind == DefKind::Class {
            Some(symbol.name.clone())
        } else {
            symbol.class_hint.clone()
        }
    }

    pub fn references_for_target(&self, target: usize, include_decl: bool) -> Vec<Span> {
        let mut spans = Vec::<Span>::new();
        if include_decl && let Some(symbol) = self.defs.get(target) {
            spans.push(symbol.span);
        }

        for reference in &self.refs {
            if reference.target == Some(target) {
                spans.push(reference.span);
            }
        }

        spans
    }

    pub fn search_symbols(&self, query: &str) -> Vec<&SymbolDef> {
        let needle = query.trim().to_ascii_lowercase();
        let mut matches = self
            .defs
            .iter()
            .filter(|symbol| symbol.span.line != 0)
            .filter(|symbol| {
                needle.is_empty() || symbol.name.to_ascii_lowercase().contains(needle.as_str())
            })
            .collect::<Vec<_>>();

        matches.sort_by(|a, b| a.name.cmp(&b.name));
        matches
    }

    fn best_match_def(&self, offset: usize) -> Option<usize> {
        best_match(
            offset,
            &self.def_order,
            |idx| self.defs[idx].span.offset,
            |idx| {
                self.defs[idx]
                    .span
                    .end_offset
                    .max(self.defs[idx].span.offset + 1)
            },
        )
    }

    fn best_match_ref(&self, offset: usize) -> Option<usize> {
        best_match(
            offset,
            &self.ref_order,
            |idx| self.refs[idx].span.offset,
            |idx| {
                self.refs[idx]
                    .span
                    .end_offset
                    .max(self.refs[idx].span.offset + 1)
            },
        )
    }
}

impl Builder {
    fn new(program: &Program) -> Self {
        let mut known_classes = HashSet::<String>::new();
        for stmt in &program.statements {
            if let Stmt::Class(class) = stmt {
                known_classes.insert(class.name.clone());
            }
        }

        let mut builder = Self {
            defs: Vec::new(),
            refs: Vec::new(),
            scopes: vec![HashMap::new()],
            class_members: HashMap::new(),
            class_ids: HashMap::new(),
            known_classes,
            current_class: None,
        };

        builder.install_builtins();
        builder.predeclare_top_level(program);
        builder
    }

    fn finish(self) -> SymbolIndex {
        let mut def_order = (0..self.defs.len()).collect::<Vec<_>>();
        def_order.sort_by_key(|idx| self.defs[*idx].span.offset);
        let mut ref_order = (0..self.refs.len()).collect::<Vec<_>>();
        ref_order.sort_by_key(|idx| self.refs[*idx].span.offset);

        SymbolIndex {
            defs: self.defs,
            refs: self.refs,
            class_members: self.class_members,
            def_order,
            ref_order,
        }
    }

    fn index_program(&mut self, program: &Program) {
        for stmt in &program.statements {
            self.visit_stmt(stmt, None);
        }
    }

    fn predeclare_top_level(&mut self, program: &Program) {
        for stmt in &program.statements {
            match stmt {
                Stmt::Function(function) => {
                    let detail = format_function_signature(function);
                    self.ensure_global_symbol(
                        function.name.clone(),
                        DefKind::Function,
                        function.span,
                        detail,
                        class_hint_from_type_expr(&function.return_type),
                    );
                }
                Stmt::Class(class) => {
                    let id = self.ensure_global_symbol(
                        class.name.clone(),
                        DefKind::Class,
                        class.span,
                        format!("class {}", class.name),
                        Some(class.name.clone()),
                    );
                    self.class_ids.insert(class.name.clone(), id);
                }
                _ => {}
            }
        }
    }

    fn install_builtins(&mut self) {
        let builtins = [
            ("to_string", "fn to_string(value) -> String", None),
            ("print", "fn print(value) -> Void", None),
            ("len", "fn len(value) -> Int", None),
            ("sum", "fn sum(array) -> Number", None),
            ("dot", "fn dot(left, right) -> Number", None),
            ("zeros", "fn zeros(length: Int) -> Float[length]", None),
            ("ones", "fn ones(length: Int) -> Float[length]", None),
            ("none", "none: Option[T]", None),
            ("some", "fn some(value: T) -> Option[T]", None),
            ("is_none", "fn is_none(value: Option[T]) -> Bool", None),
            ("unwrap", "fn unwrap(value: Option[T]) -> T", None),
            ("__none", "legacy alias for none", None),
            ("__some", "legacy alias for some", None),
            ("__is_none", "legacy alias for is_none", None),
            ("__unwrap", "legacy alias for unwrap", None),
        ];

        for (name, detail, class_hint) in builtins {
            self.ensure_global_symbol(
                name.to_string(),
                DefKind::Builtin,
                Span::default(),
                detail.to_string(),
                class_hint.map(str::to_string),
            );
        }
    }

    fn visit_stmt(&mut self, stmt: &Stmt, container: Option<usize>) {
        match stmt {
            Stmt::Let {
                name,
                ty,
                initializer,
                span,
                ..
            } => {
                self.visit_expr(initializer);
                let detail = ty
                    .as_ref()
                    .map(format_type_expr)
                    .or_else(|| infer_type_detail(initializer))
                    .unwrap_or_else(|| "unknown".to_string());
                let class_hint = ty.as_ref().and_then(class_hint_from_type_expr).or_else(|| {
                    infer_expr_class_from_initializer(initializer, &self.known_classes)
                });
                self.define_symbol(
                    name.clone(),
                    DefKind::Variable,
                    *span,
                    detail,
                    container,
                    class_hint,
                );
            }
            Stmt::Function(function) => {
                let id = self
                    .lookup_in_current_scope(&function.name)
                    .or_else(|| self.lookup_global(&function.name))
                    .unwrap_or_else(|| {
                        self.define_symbol(
                            function.name.clone(),
                            DefKind::Function,
                            function.span,
                            format_function_signature(function),
                            container,
                            class_hint_from_type_expr(&function.return_type),
                        )
                    });
                self.visit_function(function, id, None);
            }
            Stmt::Class(class) => {
                let class_id = self
                    .lookup_in_current_scope(&class.name)
                    .or_else(|| self.lookup_global(&class.name))
                    .unwrap_or_else(|| {
                        self.define_symbol(
                            class.name.clone(),
                            DefKind::Class,
                            class.span,
                            format!("class {}", class.name),
                            container,
                            Some(class.name.clone()),
                        )
                    });
                self.class_ids.insert(class.name.clone(), class_id);
                self.visit_class(class, class_id);
            }
            Stmt::Return { value, .. } => {
                if let Some(value) = value {
                    self.visit_expr(value);
                }
            }
            Stmt::While {
                condition, body, ..
            } => {
                self.visit_expr(condition);
                self.visit_block(body, container);
            }
            Stmt::ForRange {
                start, end, body, ..
            } => {
                self.visit_expr(start);
                self.visit_expr(end);
                self.visit_block(body, container);
            }
            Stmt::Expression { expr, .. } => self.visit_expr(expr),
        }
    }

    fn visit_function(
        &mut self,
        function: &FunctionDecl,
        function_id: usize,
        receiver: Option<&ClassDecl>,
    ) {
        self.push_scope();

        if let Some(class) = receiver {
            self.insert_binding(
                "self".to_string(),
                Binding::SelfType {
                    class_name: class.name.clone(),
                    class_id: *self.class_ids.get(&class.name).unwrap_or(&function_id),
                },
            );
        }

        for param in &function.params {
            let id = self.define_symbol(
                param.name.clone(),
                DefKind::Parameter,
                param.span,
                format_type_expr(&param.ty),
                Some(function_id),
                class_hint_from_type_expr(&param.ty),
            );
            self.insert_binding(param.name.clone(), Binding::Def(id));
        }

        self.visit_block(&function.body, Some(function_id));
        self.pop_scope();
    }

    fn visit_class(&mut self, class: &ClassDecl, class_id: usize) {
        self.class_members.entry(class.name.clone()).or_default();

        self.current_class = Some(class.name.clone());

        for field in &class.fields {
            let id = self.define_symbol(
                field.name.clone(),
                DefKind::Field,
                field.span,
                format_type_expr(&field.ty),
                Some(class_id),
                class_hint_from_type_expr(&field.ty),
            );
            self.class_members
                .entry(class.name.clone())
                .or_default()
                .insert(field.name.clone(), id);
        }

        for method in &class.methods {
            let method_id = self.define_symbol(
                method.name.clone(),
                DefKind::Method,
                method.span,
                format_function_signature(method),
                Some(class_id),
                class_hint_from_type_expr(&method.return_type),
            );
            self.class_members
                .entry(class.name.clone())
                .or_default()
                .insert(method.name.clone(), method_id);
            self.visit_function(method, method_id, Some(class));
        }

        if let Some(init) = &class.init {
            let init_id = self.define_symbol(
                init.name.clone(),
                DefKind::Method,
                init.span,
                format_function_signature(init),
                Some(class_id),
                Some(class.name.clone()),
            );
            self.class_members
                .entry(class.name.clone())
                .or_default()
                .insert(init.name.clone(), init_id);
            self.visit_function(init, init_id, Some(class));
        }

        self.current_class = None;
    }

    fn visit_block(&mut self, block: &BlockExpr, container: Option<usize>) {
        self.push_scope();
        for stmt in &block.statements {
            self.visit_stmt(stmt, container);
        }
        if let Some(tail) = &block.tail {
            self.visit_expr(tail);
        }
        self.pop_scope();
    }

    fn visit_expr(&mut self, expr: &Expr) {
        match expr {
            Expr::Int(..) | Expr::Float(..) | Expr::Bool(..) | Expr::String(..) => {}
            Expr::Variable(name, span) => {
                let target = self.lookup_binding(name).and_then(|binding| match binding {
                    Binding::Def(id) => Some(*id),
                    Binding::SelfType { class_id, .. } => Some(*class_id),
                });
                self.refs.push(SymbolRef {
                    span: *span,
                    target,
                });
            }
            Expr::SelfRef(span) => {
                let target = self
                    .lookup_binding("self")
                    .and_then(|binding| match binding {
                        Binding::SelfType { class_id, .. } => Some(*class_id),
                        Binding::Def(id) => Some(*id),
                    });
                self.refs.push(SymbolRef {
                    span: *span,
                    target,
                });
            }
            Expr::ArrayLiteral(items, _) => {
                for item in items {
                    self.visit_expr(item);
                }
            }
            Expr::Block(block) => self.visit_block(block, None),
            Expr::Grouping(inner, _) => self.visit_expr(inner),
            Expr::Unary { expr, .. } => self.visit_expr(expr),
            Expr::Binary { left, right, .. } | Expr::VecBinary { left, right, .. } => {
                self.visit_expr(left);
                self.visit_expr(right);
            }
            Expr::If {
                condition,
                then_branch,
                else_branch,
                ..
            } => {
                self.visit_expr(condition);
                self.visit_block(then_branch, None);
                self.visit_block(else_branch, None);
            }
            Expr::Unless {
                condition,
                then_branch,
                else_branch,
                ..
            } => {
                self.visit_expr(condition);
                self.visit_block(then_branch, None);
                if let Some(else_branch) = else_branch {
                    self.visit_block(else_branch, None);
                }
            }
            Expr::Call { callee, args, .. } => {
                self.visit_expr(callee);
                for arg in args {
                    self.visit_expr(arg);
                }
            }
            Expr::Pipeline {
                lhs, callee, args, ..
            } => {
                self.visit_expr(lhs);
                self.visit_expr(callee);
                for arg in args {
                    self.visit_expr(arg);
                }
            }
            Expr::Property { object, name, span } => {
                self.visit_expr(object);
                let target = self
                    .infer_expr_class(object)
                    .and_then(|class_name| self.class_members.get(&class_name)?.get(name).copied());
                self.refs.push(SymbolRef {
                    span: *span,
                    target,
                });
            }
            Expr::Index { target, index, .. } => {
                self.visit_expr(target);
                self.visit_expr(index);
            }
            Expr::GridIndex { target, x, y, .. } => {
                self.visit_expr(target);
                self.visit_expr(x);
                self.visit_expr(y);
            }
            Expr::Assign { target, value, .. } => {
                self.visit_assign_target(target);
                self.visit_expr(value);
            }
            Expr::Try { expr, .. } => self.visit_expr(expr),
            Expr::StringInterpolation { parts, .. } => {
                for part in parts {
                    if let muninn::ast::InterpolationPart::Expr(expr) = part {
                        self.visit_expr(expr);
                    }
                }
            }
        }
    }

    fn visit_assign_target(&mut self, target: &AssignTarget) {
        match target {
            AssignTarget::Variable(name, span) => {
                let target = self.lookup_binding(name).and_then(|binding| match binding {
                    Binding::Def(id) => Some(*id),
                    Binding::SelfType { class_id, .. } => Some(*class_id),
                });
                self.refs.push(SymbolRef {
                    span: *span,
                    target,
                });
            }
            AssignTarget::Property { object, name, span } => {
                self.visit_expr(object);
                let target = self
                    .infer_expr_class(object)
                    .and_then(|class_name| self.class_members.get(&class_name)?.get(name).copied());
                self.refs.push(SymbolRef {
                    span: *span,
                    target,
                });
            }
            AssignTarget::Index { target, index, .. } => {
                self.visit_expr(target);
                self.visit_expr(index);
            }
            AssignTarget::GridIndex { target, x, y, .. } => {
                self.visit_expr(target);
                self.visit_expr(x);
                self.visit_expr(y);
            }
        }
    }

    fn infer_expr_class(&self, expr: &Expr) -> Option<String> {
        match expr {
            Expr::SelfRef(_) => self.current_class.clone(),
            Expr::Variable(name, _) => self.class_for_binding_name(name),
            Expr::Call { callee, .. } => {
                if let Expr::Variable(name, _) = callee.as_ref()
                    && self.known_classes.contains(name)
                {
                    return Some(name.clone());
                }
                None
            }
            Expr::Property { object, name, .. } => {
                let class_name = self.infer_expr_class(object)?;
                let members = self.class_members.get(&class_name)?;
                let member_id = *members.get(name)?;
                let member = self.defs.get(member_id)?;
                member.class_hint.clone()
            }
            Expr::Grouping(inner, _) => self.infer_expr_class(inner),
            Expr::Try { expr, .. } => self.infer_expr_class(expr),
            _ => None,
        }
    }

    fn class_for_binding_name(&self, name: &str) -> Option<String> {
        let binding = self.lookup_binding(name)?;
        match binding {
            Binding::SelfType { class_name, .. } => Some(class_name.clone()),
            Binding::Def(id) => {
                let def = self.defs.get(*id)?;
                if def.kind == DefKind::Class {
                    Some(def.name.clone())
                } else {
                    def.class_hint.clone()
                }
            }
        }
    }

    fn define_symbol(
        &mut self,
        name: String,
        kind: DefKind,
        span: Span,
        detail: String,
        container: Option<usize>,
        class_hint: Option<String>,
    ) -> usize {
        let id = self.defs.len();
        self.defs.push(SymbolDef {
            id,
            name: name.clone(),
            kind,
            span,
            detail,
            container,
            class_hint,
        });
        self.insert_binding(name, Binding::Def(id));
        id
    }

    fn ensure_global_symbol(
        &mut self,
        name: String,
        kind: DefKind,
        span: Span,
        detail: String,
        class_hint: Option<String>,
    ) -> usize {
        if let Some(id) = self.lookup_global(&name) {
            return id;
        }

        let id = self.defs.len();
        self.defs.push(SymbolDef {
            id,
            name: name.clone(),
            kind,
            span,
            detail,
            container: None,
            class_hint,
        });
        if let Some(global) = self.scopes.first_mut() {
            global.insert(name, Binding::Def(id));
        }
        id
    }

    fn insert_binding(&mut self, name: String, binding: Binding) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(name, binding);
        }
    }

    fn lookup_binding(&self, name: &str) -> Option<&Binding> {
        for scope in self.scopes.iter().rev() {
            if let Some(binding) = scope.get(name) {
                return Some(binding);
            }
        }
        None
    }

    fn lookup_in_current_scope(&self, name: &str) -> Option<usize> {
        let scope = self.scopes.last()?;
        match scope.get(name) {
            Some(Binding::Def(id)) => Some(*id),
            _ => None,
        }
    }

    fn lookup_global(&self, name: &str) -> Option<usize> {
        let scope = self.scopes.first()?;
        match scope.get(name) {
            Some(Binding::Def(id)) => Some(*id),
            _ => None,
        }
    }

    fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    fn pop_scope(&mut self) {
        self.scopes.pop();
    }
}

fn best_match<Start, End>(offset: usize, order: &[usize], start: Start, end: End) -> Option<usize>
where
    Start: Fn(usize) -> usize,
    End: Fn(usize) -> usize,
{
    if order.is_empty() {
        return None;
    }

    let mut left = 0usize;
    let mut right = order.len();
    while left < right {
        let mid = (left + right) / 2;
        if start(order[mid]) <= offset {
            left = mid + 1;
        } else {
            right = mid;
        }
    }

    if left == 0 {
        return None;
    }

    let mut best: Option<usize> = None;
    for idx in order[..left].iter().rev().copied() {
        let s = start(idx);
        if s > offset {
            continue;
        }
        let e = end(idx);
        if offset < s || offset >= e {
            if best.is_some() {
                break;
            }
            continue;
        }

        match best {
            Some(current) => {
                let current_width = end(current).saturating_sub(start(current));
                let width = e.saturating_sub(s);
                if width <= current_width {
                    best = Some(idx);
                }
            }
            None => best = Some(idx),
        }
    }

    best
}

fn format_type_expr(ty: &TypeExpr) -> String {
    match ty {
        TypeExpr::Int => "Int".to_string(),
        TypeExpr::Float => "Float".to_string(),
        TypeExpr::String => "String".to_string(),
        TypeExpr::Bool => "Bool".to_string(),
        TypeExpr::Void => "Void".to_string(),
        TypeExpr::Named(name) => name.clone(),
        TypeExpr::Option(inner) => format!("Option[{}]", format_type_expr(inner)),
        TypeExpr::Array { element, len } => format!("{}[{}]", format_type_expr(element), len),
        TypeExpr::Grid {
            element,
            width,
            height,
        } => format!("{}[{}, {}]", format_type_expr(element), width, height),
    }
}

fn class_hint_from_type_expr(ty: &TypeExpr) -> Option<String> {
    match ty {
        TypeExpr::Named(name) => Some(name.clone()),
        _ => None,
    }
}

fn infer_type_detail(expr: &Expr) -> Option<String> {
    match expr {
        Expr::Int(..) => Some("Int".to_string()),
        Expr::Float(..) => Some("Float".to_string()),
        Expr::Bool(..) => Some("Bool".to_string()),
        Expr::String(..) | Expr::StringInterpolation { .. } => Some("String".to_string()),
        Expr::ArrayLiteral(items, _) => {
            let len = items.len();
            let element = items.first().and_then(infer_type_detail)?;
            Some(format!("{}[{}]", element, len))
        }
        Expr::Call { callee, .. } => {
            if let Expr::Variable(name, _) = callee.as_ref() {
                Some(name.clone())
            } else {
                None
            }
        }
        _ => None,
    }
}

fn infer_expr_class_from_initializer(
    expr: &Expr,
    known_classes: &HashSet<String>,
) -> Option<String> {
    match expr {
        Expr::Call { callee, .. } => {
            if let Expr::Variable(name, _) = callee.as_ref()
                && known_classes.contains(name)
            {
                return Some(name.clone());
            }
            None
        }
        _ => None,
    }
}

fn format_function_signature(function: &FunctionDecl) -> String {
    let params = function
        .params
        .iter()
        .map(|param| format!("{}: {}", param.name, format_type_expr(&param.ty)))
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "fn {}({}) -> {}",
        function.name,
        params,
        format_type_expr(&function.return_type)
    )
}

pub fn markdown_for_symbol(symbol: &SymbolDef) -> String {
    let kind = match symbol.kind {
        DefKind::Variable => "Variable",
        DefKind::Parameter => "Parameter",
        DefKind::Function => "Function",
        DefKind::Class => "Class",
        DefKind::Field => "Field",
        DefKind::Method => "Method",
        DefKind::Builtin => "Builtin",
    };

    if symbol.detail.is_empty() {
        format!("**{}** `{}`", kind, symbol.name)
    } else {
        format!(
            "**{}** `{}`\n\n```muninn\n{}\n```",
            kind, symbol.name, symbol.detail
        )
    }
}

#[cfg(test)]
mod tests {
    use muninn::parse_document;

    use super::SymbolIndex;

    #[test]
    fn resolves_definition_by_offset() {
        let program = parse_document("let x: Int = 1; let y: Int = x;").expect("program");
        let index = SymbolIndex::build(&program);

        let x_use = index
            .refs
            .iter()
            .find(|reference| reference.target.is_some())
            .expect("x reference");

        let def = index
            .definition_at_offset(x_use.span.offset)
            .expect("definition");
        assert_eq!(def.name, "x");
    }
}
