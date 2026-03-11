use crate::typecheck::Ty;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuiltinKind {
    Print,
    Assert,
}

#[derive(Debug, Clone, Copy)]
pub struct BuiltinSpec {
    pub name: &'static str,
    pub detail: &'static str,
    pub kind: BuiltinKind,
}

pub const BUILTINS: &[BuiltinSpec] = &[
    BuiltinSpec {
        name: "print",
        detail: "fn print(value: Int | Float | Bool | String) -> Void",
        kind: BuiltinKind::Print,
    },
    BuiltinSpec {
        name: "assert",
        detail: "fn assert(condition: Bool) -> Void",
        kind: BuiltinKind::Assert,
    },
];

pub fn builtin_by_name(name: &str) -> Option<&'static BuiltinSpec> {
    BUILTINS.iter().find(|builtin| builtin.name == name)
}

pub fn accepts_argument(kind: BuiltinKind, argument: &Ty) -> bool {
    match kind {
        BuiltinKind::Print => matches!(argument, Ty::Int | Ty::Float | Ty::Bool | Ty::String),
        BuiltinKind::Assert => matches!(argument, Ty::Bool),
    }
}

pub fn return_type(kind: BuiltinKind) -> Ty {
    match kind {
        BuiltinKind::Print | BuiltinKind::Assert => Ty::Void,
    }
}
