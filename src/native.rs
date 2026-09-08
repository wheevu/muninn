use std::collections::HashSet;
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::error::MuninnError;
use crate::runtime::{VmResult, vm_error};
use crate::span::Span;
use crate::tensor::{
    Tensor, arc_tensor, matmul, scalar_tensor_binary, tensor_binary, tensor_scalar_binary,
};
use crate::value::Value;

pub type NativeCallable = for<'a> fn(NativeCallContext<'a>) -> VmResult<Value>;

/// Built-in tensor element limit from `tensor.rs`; the host default.
const BUILTIN_MAX_ELEMENTS: usize = 10_000_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NativeFunctionKind {
    Print,
    Assert,
    TensorZeros,
    TensorFill,
    TensorReshape,
    TensorMatmul,
    TensorSum,
    ArgsLen,
    ArgsGet,
    EnvGet,
    EnvHas,
    FsRead,
    FsExists,
    FsWrite,
    ClockMs,
    StdinRead,
    Eprint,
    Exit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum NativeType {
    Int,
    Float,
    Bool,
    String,
    Void,
    Tensor,
    Record,
}

#[derive(Debug, Clone, Copy)]
pub struct NativeSignature {
    pub params: &'static [NativeType],
    pub return_type: NativeType,
}

#[derive(Debug, Clone, Copy)]
pub struct NativeSpec {
    pub kind: NativeFunctionKind,
    pub name: &'static str,
    pub detail: &'static str,
    pub signatures: &'static [NativeSignature],
    pub runtime: NativeCallable,
}

#[derive(Clone, Copy)]
pub struct NativeCallContext<'a> {
    args: &'a [Value],
    span: Span,
    host: HostCaps<'a>,
}

/// The host-granted capabilities visible to IO natives. Built by the VM
/// from [`crate::vm::HostPolicy`] on every call, so scripts never see
/// ambient authority: no argv, env, filesystem, or stdin exists unless
/// the host put it here. Pure natives (print, assert, tensors) ignore it.
#[derive(Debug, Clone, Copy)]
pub struct HostCaps<'a> {
    pub argv: &'a [String],
    /// `None` allows every env name (trusted CLI default); `Some(set)`
    /// allows only names in the set (sandboxed default is empty: deny all).
    pub allowed_env: &'a Option<HashSet<String>>,
    /// `None` denies all filesystem access. `Some(root)` scopes every fs
    /// builtin under `root`; absolute paths and escapes trap.
    pub fs_root: &'a Option<PathBuf>,
    pub stdin_data: &'a str,
}

impl<'a> NativeCallContext<'a> {
    pub fn new(args: &'a [Value], span: Span, host: HostCaps<'a>) -> Self {
        Self { args, span, host }
    }

    pub fn args(&self) -> &'a [Value] {
        self.args
    }

    pub fn span(&self) -> Span {
        self.span
    }

    pub fn host(&self) -> &HostCaps<'a> {
        &self.host
    }

    pub fn expect_int(&self, index: usize, name: &str) -> VmResult<i64> {
        match self.args.get(index) {
            Some(Value::Int(value)) => Ok(*value),
            Some(other) => Err(self.type_error(index, name, "Int", other)),
            None => Err(vm_error(
                format!("missing argument '{}' at position {}", name, index),
                self.span,
            )),
        }
    }

    pub fn expect_float(&self, index: usize, name: &str) -> VmResult<f64> {
        match self.args.get(index) {
            Some(Value::Float(value)) => Ok(*value),
            Some(other) => Err(self.type_error(index, name, "Float", other)),
            None => Err(vm_error(
                format!("missing argument '{}' at position {}", name, index),
                self.span,
            )),
        }
    }

    pub fn expect_bool(&self, index: usize, name: &str) -> VmResult<bool> {
        match self.args.get(index) {
            Some(Value::Bool(value)) => Ok(*value),
            Some(other) => Err(self.type_error(index, name, "Bool", other)),
            None => Err(vm_error(
                format!("missing argument '{}' at position {}", name, index),
                self.span,
            )),
        }
    }

    pub fn expect_string(&self, index: usize, name: &str) -> VmResult<String> {
        match self.args.get(index) {
            Some(Value::String(value)) => Ok(value.to_string()),
            Some(other) => Err(self.type_error(index, name, "String", other)),
            None => Err(vm_error(
                format!("missing argument '{}' at position {}", name, index),
                self.span,
            )),
        }
    }

    pub fn expect_tensor(&self, index: usize, name: &str) -> VmResult<Arc<Tensor>> {
        match self.args.get(index) {
            Some(Value::Tensor(value)) => Ok(Arc::clone(value)),
            Some(other) => Err(self.type_error(index, name, "Tensor", other)),
            None => Err(vm_error(
                format!("missing argument '{}' at position {}", name, index),
                self.span,
            )),
        }
    }

    fn type_error(&self, index: usize, name: &str, expected: &str, actual: &Value) -> MuninnError {
        vm_error(
            format!(
                "argument {} ('{}') expects {}, got {}",
                index,
                name,
                expected,
                actual.stringify()
            ),
            self.span,
        )
    }
}

const PRINT_SIGNATURES: &[NativeSignature] = &[
    NativeSignature {
        params: &[NativeType::Int],
        return_type: NativeType::Void,
    },
    NativeSignature {
        params: &[NativeType::Float],
        return_type: NativeType::Void,
    },
    NativeSignature {
        params: &[NativeType::Bool],
        return_type: NativeType::Void,
    },
    NativeSignature {
        params: &[NativeType::String],
        return_type: NativeType::Void,
    },
    NativeSignature {
        params: &[NativeType::Tensor],
        return_type: NativeType::Void,
    },
    NativeSignature {
        params: &[NativeType::Record],
        return_type: NativeType::Void,
    },
];

const ASSERT_SIGNATURES: &[NativeSignature] = &[NativeSignature {
    params: &[NativeType::Bool],
    return_type: NativeType::Void,
}];

const TENSOR_ZEROS_SIGNATURES: &[NativeSignature] = &[
    NativeSignature {
        params: &[NativeType::Int],
        return_type: NativeType::Tensor,
    },
    NativeSignature {
        params: &[NativeType::Int, NativeType::Int],
        return_type: NativeType::Tensor,
    },
];

const TENSOR_FILL_SIGNATURES: &[NativeSignature] = &[
    NativeSignature {
        params: &[NativeType::Int, NativeType::Float],
        return_type: NativeType::Tensor,
    },
    NativeSignature {
        params: &[NativeType::Int, NativeType::Int, NativeType::Float],
        return_type: NativeType::Tensor,
    },
];

const TENSOR_RESHAPE_SIGNATURES: &[NativeSignature] = &[
    NativeSignature {
        params: &[NativeType::Tensor, NativeType::Int],
        return_type: NativeType::Tensor,
    },
    NativeSignature {
        params: &[NativeType::Tensor, NativeType::Int, NativeType::Int],
        return_type: NativeType::Tensor,
    },
];

const TENSOR_MATMUL_SIGNATURES: &[NativeSignature] = &[NativeSignature {
    params: &[NativeType::Tensor, NativeType::Tensor],
    return_type: NativeType::Tensor,
}];

const TENSOR_SUM_SIGNATURES: &[NativeSignature] = &[NativeSignature {
    params: &[NativeType::Tensor],
    return_type: NativeType::Float,
}];

const ARGS_LEN_SIGNATURES: &[NativeSignature] = &[NativeSignature {
    params: &[],
    return_type: NativeType::Int,
}];

const ARGS_GET_SIGNATURES: &[NativeSignature] = &[NativeSignature {
    params: &[NativeType::Int],
    return_type: NativeType::String,
}];

const ENV_GET_SIGNATURES: &[NativeSignature] = &[NativeSignature {
    params: &[NativeType::String],
    return_type: NativeType::String,
}];

const ENV_HAS_SIGNATURES: &[NativeSignature] = &[NativeSignature {
    params: &[NativeType::String],
    return_type: NativeType::Bool,
}];

const FS_READ_SIGNATURES: &[NativeSignature] = &[NativeSignature {
    params: &[NativeType::String],
    return_type: NativeType::String,
}];

const FS_EXISTS_SIGNATURES: &[NativeSignature] = &[NativeSignature {
    params: &[NativeType::String],
    return_type: NativeType::Bool,
}];

const FS_WRITE_SIGNATURES: &[NativeSignature] = &[NativeSignature {
    params: &[NativeType::String, NativeType::String],
    return_type: NativeType::Void,
}];

const CLOCK_MS_SIGNATURES: &[NativeSignature] = &[NativeSignature {
    params: &[],
    return_type: NativeType::Int,
}];

const STDIN_READ_SIGNATURES: &[NativeSignature] = &[NativeSignature {
    params: &[],
    return_type: NativeType::String,
}];

const EPRINT_SIGNATURES: &[NativeSignature] = &[
    NativeSignature {
        params: &[NativeType::Int],
        return_type: NativeType::Void,
    },
    NativeSignature {
        params: &[NativeType::Float],
        return_type: NativeType::Void,
    },
    NativeSignature {
        params: &[NativeType::Bool],
        return_type: NativeType::Void,
    },
    NativeSignature {
        params: &[NativeType::String],
        return_type: NativeType::Void,
    },
    NativeSignature {
        params: &[NativeType::Tensor],
        return_type: NativeType::Void,
    },
    NativeSignature {
        params: &[NativeType::Record],
        return_type: NativeType::Void,
    },
];

const EXIT_SIGNATURES: &[NativeSignature] = &[NativeSignature {
    params: &[NativeType::Int],
    return_type: NativeType::Void,
}];

static NATIVE_SPECS: &[NativeSpec] = &[
    NativeSpec {
        kind: NativeFunctionKind::Print,
        name: "print",
        detail: "fn print(value: Int | Float | Bool | String | Tensor | Record) -> Void",
        signatures: PRINT_SIGNATURES,
        runtime: native_print,
    },
    NativeSpec {
        kind: NativeFunctionKind::Assert,
        name: "assert",
        detail: "fn assert(condition: Bool) -> Void",
        signatures: ASSERT_SIGNATURES,
        runtime: native_assert,
    },
    NativeSpec {
        kind: NativeFunctionKind::TensorZeros,
        name: "tensor_zeros",
        detail: "fn tensor_zeros(size: Int) -> Tensor | fn tensor_zeros(rows: Int, cols: Int) -> Tensor",
        signatures: TENSOR_ZEROS_SIGNATURES,
        runtime: native_tensor_zeros,
    },
    NativeSpec {
        kind: NativeFunctionKind::TensorFill,
        name: "tensor_fill",
        detail: "fn tensor_fill(size: Int, value: Float) -> Tensor | fn tensor_fill(rows: Int, cols: Int, value: Float) -> Tensor",
        signatures: TENSOR_FILL_SIGNATURES,
        runtime: native_tensor_fill,
    },
    NativeSpec {
        kind: NativeFunctionKind::TensorReshape,
        name: "tensor_reshape",
        detail: "fn tensor_reshape(tensor: Tensor, size: Int) -> Tensor | fn tensor_reshape(tensor: Tensor, rows: Int, cols: Int) -> Tensor",
        signatures: TENSOR_RESHAPE_SIGNATURES,
        runtime: native_tensor_reshape,
    },
    NativeSpec {
        kind: NativeFunctionKind::TensorMatmul,
        name: "tensor_matmul",
        detail: "fn tensor_matmul(left: Tensor, right: Tensor) -> Tensor",
        signatures: TENSOR_MATMUL_SIGNATURES,
        runtime: native_tensor_matmul,
    },
    NativeSpec {
        kind: NativeFunctionKind::TensorSum,
        name: "tensor_sum",
        detail: "fn tensor_sum(tensor: Tensor) -> Float",
        signatures: TENSOR_SUM_SIGNATURES,
        runtime: native_tensor_sum,
    },
    NativeSpec {
        kind: NativeFunctionKind::ArgsLen,
        name: "args_len",
        detail: "fn args_len() -> Int",
        signatures: ARGS_LEN_SIGNATURES,
        runtime: native_args_len,
    },
    NativeSpec {
        kind: NativeFunctionKind::ArgsGet,
        name: "args_get",
        detail: "fn args_get(index: Int) -> String",
        signatures: ARGS_GET_SIGNATURES,
        runtime: native_args_get,
    },
    NativeSpec {
        kind: NativeFunctionKind::EnvGet,
        name: "env_get",
        detail: "fn env_get(name: String) -> String",
        signatures: ENV_GET_SIGNATURES,
        runtime: native_env_get,
    },
    NativeSpec {
        kind: NativeFunctionKind::EnvHas,
        name: "env_has",
        detail: "fn env_has(name: String) -> Bool",
        signatures: ENV_HAS_SIGNATURES,
        runtime: native_env_has,
    },
    NativeSpec {
        kind: NativeFunctionKind::FsRead,
        name: "fs_read",
        detail: "fn fs_read(path: String) -> String",
        signatures: FS_READ_SIGNATURES,
        runtime: native_fs_read,
    },
    NativeSpec {
        kind: NativeFunctionKind::FsExists,
        name: "fs_exists",
        detail: "fn fs_exists(path: String) -> Bool",
        signatures: FS_EXISTS_SIGNATURES,
        runtime: native_fs_exists,
    },
    NativeSpec {
        kind: NativeFunctionKind::FsWrite,
        name: "fs_write",
        detail: "fn fs_write(path: String, data: String) -> Void",
        signatures: FS_WRITE_SIGNATURES,
        runtime: native_fs_write,
    },
    NativeSpec {
        kind: NativeFunctionKind::ClockMs,
        name: "clock_ms",
        detail: "fn clock_ms() -> Int",
        signatures: CLOCK_MS_SIGNATURES,
        runtime: native_clock_ms,
    },
    NativeSpec {
        kind: NativeFunctionKind::StdinRead,
        name: "stdin_read",
        detail: "fn stdin_read() -> String",
        signatures: STDIN_READ_SIGNATURES,
        runtime: native_stdin_read,
    },
    NativeSpec {
        kind: NativeFunctionKind::Eprint,
        name: "eprint",
        detail: "fn eprint(value: Int | Float | Bool | String | Tensor | Record) -> Void",
        signatures: EPRINT_SIGNATURES,
        runtime: native_eprint,
    },
    NativeSpec {
        kind: NativeFunctionKind::Exit,
        name: "exit",
        detail: "fn exit(code: Int) -> Void",
        signatures: EXIT_SIGNATURES,
        runtime: native_exit,
    },
];

pub fn registered_natives() -> &'static [NativeSpec] {
    NATIVE_SPECS
}

pub fn native_by_name(name: &str) -> Option<&'static NativeSpec> {
    NATIVE_SPECS.iter().find(|spec| spec.name == name)
}

pub fn native_by_kind(kind: NativeFunctionKind) -> Option<&'static NativeSpec> {
    NATIVE_SPECS.iter().find(|spec| spec.kind == kind)
}

/// Display name of a native builtin, used in host-policy diagnostics.
pub fn native_name(kind: NativeFunctionKind) -> &'static str {
    native_by_kind(kind)
        .map(|spec| spec.name)
        .unwrap_or("<unknown native>")
}

pub fn invoke_native(kind: NativeFunctionKind, args: &[Value], span: Span) -> VmResult<Value> {
    // Legacy pure path: no host capabilities granted. IO natives trap
    // with a policy diagnostic instead of seeing ambient authority.
    let empty_argv: &[String] = &[];
    let deny_env: Option<HashSet<String>> = Some(HashSet::new());
    let no_root: Option<PathBuf> = None;
    let caps = HostCaps {
        argv: empty_argv,
        allowed_env: &deny_env,
        fs_root: &no_root,
        stdin_data: "",
    };
    invoke_native_with_limit(kind, args, span, BUILTIN_MAX_ELEMENTS, &caps)
}

/// Dispatches a native builtin with a host-enforced tensor element cap.
///
/// Shapes for constructors (`tensor_zeros`, `tensor_fill`) are validated
/// before allocation so a hostile shape traps without allocating. Caps at
/// or above the built-in 10M limit delegate to [`invoke_native`], whose
/// existing `element_count` path enforces the built-in limit.
pub fn invoke_native_with_limit(
    kind: NativeFunctionKind,
    args: &[Value],
    span: Span,
    max_elements: usize,
    host: &HostCaps,
) -> VmResult<Value> {
    if max_elements < BUILTIN_MAX_ELEMENTS {
        check_shape_limit(kind, args, span, max_elements)?;
    }
    let spec = native_by_kind(kind)
        .ok_or_else(|| vm_error(format!("unknown native function {:?}", kind), span))?;
    let result = (spec.runtime)(NativeCallContext::new(args, span, *host))?;
    if let Value::Tensor(tensor) = &result
        && tensor.data().len() > max_elements
    {
        return Err(vm_error(
            format!(
                "tensor has {} elements, maximum is {} for this host",
                tensor.data().len(),
                max_elements
            ),
            span,
        ));
    }
    Ok(result)
}

/// Pre-allocation shape check for tensor constructors.
fn check_shape_limit(
    kind: NativeFunctionKind,
    args: &[Value],
    span: Span,
    max_elements: usize,
) -> VmResult<()> {
    let dims: Vec<i64> = match kind {
        NativeFunctionKind::TensorZeros => match args {
            [Value::Int(a)] => vec![*a],
            [Value::Int(a), Value::Int(b)] => vec![*a, *b],
            _ => return Ok(()),
        },
        NativeFunctionKind::TensorFill => match args {
            [Value::Int(a), Value::Float(_)] => vec![*a],
            [Value::Int(a), Value::Int(b), Value::Float(_)] => vec![*a, *b],
            _ => return Ok(()),
        },
        _ => return Ok(()),
    };
    let mut count: usize = 1;
    for dim in dims {
        if dim <= 0 {
            return Ok(());
        }
        count = count
            .checked_mul(dim as usize)
            .ok_or_else(|| vm_error("tensor shape is too large", span))?;
        if count > max_elements {
            return Err(vm_error(
                format!(
                    "tensor has at least {} elements, maximum is {} for this host",
                    count, max_elements
                ),
                span,
            ));
        }
    }
    Ok(())
}

fn native_print(ctx: NativeCallContext<'_>) -> VmResult<Value> {
    if ctx.args().len() != 1 {
        return Err(vm_error("print expects exactly 1 argument", ctx.span()));
    }
    println!("{}", ctx.args()[0]);
    Ok(Value::Nil)
}

fn native_assert(ctx: NativeCallContext<'_>) -> VmResult<Value> {
    if ctx.args().len() != 1 {
        return Err(vm_error("assert expects exactly 1 argument", ctx.span()));
    }
    match ctx.expect_bool(0, "condition")? {
        true => Ok(Value::Nil),
        false => Err(vm_error("assertion failed", ctx.span())),
    }
}

fn native_tensor_zeros(ctx: NativeCallContext<'_>) -> VmResult<Value> {
    let shape = expect_shape(ctx)?;
    Ok(Value::Tensor(arc_tensor(Tensor::zeros(shape, ctx.span())?)))
}

fn native_tensor_fill(ctx: NativeCallContext<'_>) -> VmResult<Value> {
    let (shape, fill_index) = expect_shape_with_tail(ctx)?;
    let value = ctx.expect_float(fill_index, "value")?;
    Ok(Value::Tensor(arc_tensor(Tensor::filled(
        shape,
        value,
        ctx.span(),
    )?)))
}

fn native_tensor_reshape(ctx: NativeCallContext<'_>) -> VmResult<Value> {
    if !(ctx.args().len() == 2 || ctx.args().len() == 3) {
        return Err(vm_error(
            "tensor_reshape expects 2 or 3 arguments",
            ctx.span(),
        ));
    }
    let tensor = ctx.expect_tensor(0, "tensor")?;
    let shape = if ctx.args().len() == 2 {
        vec![positive_dim(ctx.expect_int(1, "size")?, ctx.span())?]
    } else {
        vec![
            positive_dim(ctx.expect_int(1, "rows")?, ctx.span())?,
            positive_dim(ctx.expect_int(2, "cols")?, ctx.span())?,
        ]
    };
    Ok(Value::Tensor(arc_tensor(
        tensor.reshape(shape, ctx.span())?,
    )))
}

fn native_tensor_matmul(ctx: NativeCallContext<'_>) -> VmResult<Value> {
    if ctx.args().len() != 2 {
        return Err(vm_error(
            "tensor_matmul expects exactly 2 arguments",
            ctx.span(),
        ));
    }
    let left = ctx.expect_tensor(0, "left")?;
    let right = ctx.expect_tensor(1, "right")?;
    Ok(Value::Tensor(arc_tensor(matmul(
        &left,
        &right,
        ctx.span(),
    )?)))
}

fn native_tensor_sum(ctx: NativeCallContext<'_>) -> VmResult<Value> {
    if ctx.args().len() != 1 {
        return Err(vm_error(
            "tensor_sum expects exactly 1 argument",
            ctx.span(),
        ));
    }
    let tensor = ctx.expect_tensor(0, "tensor")?;
    Ok(Value::Float(tensor.sum()))
}

/// Parses a script `exit(code)` trap message back into its code. The CLI
/// maps it to the process exit status; library hosts see the trap.
pub fn parse_exit_code(message: &str) -> Option<i64> {
    message.strip_prefix("exit code ")?.parse().ok()
}

fn native_args_len(ctx: NativeCallContext<'_>) -> VmResult<Value> {
    if !ctx.args().is_empty() {
        return Err(vm_error("args_len expects no arguments", ctx.span()));
    }
    Ok(Value::Int(ctx.host().argv.len() as i64))
}

fn native_args_get(ctx: NativeCallContext<'_>) -> VmResult<Value> {
    if ctx.args().len() != 1 {
        return Err(vm_error("args_get expects exactly 1 argument", ctx.span()));
    }
    let index = ctx.expect_int(0, "index")?;
    let argc = ctx.host().argv.len() as i64;
    if index < 0 || index >= argc {
        return Err(vm_error(
            format!("argv index {index} out of range (argc {argc})"),
            ctx.span(),
        ));
    }
    Ok(Value::String(Arc::<str>::from(
        ctx.host().argv[index as usize].as_str(),
    )))
}

fn check_env_allowed(host: &HostCaps, name: &str, span: Span) -> VmResult<()> {
    match host.allowed_env {
        None => Ok(()),
        Some(set) if set.contains(name) => Ok(()),
        Some(_) => Err(vm_error(
            format!("env '{name}' is disabled by host policy"),
            span,
        )),
    }
}

fn native_env_has(ctx: NativeCallContext<'_>) -> VmResult<Value> {
    if ctx.args().len() != 1 {
        return Err(vm_error("env_has expects exactly 1 argument", ctx.span()));
    }
    let name = ctx.expect_string(0, "name")?;
    check_env_allowed(ctx.host(), &name, ctx.span())?;
    Ok(Value::Bool(std::env::var_os(&name).is_some()))
}

fn native_env_get(ctx: NativeCallContext<'_>) -> VmResult<Value> {
    if ctx.args().len() != 1 {
        return Err(vm_error("env_get expects exactly 1 argument", ctx.span()));
    }
    let name = ctx.expect_string(0, "name")?;
    check_env_allowed(ctx.host(), &name, ctx.span())?;
    match std::env::var(&name) {
        Ok(value) => Ok(Value::String(Arc::<str>::from(value))),
        Err(_) => Err(vm_error(
            format!("env var '{name}' is not set (probe with env_has first)"),
            ctx.span(),
        )),
    }
}

/// Joins a script-supplied relative path under the granted fs root.
/// Lexical escapes (`..` above the root), absolute paths, and the
/// reserved `@` prefix trap here, before any filesystem access.
fn scoped_join(root: &Path, user: &str, span: Span) -> VmResult<PathBuf> {
    if user.is_empty() {
        return Err(vm_error("path must not be empty", span));
    }
    if user.starts_with('@') {
        return Err(vm_error(
            "'@' prefixes are reserved for future package aliases; use a './' or '../' relative path",
            span,
        ));
    }
    let relative = Path::new(user);
    if relative.is_absolute() {
        return Err(vm_error(
            format!(
                "absolute paths are not allowed: '{user}' (use a path relative to the granted fs root)"
            ),
            span,
        ));
    }
    let mut depth: i32 = 0;
    for component in relative.components() {
        match component {
            Component::Prefix(_) | Component::RootDir => {
                return Err(vm_error(
                    format!("absolute paths are not allowed: '{user}'"),
                    span,
                ));
            }
            Component::CurDir => {}
            Component::ParentDir => {
                depth -= 1;
                if depth < 0 {
                    return Err(vm_error(
                        format!("path escapes the granted fs root: '{user}'"),
                        span,
                    ));
                }
            }
            Component::Normal(_) => depth += 1,
        }
    }
    Ok(root.join(relative))
}

fn fs_root_or_trap<'h>(host: &'h HostCaps, span: Span) -> VmResult<&'h Path> {
    host.fs_root.as_deref().ok_or_else(|| {
        vm_error(
            "fs access is disabled by host policy (no fs root granted)",
            span,
        )
    })
}

fn canonical_root(root: &Path, span: Span) -> VmResult<PathBuf> {
    root.canonicalize().map_err(|_| {
        vm_error(
            "fs root granted by host is not accessible".to_string(),
            span,
        )
    })
}

/// Resolves `joined` (already lexically joined under the root) through
/// symlinks and traps when the target escapes the root. Missing targets
/// report ` Ok(None)` so callers can distinguish absence from escape.
fn resolve_within(
    root_canon: &Path,
    joined: &Path,
    user: &str,
    span: Span,
) -> VmResult<Option<PathBuf>> {
    match joined.canonicalize() {
        Ok(target) => {
            if target.starts_with(root_canon) {
                Ok(Some(target))
            } else {
                Err(vm_error(
                    format!("path escapes the granted fs root: '{user}'"),
                    span,
                ))
            }
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(vm_error(
            format!("failed to resolve '{user}': {error}"),
            span,
        )),
    }
}

fn native_fs_exists(ctx: NativeCallContext<'_>) -> VmResult<Value> {
    if ctx.args().len() != 1 {
        return Err(vm_error("fs_exists expects exactly 1 argument", ctx.span()));
    }
    let user = ctx.expect_string(0, "path")?;
    let root = fs_root_or_trap(ctx.host(), ctx.span())?;
    let joined = scoped_join(root, &user, ctx.span())?;
    let root_canon = canonical_root(root, ctx.span())?;
    Ok(Value::Bool(
        resolve_within(&root_canon, &joined, &user, ctx.span())?.is_some(),
    ))
}

fn native_fs_read(ctx: NativeCallContext<'_>) -> VmResult<Value> {
    if ctx.args().len() != 1 {
        return Err(vm_error("fs_read expects exactly 1 argument", ctx.span()));
    }
    let user = ctx.expect_string(0, "path")?;
    let root = fs_root_or_trap(ctx.host(), ctx.span())?;
    let joined = scoped_join(root, &user, ctx.span())?;
    let root_canon = canonical_root(root, ctx.span())?;
    let Some(target) = resolve_within(&root_canon, &joined, &user, ctx.span())? else {
        return Err(vm_error(
            format!("file not found: '{user}' (probe with fs_exists first)"),
            ctx.span(),
        ));
    };
    match std::fs::read_to_string(&target) {
        Ok(text) => Ok(Value::String(Arc::<str>::from(text))),
        Err(error) => Err(vm_error(
            format!("failed to read '{user}': {error}"),
            ctx.span(),
        )),
    }
}

fn native_fs_write(ctx: NativeCallContext<'_>) -> VmResult<Value> {
    if ctx.args().len() != 2 {
        return Err(vm_error("fs_write expects exactly 2 arguments", ctx.span()));
    }
    let user = ctx.expect_string(0, "path")?;
    let data = ctx.expect_string(1, "data")?;
    let root = fs_root_or_trap(ctx.host(), ctx.span())?;
    let joined = scoped_join(root, &user, ctx.span())?;
    let root_canon = canonical_root(root, ctx.span())?;
    // The parent must already exist: no implicit directory creation.
    // Resolving the parent (not the target) keeps symlink-parent escapes
    // visible even when the target itself is new.
    let parent = joined
        .parent()
        .ok_or_else(|| vm_error(format!("invalid path: '{user}'"), ctx.span()))?;
    let parent_canon = parent.canonicalize().map_err(|_| {
        vm_error(
            format!("no such directory for '{user}' (parent must exist)"),
            ctx.span(),
        )
    })?;
    if !parent_canon.starts_with(&root_canon) {
        return Err(vm_error(
            format!("path escapes the granted fs root: '{user}'"),
            ctx.span(),
        ));
    }
    let file_name = joined
        .file_name()
        .ok_or_else(|| vm_error(format!("invalid path: '{user}'"), ctx.span()))?;
    if let Err(error) = std::fs::write(parent_canon.join(file_name), &data) {
        return Err(vm_error(
            format!("failed to write '{user}': {error}"),
            ctx.span(),
        ));
    }
    Ok(Value::Nil)
}

fn native_clock_ms(ctx: NativeCallContext<'_>) -> VmResult<Value> {
    if !ctx.args().is_empty() {
        return Err(vm_error("clock_ms expects no arguments", ctx.span()));
    }
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| vm_error(format!("clock is unavailable: {error}"), ctx.span()))?
        .as_millis();
    let value = i64::try_from(millis)
        .map_err(|_| vm_error("clock value is too large".to_string(), ctx.span()))?;
    Ok(Value::Int(value))
}

fn native_stdin_read(ctx: NativeCallContext<'_>) -> VmResult<Value> {
    if !ctx.args().is_empty() {
        return Err(vm_error("stdin_read expects no arguments", ctx.span()));
    }
    Ok(Value::String(Arc::<str>::from(ctx.host().stdin_data)))
}

fn native_eprint(ctx: NativeCallContext<'_>) -> VmResult<Value> {
    if ctx.args().len() != 1 {
        return Err(vm_error("eprint expects exactly 1 argument", ctx.span()));
    }
    eprintln!("{}", ctx.args()[0]);
    Ok(Value::Nil)
}

fn native_exit(ctx: NativeCallContext<'_>) -> VmResult<Value> {
    if ctx.args().len() != 1 {
        return Err(vm_error("exit expects exactly 1 argument", ctx.span()));
    }
    let code = ctx.expect_int(0, "code")?;
    // The CLI maps this trap to the process exit status (0-125);
    // library hosts observe it as an ordinary vm-phase trap.
    Err(vm_error(format!("exit code {code}"), ctx.span()))
}

pub fn add_values(left: Value, right: Value, span: Span) -> VmResult<Value> {
    match (left, right) {
        (Value::Int(left), Value::Int(right)) => {
            let value = left
                .checked_add(right)
                .ok_or_else(|| vm_error("integer overflow in addition", span))?;
            Ok(Value::Int(value))
        }
        (Value::Float(left), Value::Float(right)) => Ok(Value::Float(left + right)),
        (Value::String(left), Value::String(right)) => {
            let mut value = String::with_capacity(left.len() + right.len());
            value.push_str(&left);
            value.push_str(&right);
            Ok(Value::String(Arc::<str>::from(value)))
        }
        (Value::Tensor(left), Value::Tensor(right)) => Ok(Value::Tensor(arc_tensor(
            tensor_binary(&left, &right, span, "tensor add", |a, b| a + b)?,
        ))),
        (Value::Tensor(left), Value::Float(right)) => Ok(Value::Tensor(arc_tensor(
            tensor_scalar_binary(&left, right, |a, b| a + b),
        ))),
        (Value::Float(left), Value::Tensor(right)) => Ok(Value::Tensor(arc_tensor(
            scalar_tensor_binary(left, &right, |a, b| a + b),
        ))),
        (left, right) => Err(vm_error(
            format!(
                "'+' expects matching Int, Float, String, or Tensor operands, got {} and {}",
                left.stringify(),
                right.stringify()
            ),
            span,
        )),
    }
}

pub fn subtract_values(left: Value, right: Value, span: Span) -> VmResult<Value> {
    match (left, right) {
        (Value::Int(left), Value::Int(right)) => {
            Ok(Value::Int(left.checked_sub(right).ok_or_else(|| {
                vm_error("integer overflow in subtraction", span)
            })?))
        }
        (Value::Float(left), Value::Float(right)) => Ok(Value::Float(left - right)),
        (Value::Tensor(left), Value::Tensor(right)) => Ok(Value::Tensor(arc_tensor(
            tensor_binary(&left, &right, span, "tensor subtract", |a, b| a - b)?,
        ))),
        (Value::Tensor(left), Value::Float(right)) => Ok(Value::Tensor(arc_tensor(
            tensor_scalar_binary(&left, right, |a, b| a - b),
        ))),
        (Value::Float(left), Value::Tensor(right)) => Ok(Value::Tensor(arc_tensor(
            scalar_tensor_binary(left, &right, |a, b| a - b),
        ))),
        (left, right) => Err(vm_error(
            format!(
                "numeric operation expects matching numeric or Tensor/Float types, got {} and {}",
                left.stringify(),
                right.stringify()
            ),
            span,
        )),
    }
}

pub fn multiply_values(left: Value, right: Value, span: Span) -> VmResult<Value> {
    match (left, right) {
        (Value::Int(left), Value::Int(right)) => {
            Ok(Value::Int(left.checked_mul(right).ok_or_else(|| {
                vm_error("integer overflow in multiplication", span)
            })?))
        }
        (Value::Float(left), Value::Float(right)) => Ok(Value::Float(left * right)),
        (Value::Tensor(left), Value::Tensor(right)) => Ok(Value::Tensor(arc_tensor(
            tensor_binary(&left, &right, span, "tensor multiply", |a, b| a * b)?,
        ))),
        (Value::Tensor(left), Value::Float(right)) => Ok(Value::Tensor(arc_tensor(
            tensor_scalar_binary(&left, right, |a, b| a * b),
        ))),
        (Value::Float(left), Value::Tensor(right)) => Ok(Value::Tensor(arc_tensor(
            scalar_tensor_binary(left, &right, |a, b| a * b),
        ))),
        (left, right) => Err(vm_error(
            format!(
                "numeric operation expects matching numeric or Tensor/Float types, got {} and {}",
                left.stringify(),
                right.stringify()
            ),
            span,
        )),
    }
}

pub fn divide_values(left: Value, right: Value, span: Span) -> VmResult<Value> {
    match (left, right) {
        (Value::Int(left), Value::Int(right)) => {
            if right == 0 {
                return Err(vm_error("division by zero", span));
            }
            Ok(Value::Int(left.checked_div(right).ok_or_else(|| {
                vm_error("integer overflow in division", span)
            })?))
        }
        (Value::Float(left), Value::Float(right)) => Ok(Value::Float(left / right)),
        (Value::Tensor(left), Value::Tensor(right)) => Ok(Value::Tensor(arc_tensor(
            tensor_binary(&left, &right, span, "tensor divide", |a, b| a / b)?,
        ))),
        (Value::Tensor(left), Value::Float(right)) => Ok(Value::Tensor(arc_tensor(
            tensor_scalar_binary(&left, right, |a, b| a / b),
        ))),
        (Value::Float(left), Value::Tensor(right)) => Ok(Value::Tensor(arc_tensor(
            scalar_tensor_binary(left, &right, |a, b| a / b),
        ))),
        (left, right) => Err(vm_error(
            format!(
                "numeric operation expects matching numeric or Tensor/Float types, got {} and {}",
                left.stringify(),
                right.stringify()
            ),
            span,
        )),
    }
}

fn expect_shape(ctx: NativeCallContext<'_>) -> VmResult<Vec<usize>> {
    match ctx.args().len() {
        1 => Ok(vec![positive_dim(ctx.expect_int(0, "size")?, ctx.span())?]),
        2 => Ok(vec![
            positive_dim(ctx.expect_int(0, "rows")?, ctx.span())?,
            positive_dim(ctx.expect_int(1, "cols")?, ctx.span())?,
        ]),
        _ => Err(vm_error(
            "expected 1 or 2 integer shape arguments",
            ctx.span(),
        )),
    }
}

fn expect_shape_with_tail(ctx: NativeCallContext<'_>) -> VmResult<(Vec<usize>, usize)> {
    match ctx.args().len() {
        2 => Ok((
            vec![positive_dim(ctx.expect_int(0, "size")?, ctx.span())?],
            1,
        )),
        3 => Ok((
            vec![
                positive_dim(ctx.expect_int(0, "rows")?, ctx.span())?,
                positive_dim(ctx.expect_int(1, "cols")?, ctx.span())?,
            ],
            2,
        )),
        _ => Err(vm_error(
            "expected 2 or 3 arguments with trailing fill value",
            ctx.span(),
        )),
    }
}

fn positive_dim(value: i64, span: Span) -> VmResult<usize> {
    if value <= 0 {
        return Err(vm_error(
            format!("tensor dimensions must be positive, got {}", value),
            span,
        ));
    }
    usize::try_from(value).map_err(|_| {
        vm_error(
            format!("tensor dimension is too large, got {}", value),
            span,
        )
    })
}

pub fn format_native_overload(name: &str, args: &[NativeType]) -> String {
    let args = args
        .iter()
        .map(native_type_name)
        .collect::<Vec<_>>()
        .join(", ");
    format!("{}({})", name, args)
}

pub fn native_type_name(ty: &NativeType) -> &'static str {
    match ty {
        NativeType::Int => "Int",
        NativeType::Float => "Float",
        NativeType::Bool => "Bool",
        NativeType::String => "String",
        NativeType::Void => "Void",
        NativeType::Tensor => "Tensor",
        NativeType::Record => "Record",
    }
}

pub fn native_value_type(value: &Value) -> &'static str {
    match value {
        Value::Int(_) => "Int",
        Value::Float(_) => "Float",
        Value::Bool(_) => "Bool",
        Value::String(_) => "String",
        Value::Tensor(_) => "Tensor",
        Value::Record(_) => "Record",
        Value::Function(_) => "Function",
        Value::Native(_) => "NativeFunction",
        Value::Nil => "Void",
    }
}
