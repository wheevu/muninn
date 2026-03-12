use std::sync::Arc;

use crate::runtime::{VmError, VmResult};
use crate::span::Span;
use crate::tensor::{
    Tensor, arc_tensor, matmul, scalar_tensor_binary, tensor_binary, tensor_scalar_binary,
};
use crate::value::Value;

pub type NativeCallable = for<'a> fn(NativeCallContext<'a>) -> VmResult<Value>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NativeFunctionKind {
    Print,
    Assert,
    TensorZeros,
    TensorFill,
    TensorReshape,
    TensorMatmul,
    TensorSum,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum NativeType {
    Int,
    Float,
    Bool,
    String,
    Void,
    Tensor,
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
}

impl<'a> NativeCallContext<'a> {
    pub fn new(args: &'a [Value], span: Span) -> Self {
        Self { args, span }
    }

    pub fn args(&self) -> &'a [Value] {
        self.args
    }

    pub fn span(&self) -> Span {
        self.span
    }

    pub fn expect_int(&self, index: usize, name: &str) -> VmResult<i64> {
        match self.args.get(index) {
            Some(Value::Int(value)) => Ok(*value),
            Some(other) => Err(self.type_error(index, name, "Int", other)),
            None => Err(VmError::new(
                format!("missing argument '{}' at position {}", name, index),
                self.span,
            )),
        }
    }

    pub fn expect_float(&self, index: usize, name: &str) -> VmResult<f64> {
        match self.args.get(index) {
            Some(Value::Float(value)) => Ok(*value),
            Some(other) => Err(self.type_error(index, name, "Float", other)),
            None => Err(VmError::new(
                format!("missing argument '{}' at position {}", name, index),
                self.span,
            )),
        }
    }

    pub fn expect_bool(&self, index: usize, name: &str) -> VmResult<bool> {
        match self.args.get(index) {
            Some(Value::Bool(value)) => Ok(*value),
            Some(other) => Err(self.type_error(index, name, "Bool", other)),
            None => Err(VmError::new(
                format!("missing argument '{}' at position {}", name, index),
                self.span,
            )),
        }
    }

    pub fn expect_tensor(&self, index: usize, name: &str) -> VmResult<Arc<Tensor>> {
        match self.args.get(index) {
            Some(Value::Tensor(value)) => Ok(Arc::clone(value)),
            Some(other) => Err(self.type_error(index, name, "Tensor", other)),
            None => Err(VmError::new(
                format!("missing argument '{}' at position {}", name, index),
                self.span,
            )),
        }
    }

    fn type_error(&self, index: usize, name: &str, expected: &str, actual: &Value) -> VmError {
        VmError::new(
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

static NATIVE_SPECS: &[NativeSpec] = &[
    NativeSpec {
        kind: NativeFunctionKind::Print,
        name: "print",
        detail: "fn print(value: Int | Float | Bool | String | Tensor) -> Void",
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
];

pub fn registered_natives() -> &'static [NativeSpec] {
    NATIVE_SPECS
}

pub fn native_by_name(name: &str) -> Option<&'static NativeSpec> {
    NATIVE_SPECS.iter().find(|spec| spec.name == name)
}

pub fn native_by_kind(kind: NativeFunctionKind) -> &'static NativeSpec {
    NATIVE_SPECS
        .iter()
        .find(|spec| spec.kind == kind)
        .expect("native function kind")
}

pub fn invoke_native(kind: NativeFunctionKind, args: &[Value], span: Span) -> VmResult<Value> {
    let spec = native_by_kind(kind);
    (spec.runtime)(NativeCallContext::new(args, span))
}

fn native_print(ctx: NativeCallContext<'_>) -> VmResult<Value> {
    if ctx.args().len() != 1 {
        return Err(VmError::new("print expects exactly 1 argument", ctx.span()));
    }
    println!("{}", ctx.args()[0]);
    Ok(Value::Nil)
}

fn native_assert(ctx: NativeCallContext<'_>) -> VmResult<Value> {
    if ctx.args().len() != 1 {
        return Err(VmError::new("assert expects exactly 1 argument", ctx.span()));
    }
    match ctx.expect_bool(0, "condition")? {
        true => Ok(Value::Nil),
        false => Err(VmError::new("assertion failed", ctx.span())),
    }
}

fn native_tensor_zeros(ctx: NativeCallContext<'_>) -> VmResult<Value> {
    let shape = expect_shape(ctx)?;
    Ok(Value::Tensor(arc_tensor(Tensor::zeros(shape))))
}

fn native_tensor_fill(ctx: NativeCallContext<'_>) -> VmResult<Value> {
    let (shape, fill_index) = expect_shape_with_tail(ctx)?;
    let value = ctx.expect_float(fill_index, "value")?;
    Ok(Value::Tensor(arc_tensor(Tensor::filled(shape, value))))
}

fn native_tensor_reshape(ctx: NativeCallContext<'_>) -> VmResult<Value> {
    if !(ctx.args().len() == 2 || ctx.args().len() == 3) {
        return Err(VmError::new(
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
    Ok(Value::Tensor(arc_tensor(tensor.reshape(shape, ctx.span())?)))
}

fn native_tensor_matmul(ctx: NativeCallContext<'_>) -> VmResult<Value> {
    if ctx.args().len() != 2 {
        return Err(VmError::new(
            "tensor_matmul expects exactly 2 arguments",
            ctx.span(),
        ));
    }
    let left = ctx.expect_tensor(0, "left")?;
    let right = ctx.expect_tensor(1, "right")?;
    Ok(Value::Tensor(arc_tensor(matmul(&left, &right, ctx.span())?)))
}

fn native_tensor_sum(ctx: NativeCallContext<'_>) -> VmResult<Value> {
    if ctx.args().len() != 1 {
        return Err(VmError::new(
            "tensor_sum expects exactly 1 argument",
            ctx.span(),
        ));
    }
    let tensor = ctx.expect_tensor(0, "tensor")?;
    Ok(Value::Float(tensor.sum()))
}

pub fn add_values(left: Value, right: Value, span: Span) -> VmResult<Value> {
    match (left, right) {
        (Value::Int(left), Value::Int(right)) => {
            let value = left
                .checked_add(right)
                .ok_or_else(|| VmError::new("integer overflow in addition", span))?;
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
        (left, right) => Err(VmError::new(
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
        (Value::Int(left), Value::Int(right)) => Ok(Value::Int(
            left.checked_sub(right)
                .ok_or_else(|| VmError::new("integer overflow in subtraction", span))?,
        )),
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
        (left, right) => Err(VmError::new(
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
        (Value::Int(left), Value::Int(right)) => Ok(Value::Int(
            left.checked_mul(right)
                .ok_or_else(|| VmError::new("integer overflow in multiplication", span))?,
        )),
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
        (left, right) => Err(VmError::new(
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
                return Err(VmError::new("division by zero", span));
            }
            Ok(Value::Int(
                left.checked_div(right)
                    .ok_or_else(|| VmError::new("integer overflow in division", span))?,
            ))
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
        (left, right) => Err(VmError::new(
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
        _ => Err(VmError::new(
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
        _ => Err(VmError::new(
            "expected 2 or 3 arguments with trailing fill value",
            ctx.span(),
        )),
    }
}

fn positive_dim(value: i64, span: Span) -> VmResult<usize> {
    if value <= 0 {
        return Err(VmError::new(
            format!("tensor dimensions must be positive, got {}", value),
            span,
        ));
    }
    Ok(value as usize)
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
    }
}

pub fn native_value_type(value: &Value) -> &'static str {
    match value {
        Value::Int(_) => "Int",
        Value::Float(_) => "Float",
        Value::Bool(_) => "Bool",
        Value::String(_) => "String",
        Value::Tensor(_) => "Tensor",
        Value::Function(_) => "Function",
        Value::Native(_) => "NativeFunction",
        Value::Nil => "Void",
    }
}
