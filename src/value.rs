use std::collections::BTreeMap;
use std::fmt::{Display, Formatter};
use std::sync::Arc;

use crate::native::NativeFunctionKind;
use crate::tensor::Tensor;

#[derive(Debug, Clone)]
pub enum Value {
    Int(i64),
    Float(f64),
    Bool(bool),
    String(Arc<str>),
    Tensor(Arc<Tensor>),
    /// A nominal record value: field names to values in sorted order.
    /// Construction order is irrelevant; equality is structural.
    Record(BTreeMap<String, Value>),
    Function(usize),
    Native(NativeFunctionKind),
    Nil,
}

impl Value {
    pub fn kind_name(&self) -> &'static str {
        match self {
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

    pub fn stringify(&self) -> String {
        match self {
            Value::Int(value) => value.to_string(),
            Value::Float(value) => {
                if value.fract() == 0.0 {
                    format!("{value:.1}")
                } else {
                    value.to_string()
                }
            }
            Value::Bool(value) => value.to_string(),
            Value::String(value) => value.to_string(),
            Value::Tensor(value) => value.format(),
            Value::Record(fields) => {
                let body = fields
                    .iter()
                    .map(|(name, value)| format!("{}: {}", name, value.stringify()))
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("{{{}}}", body)
            }
            Value::Function(_) => "<fn>".to_string(),
            Value::Native(_) => "<native fn>".to_string(),
            Value::Nil => "nil".to_string(),
        }
    }

    pub fn equals(&self, other: &Value) -> bool {
        match (self, other) {
            (Value::Int(left), Value::Int(right)) => left == right,
            (Value::Float(left), Value::Float(right)) => left == right,
            (Value::Bool(left), Value::Bool(right)) => left == right,
            (Value::String(left), Value::String(right)) => left == right,
            (Value::Record(left), Value::Record(right)) => {
                left.len() == right.len()
                    && left.iter().all(|(name, value)| {
                        right.get(name).is_some_and(|other| value.equals(other))
                    })
            }
            (Value::Nil, Value::Nil) => true,
            _ => false,
        }
    }
}

impl Display for Value {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.stringify())
    }
}
