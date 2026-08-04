use std::sync::Arc;

use crate::runtime::{VmResult, vm_error};
use crate::span::Span;

const MAX_TENSOR_ELEMENTS: usize = 10_000_000;

/// A dense tensor with row-major storage.
///
/// The empty shape `[]` represents a scalar with one value. Non-scalar
/// tensors must have strictly positive dimensions; zero-element tensors are
/// intentionally unsupported so shape arithmetic cannot silently turn an
/// empty tensor into a one-element tensor.
#[derive(Debug, Clone)]
pub struct Tensor {
    shape: Vec<usize>,
    data: Vec<f64>,
}

impl Tensor {
    pub fn from_data(shape: Vec<usize>, data: Vec<f64>, span: Span) -> VmResult<Self> {
        let expected = element_count(&shape, span)?;
        if data.len() != expected {
            return Err(vm_error(
                format!(
                    "tensor data length {} does not match shape {} (expected {})",
                    data.len(),
                    format_shape(&shape),
                    expected
                ),
                span,
            ));
        }
        Ok(Self { shape, data })
    }

    pub fn scalar(value: f64) -> Self {
        Self {
            shape: Vec::new(),
            data: vec![value],
        }
    }

    pub fn zeros(shape: Vec<usize>, span: Span) -> VmResult<Self> {
        let len = element_count(&shape, span)?;
        Ok(Self {
            shape,
            data: vec![0.0; len],
        })
    }

    pub fn filled(shape: Vec<usize>, value: f64, span: Span) -> VmResult<Self> {
        let len = element_count(&shape, span)?;
        Ok(Self {
            shape,
            data: vec![value; len],
        })
    }

    pub fn reshape(&self, shape: Vec<usize>, span: Span) -> VmResult<Self> {
        let expected = element_count(&shape, span)?;
        if expected != self.data.len() {
            return Err(vm_error(
                format!(
                    "cannot reshape tensor with {} elements into shape {}",
                    self.data.len(),
                    format_shape(&shape)
                ),
                span,
            ));
        }
        Ok(Self {
            shape,
            data: self.data.clone(),
        })
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn data(&self) -> &[f64] {
        &self.data
    }

    pub fn sum(&self) -> f64 {
        self.data.iter().sum()
    }

    pub fn format(&self) -> String {
        format!(
            "tensor(shape={}, data={:?})",
            format_shape(&self.shape),
            self.data
        )
    }
}

pub fn arc_tensor(tensor: Tensor) -> Arc<Tensor> {
    Arc::new(tensor)
}

pub fn tensor_binary(
    left: &Tensor,
    right: &Tensor,
    span: Span,
    op_name: &str,
    op: impl Fn(f64, f64) -> f64,
) -> VmResult<Tensor> {
    let shape = broadcast_shape(left.shape(), right.shape(), span, op_name)?;
    let result_len = element_count(&shape, span)?;
    let left_strides = strides(left.shape(), span)?;
    let right_strides = strides(right.shape(), span)?;
    let result_strides = strides(&shape, span)?;

    let mut data = Vec::with_capacity(result_len);
    for linear in 0..result_len {
        let index = unravel_index(linear, &shape, &result_strides);
        let left_index = broadcast_index(&index, &shape, left.shape(), &left_strides);
        let right_index = broadcast_index(&index, &shape, right.shape(), &right_strides);
        data.push(op(left.data[left_index], right.data[right_index]));
    }

    Ok(Tensor { shape, data })
}

pub fn tensor_scalar_binary(tensor: &Tensor, scalar: f64, op: impl Fn(f64, f64) -> f64) -> Tensor {
    Tensor {
        shape: tensor.shape.clone(),
        data: tensor.data.iter().map(|value| op(*value, scalar)).collect(),
    }
}

pub fn scalar_tensor_binary(scalar: f64, tensor: &Tensor, op: impl Fn(f64, f64) -> f64) -> Tensor {
    Tensor {
        shape: tensor.shape.clone(),
        data: tensor.data.iter().map(|value| op(scalar, *value)).collect(),
    }
}

pub fn matmul(left: &Tensor, right: &Tensor, span: Span) -> VmResult<Tensor> {
    if left.shape.len() != 2 || right.shape.len() != 2 {
        return Err(vm_error(
            "tensor_matmul expects rank-2 tensors".to_string(),
            span,
        ));
    }

    let (m, k) = (left.shape[0], left.shape[1]);
    let (rhs_k, n) = (right.shape[0], right.shape[1]);
    if k != rhs_k {
        return Err(vm_error(
            format!(
                "tensor_matmul shape mismatch: left {} and right {}",
                format_shape(left.shape()),
                format_shape(right.shape())
            ),
            span,
        ));
    }

    let result_shape = vec![m, n];
    validate_shape(&result_shape, span)?;

    let result_len = m
        .checked_mul(n)
        .ok_or_else(|| vm_error("tensor_matmul result shape is too large", span))?;
    if result_len > MAX_TENSOR_ELEMENTS {
        return Err(vm_error(
            format!(
                "tensor_matmul result has {} elements, maximum is {}",
                result_len, MAX_TENSOR_ELEMENTS
            ),
            span,
        ));
    }

    let mut data = vec![0.0; result_len];
    for row in 0..m {
        for col in 0..n {
            let mut sum = 0.0;
            for inner in 0..k {
                sum += left.data[row * k + inner] * right.data[inner * n + col];
            }
            data[row * n + col] = sum;
        }
    }

    Ok(Tensor {
        shape: result_shape,
        data,
    })
}

fn broadcast_shape(
    left: &[usize],
    right: &[usize],
    span: Span,
    op_name: &str,
) -> VmResult<Vec<usize>> {
    let rank = left.len().max(right.len());
    let mut shape = Vec::with_capacity(rank);

    for index in 0..rank {
        let left_dim = left.iter().rev().nth(index).copied().unwrap_or(1);
        let right_dim = right.iter().rev().nth(index).copied().unwrap_or(1);
        let dim = if left_dim == right_dim {
            left_dim
        } else if left_dim == 1 {
            right_dim
        } else if right_dim == 1 {
            left_dim
        } else {
            return Err(vm_error(
                format!(
                    "{} shape mismatch: left {} and right {} cannot be broadcast",
                    op_name,
                    format_shape(left),
                    format_shape(right)
                ),
                span,
            ));
        };
        shape.push(dim);
    }

    shape.reverse();
    Ok(shape)
}

fn strides(shape: &[usize], span: Span) -> VmResult<Vec<usize>> {
    let mut strides = vec![1usize; shape.len()];
    for index in (1..shape.len()).rev() {
        strides[index - 1] = strides[index]
            .checked_mul(shape[index])
            .ok_or_else(|| vm_error("tensor shape is too large", span))?;
    }
    Ok(strides)
}

fn unravel_index(linear: usize, shape: &[usize], strides: &[usize]) -> Vec<usize> {
    let mut remaining = linear;
    let mut index = Vec::with_capacity(shape.len());
    for (dim, stride) in shape.iter().zip(strides.iter()) {
        if *dim == 0 {
            index.push(0);
            continue;
        }
        index.push(remaining / stride);
        remaining %= stride;
    }
    index
}

fn broadcast_index(
    index: &[usize],
    result_shape: &[usize],
    operand_shape: &[usize],
    operand_strides: &[usize],
) -> usize {
    let offset = result_shape.len().saturating_sub(operand_shape.len());
    let mut linear = 0usize;
    for (dim_index, operand_dim) in operand_shape.iter().enumerate() {
        let source_index = if *operand_dim == 1 {
            0
        } else {
            index[offset + dim_index]
        };
        linear += source_index * operand_strides[dim_index];
    }
    linear
}

pub(crate) fn element_count(shape: &[usize], span: Span) -> VmResult<usize> {
    validate_shape(shape, span)?;
    let count = shape.iter().try_fold(1usize, |total, dim| {
        total
            .checked_mul(*dim)
            .ok_or_else(|| vm_error("tensor shape is too large", span))
    })?;
    if count > MAX_TENSOR_ELEMENTS {
        return Err(vm_error(
            format!(
                "tensor has {} elements, maximum is {}",
                count, MAX_TENSOR_ELEMENTS
            ),
            span,
        ));
    }
    Ok(count)
}

fn validate_shape(shape: &[usize], span: Span) -> VmResult<()> {
    if let Some(axis) = shape.iter().position(|dim| *dim == 0) {
        return Err(vm_error(
            format!(
                "tensor shape dimensions must be positive; axis {} is zero in shape {}",
                axis,
                format_shape(shape)
            ),
            span,
        ));
    }
    Ok(())
}

pub fn format_shape(shape: &[usize]) -> String {
    let dims = shape
        .iter()
        .map(|dim| dim.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    format!("[{}]", dims)
}

#[cfg(test)]
mod tests {
    use super::{Tensor, matmul, tensor_binary};
    use crate::span::Span;

    #[test]
    fn broadcasts_trailing_dimensions() {
        let left = Tensor::filled(vec![2, 1], 2.0, Span::default()).expect("left");
        let right = Tensor::filled(vec![1, 3], 3.0, Span::default()).expect("right");
        let result = tensor_binary(&left, &right, Span::default(), "add", |a, b| a + b)
            .expect("broadcast result");

        assert_eq!(result.shape(), &[2, 3]);
        assert_eq!(result.data(), &[5.0, 5.0, 5.0, 5.0, 5.0, 5.0]);
    }

    #[test]
    fn multiplies_rank_two_tensors() {
        let left = Tensor {
            shape: vec![2, 2],
            data: vec![1.0, 2.0, 3.0, 4.0],
        };
        let right = Tensor {
            shape: vec![2, 2],
            data: vec![5.0, 6.0, 7.0, 8.0],
        };

        let result = matmul(&left, &right, Span::default()).expect("matmul");
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result.data(), &[19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn rejects_tensor_shapes_above_element_limit() {
        let error = Tensor::zeros(vec![10_000_001], Span::default()).expect_err("too large");

        assert!(error.message.contains("maximum is 10000000"));
    }

    #[test]
    fn rejects_tensor_shape_product_overflow() {
        let error = Tensor::zeros(vec![usize::MAX, 2], Span::default()).expect_err("overflow");

        assert!(error.message.contains("tensor shape is too large"));
    }

    #[test]
    fn rejects_zero_dimensions_in_tensor_constructors() {
        let from_data = Tensor::from_data(vec![0], vec![1.0], Span::default())
            .expect_err("zero-sized from_data tensor");
        let zeros =
            Tensor::zeros(vec![2, 0], Span::default()).expect_err("zero-sized zeros tensor");
        let filled =
            Tensor::filled(vec![0, 3], 1.0, Span::default()).expect_err("zero-sized filled tensor");
        let reshaped = Tensor::scalar(1.0)
            .reshape(vec![0, 1], Span::default())
            .expect_err("zero-sized reshaped tensor");

        for error in [from_data, zeros, filled, reshaped] {
            assert!(error.message.contains("dimensions must be positive"));
            assert!(error.message.contains("shape"));
        }
    }

    #[test]
    fn rejects_zero_dimensions_in_matmul_results() {
        let left = Tensor {
            shape: vec![0, 2],
            data: Vec::new(),
        };
        let right = Tensor {
            shape: vec![2, 1],
            data: vec![1.0, 2.0],
        };

        let error = matmul(&left, &right, Span::default()).expect_err("zero-sized matmul");

        assert!(error.message.contains("dimensions must be positive"));
    }
}
