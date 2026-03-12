use std::sync::Arc;

use crate::runtime::{VmError, VmResult};
use crate::span::Span;

#[derive(Debug, Clone)]
pub struct Tensor {
    shape: Vec<usize>,
    data: Vec<f64>,
}

impl Tensor {
    pub fn zeros(shape: Vec<usize>) -> Self {
        let len = element_count(&shape);
        Self {
            shape,
            data: vec![0.0; len],
        }
    }

    pub fn filled(shape: Vec<usize>, value: f64) -> Self {
        let len = element_count(&shape);
        Self {
            shape,
            data: vec![value; len],
        }
    }

    pub fn reshape(&self, shape: Vec<usize>, span: Span) -> VmResult<Self> {
        let expected = element_count(&shape);
        if expected != self.data.len() {
            return Err(VmError::new(
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
    let result_len = element_count(&shape);
    let left_strides = strides(left.shape());
    let right_strides = strides(right.shape());
    let result_strides = strides(&shape);

    let mut data = Vec::with_capacity(result_len);
    for linear in 0..result_len {
        let index = unravel_index(linear, &shape, &result_strides);
        let left_index = broadcast_index(&index, &shape, left.shape(), &left_strides);
        let right_index = broadcast_index(&index, &shape, right.shape(), &right_strides);
        data.push(op(left.data[left_index], right.data[right_index]));
    }

    Ok(Tensor { shape, data })
}

pub fn tensor_scalar_binary(
    tensor: &Tensor,
    scalar: f64,
    op: impl Fn(f64, f64) -> f64,
) -> Tensor {
    Tensor {
        shape: tensor.shape.clone(),
        data: tensor.data.iter().map(|value| op(*value, scalar)).collect(),
    }
}

pub fn scalar_tensor_binary(
    scalar: f64,
    tensor: &Tensor,
    op: impl Fn(f64, f64) -> f64,
) -> Tensor {
    Tensor {
        shape: tensor.shape.clone(),
        data: tensor.data.iter().map(|value| op(scalar, *value)).collect(),
    }
}

pub fn matmul(left: &Tensor, right: &Tensor, span: Span) -> VmResult<Tensor> {
    if left.shape.len() != 2 || right.shape.len() != 2 {
        return Err(VmError::new(
            "tensor_matmul expects rank-2 tensors".to_string(),
            span,
        ));
    }

    let (m, k) = (left.shape[0], left.shape[1]);
    let (rhs_k, n) = (right.shape[0], right.shape[1]);
    if k != rhs_k {
        return Err(VmError::new(
            format!(
                "tensor_matmul shape mismatch: left {} and right {}",
                format_shape(left.shape()),
                format_shape(right.shape())
            ),
            span,
        ));
    }

    let mut data = vec![0.0; m * n];
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
        shape: vec![m, n],
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
        let left_dim = left
            .iter()
            .rev()
            .nth(index)
            .copied()
            .unwrap_or(1);
        let right_dim = right
            .iter()
            .rev()
            .nth(index)
            .copied()
            .unwrap_or(1);
        let dim = if left_dim == right_dim {
            left_dim
        } else if left_dim == 1 {
            right_dim
        } else if right_dim == 1 {
            left_dim
        } else {
            return Err(VmError::new(
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

fn strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for index in (1..shape.len()).rev() {
        strides[index - 1] = strides[index] * shape[index];
    }
    strides
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

fn element_count(shape: &[usize]) -> usize {
    shape.iter().copied().product::<usize>().max(1)
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
        let left = Tensor::filled(vec![2, 1], 2.0);
        let right = Tensor::filled(vec![1, 3], 3.0);
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
}
