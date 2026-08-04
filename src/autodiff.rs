//! Eager reverse-mode automatic differentiation for tensor expressions.
//!
//! The autodiff API is deliberately separate from Muninn source execution. A
//! [`Tape`] evaluates each expression eagerly and records only the operation
//! graph needed by [`grad`]. This is the interpreter fallback for gradients:
//! it does not emit bytecode, enter the VM, or ask the experimental JIT to
//! compile anything.

use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt::{Display, Formatter};
use std::rc::Rc;

use crate::error::MuninnError;
use crate::span::Span;
use crate::tensor::{Tensor, element_count, format_shape, matmul, tensor_binary};

type NodeId = usize;

/// The category of a differentiation failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AutodiffErrorKind {
    /// The expression and differentiation target belong to different tapes.
    DifferentGraphs,
    /// The requested target was created as a constant rather than a variable.
    InvalidTarget,
    /// Reverse-mode differentiation requires a scalar loss.
    NonScalarLoss,
    /// The target variable is not an ancestor of the loss.
    MissingGradient,
    /// A reduction axis was outside the tensor rank.
    InvalidAxis,
    /// An operation or gradient rule received incompatible shapes.
    ShapeMismatch,
    /// An eager tensor operation failed for a non-shape reason.
    Runtime,
}

/// A structured, shape-aware error from the eager autodiff path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AutodiffError {
    pub kind: AutodiffErrorKind,
    pub message: String,
}

impl AutodiffError {
    fn new(kind: AutodiffErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }

    /// Returns the stable error category for programmatic handling.
    pub fn kind(&self) -> AutodiffErrorKind {
        self.kind
    }
}

impl Display for AutodiffError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "autodiff error: {}", self.message)
    }
}

impl std::error::Error for AutodiffError {}

impl From<MuninnError> for AutodiffError {
    fn from(error: MuninnError) -> Self {
        let kind = if error.message.contains("shape")
            || error.message.contains("broadcast")
            || error.message.contains("rank-2")
        {
            AutodiffErrorKind::ShapeMismatch
        } else {
            AutodiffErrorKind::Runtime
        };
        Self::new(kind, error.message)
    }
}

#[derive(Debug, Clone)]
enum Operation {
    Variable,
    Constant,
    Add { left: NodeId, right: NodeId },
    Subtract { left: NodeId, right: NodeId },
    Multiply { left: NodeId, right: NodeId },
    Negate { input: NodeId },
    Matmul { left: NodeId, right: NodeId },
    Sum { input: NodeId },
    SumAxis { input: NodeId, axis: usize },
}

#[derive(Debug, Clone)]
struct Node {
    operation: Operation,
    value: Tensor,
}

#[derive(Debug, Default)]
struct Graph {
    nodes: Vec<Node>,
}

/// An owned eager computation graph.
///
/// Nodes keep parent IDs instead of owning parent expressions, so expression
/// handles cannot form reference cycles. Dropping the last [`Tape`] or
/// [`TensorExpr`] releases the graph and all captured tensor values.
#[derive(Debug, Clone, Default)]
pub struct Tape {
    graph: Rc<RefCell<Graph>>,
}

/// A tensor value plus its position in a [`Tape`]. Operations evaluate eagerly
/// and append a node to the same tape.
#[derive(Debug, Clone)]
pub struct TensorExpr {
    graph: Rc<RefCell<Graph>>,
    node: NodeId,
}

/// A variable is an expression created by [`Tape::variable`]. The alias keeps
/// the public API concise while the tape records whether a node is a variable
/// or a constant for target validation.
pub type Variable = TensorExpr;

impl Tape {
    /// Creates an empty eager graph.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a differentiable leaf to the tape.
    pub fn variable(&self, value: Tensor) -> TensorExpr {
        self.push(Operation::Variable, value)
    }

    /// Adds a non-differentiable value to the tape.
    pub fn constant(&self, value: Tensor) -> TensorExpr {
        self.push(Operation::Constant, value)
    }

    /// Adds a scalar constant with scalar shape `[]` to the tape.
    pub fn scalar(&self, value: f64) -> TensorExpr {
        self.constant(Tensor::scalar(value))
    }

    /// Returns the number of captured graph nodes.
    pub fn node_count(&self) -> usize {
        self.graph.borrow().nodes.len()
    }

    fn push(&self, operation: Operation, value: Tensor) -> TensorExpr {
        let node = {
            let mut graph = self.graph.borrow_mut();
            let node = graph.nodes.len();
            graph.nodes.push(Node { operation, value });
            node
        };
        TensorExpr {
            graph: Rc::clone(&self.graph),
            node,
        }
    }
}

impl TensorExpr {
    /// Returns a copy of the eagerly evaluated tensor value.
    pub fn value(&self) -> Tensor {
        self.graph.borrow().nodes[self.node].value.clone()
    }

    /// Returns the shape of the eagerly evaluated value.
    pub fn shape(&self) -> Vec<usize> {
        self.value().shape().to_vec()
    }

    /// Adds two tensor expressions using the same broadcasting rules as the
    /// ordinary Muninn tensor runtime.
    pub fn add(&self, right: &Self) -> Result<Self, AutodiffError> {
        self.binary(right, BinaryOperation::Add)
    }

    /// Subtracts two tensor expressions using broadcasting.
    pub fn sub(&self, right: &Self) -> Result<Self, AutodiffError> {
        self.binary(right, BinaryOperation::Subtract)
    }

    /// Multiplies two tensor expressions element by element using broadcasting.
    pub fn mul(&self, right: &Self) -> Result<Self, AutodiffError> {
        self.binary(right, BinaryOperation::Multiply)
    }

    /// Negates a tensor expression element by element.
    pub fn neg(&self) -> Result<Self, AutodiffError> {
        let value = negate_tensor(&self.value())?;
        Ok(self.push(Operation::Negate { input: self.node }, value))
    }

    /// Multiplies two rank-2 tensor expressions.
    pub fn matmul(&self, right: &Self) -> Result<Self, AutodiffError> {
        self.ensure_same_graph(right)?;
        let value = matmul(&self.value(), &right.value(), Span::default())?;
        Ok(self.push(
            Operation::Matmul {
                left: self.node,
                right: right.node,
            },
            value,
        ))
    }

    /// Reduces every element to a scalar tensor with shape `[]`.
    pub fn sum(&self) -> Result<Self, AutodiffError> {
        let value = Tensor::scalar(self.value().sum());
        Ok(self.push(Operation::Sum { input: self.node }, value))
    }

    /// Sums one axis and removes that axis from the resulting shape.
    pub fn sum_axis(&self, axis: usize) -> Result<Self, AutodiffError> {
        let input = self.value();
        let value = sum_axis_tensor(&input, axis)?;
        Ok(self.push(
            Operation::SumAxis {
                input: self.node,
                axis,
            },
            value,
        ))
    }

    fn binary(&self, right: &Self, operation: BinaryOperation) -> Result<Self, AutodiffError> {
        self.ensure_same_graph(right)?;
        let left_value = self.value();
        let right_value = right.value();
        let (value, node_operation) = match operation {
            BinaryOperation::Add => (
                tensor_binary(
                    &left_value,
                    &right_value,
                    Span::default(),
                    "autodiff add",
                    |left, right| left + right,
                )?,
                Operation::Add {
                    left: self.node,
                    right: right.node,
                },
            ),
            BinaryOperation::Subtract => (
                tensor_binary(
                    &left_value,
                    &right_value,
                    Span::default(),
                    "autodiff subtract",
                    |left, right| left - right,
                )?,
                Operation::Subtract {
                    left: self.node,
                    right: right.node,
                },
            ),
            BinaryOperation::Multiply => (
                tensor_binary(
                    &left_value,
                    &right_value,
                    Span::default(),
                    "autodiff multiply",
                    |left, right| left * right,
                )?,
                Operation::Multiply {
                    left: self.node,
                    right: right.node,
                },
            ),
        };
        Ok(self.push(node_operation, value))
    }

    fn ensure_same_graph(&self, other: &Self) -> Result<(), AutodiffError> {
        if Rc::ptr_eq(&self.graph, &other.graph) {
            Ok(())
        } else {
            Err(AutodiffError::new(
                AutodiffErrorKind::DifferentGraphs,
                "cannot combine tensor expressions from different tapes",
            ))
        }
    }

    fn push(&self, operation: Operation, value: Tensor) -> Self {
        let node = {
            let mut graph = self.graph.borrow_mut();
            let node = graph.nodes.len();
            graph.nodes.push(Node { operation, value });
            node
        };
        Self {
            graph: Rc::clone(&self.graph),
            node,
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum BinaryOperation {
    Add,
    Subtract,
    Multiply,
}

/// Computes the reverse-mode gradient of a scalar loss with respect to a leaf
/// variable.
///
/// The graph is traversed eagerly in reverse creation order. The returned
/// tensor has exactly the variable's shape. `loss` must have scalar shape `[]`;
/// use [`TensorExpr::sum`] or [`TensorExpr::sum_axis`] to make a reduction
/// explicit before calling this function.
pub fn grad(loss: &TensorExpr, variable: &Variable) -> Result<Tensor, AutodiffError> {
    if !Rc::ptr_eq(&loss.graph, &variable.graph) {
        return Err(AutodiffError::new(
            AutodiffErrorKind::DifferentGraphs,
            "loss and variable belong to different tapes",
        ));
    }
    let loss_shape = loss.shape();
    if !loss_shape.is_empty() {
        return Err(AutodiffError::new(
            AutodiffErrorKind::NonScalarLoss,
            format!(
                "grad expects a scalar loss with shape [], got loss shape {}",
                format_shape(&loss_shape)
            ),
        ));
    }

    let graph = loss.graph.borrow();
    let variable_operation = graph
        .nodes
        .get(variable.node)
        .map(|node| node.operation.clone())
        .ok_or_else(|| {
            AutodiffError::new(
                AutodiffErrorKind::InvalidTarget,
                "gradient target is not a node in its tape",
            )
        })?;
    if !matches!(variable_operation, Operation::Variable) {
        return Err(AutodiffError::new(
            AutodiffErrorKind::InvalidTarget,
            "grad target must be a variable created by Tape::variable",
        ));
    }

    let loss_node = graph.nodes.get(loss.node).ok_or_else(|| {
        AutodiffError::new(
            AutodiffErrorKind::Runtime,
            "loss expression is not a node in its tape",
        )
    })?;
    if loss_node.value.shape().is_empty() && loss_node.value.data().len() != 1 {
        return Err(AutodiffError::new(
            AutodiffErrorKind::Runtime,
            "scalar loss contains an invalid number of values",
        ));
    }

    let mut gradients = HashMap::<NodeId, Tensor>::new();
    gradients.insert(loss.node, Tensor::scalar(1.0));
    let mut target_gradient = None;

    for node_id in (0..=loss.node).rev() {
        let Some(upstream) = gradients.remove(&node_id) else {
            continue;
        };
        let operation = graph
            .nodes
            .get(node_id)
            .map(|node| node.operation.clone())
            .ok_or_else(|| {
                AutodiffError::new(
                    AutodiffErrorKind::Runtime,
                    "gradient graph contains an invalid node reference",
                )
            })?;

        match operation {
            Operation::Variable => {
                if node_id == variable.node {
                    target_gradient = Some(upstream);
                }
            }
            Operation::Constant => {}
            Operation::Add { left, right } => {
                let left_shape = node_value(&graph.nodes, left)?.shape().to_vec();
                let right_shape = node_value(&graph.nodes, right)?.shape().to_vec();
                accumulate(
                    &mut gradients,
                    left,
                    reduce_to_shape(&upstream, &left_shape, "add gradient")?,
                )?;
                accumulate(
                    &mut gradients,
                    right,
                    reduce_to_shape(&upstream, &right_shape, "add gradient")?,
                )?;
            }
            Operation::Subtract { left, right } => {
                let left_shape = node_value(&graph.nodes, left)?.shape().to_vec();
                let right_shape = node_value(&graph.nodes, right)?.shape().to_vec();
                accumulate(
                    &mut gradients,
                    left,
                    reduce_to_shape(&upstream, &left_shape, "subtract gradient")?,
                )?;
                accumulate(
                    &mut gradients,
                    right,
                    reduce_to_shape(
                        &negate_tensor(&upstream)?,
                        &right_shape,
                        "subtract gradient",
                    )?,
                )?;
            }
            Operation::Multiply { left, right } => {
                let left_value = node_value(&graph.nodes, left)?;
                let right_value = node_value(&graph.nodes, right)?;
                let left_contribution = tensor_binary(
                    &upstream,
                    right_value,
                    Span::default(),
                    "autodiff multiply gradient",
                    |gradient, value| gradient * value,
                )?;
                let right_contribution = tensor_binary(
                    &upstream,
                    left_value,
                    Span::default(),
                    "autodiff multiply gradient",
                    |gradient, value| gradient * value,
                )?;
                accumulate(
                    &mut gradients,
                    left,
                    reduce_to_shape(&left_contribution, left_value.shape(), "multiply gradient")?,
                )?;
                accumulate(
                    &mut gradients,
                    right,
                    reduce_to_shape(
                        &right_contribution,
                        right_value.shape(),
                        "multiply gradient",
                    )?,
                )?;
            }
            Operation::Negate { input } => {
                accumulate(&mut gradients, input, negate_tensor(&upstream)?)?;
            }
            Operation::Matmul { left, right } => {
                let left_value = node_value(&graph.nodes, left)?;
                let right_value = node_value(&graph.nodes, right)?;
                let right_transposed = transpose_tensor(right_value)?;
                let left_transposed = transpose_tensor(left_value)?;
                accumulate(
                    &mut gradients,
                    left,
                    matmul(&upstream, &right_transposed, Span::default())?,
                )?;
                accumulate(
                    &mut gradients,
                    right,
                    matmul(&left_transposed, &upstream, Span::default())?,
                )?;
            }
            Operation::Sum { input } => {
                let input_shape = node_value(&graph.nodes, input)?.shape().to_vec();
                accumulate(
                    &mut gradients,
                    input,
                    broadcast_to_shape(&upstream, &input_shape, "sum gradient")?,
                )?;
            }
            Operation::SumAxis { input, axis } => {
                let input_shape = node_value(&graph.nodes, input)?.shape().to_vec();
                accumulate(
                    &mut gradients,
                    input,
                    broadcast_after_axis(&upstream, &input_shape, axis)?,
                )?;
            }
        }
    }

    let variable_shape = node_value(&graph.nodes, variable.node)?.shape().to_vec();
    target_gradient.ok_or_else(|| {
        AutodiffError::new(
            AutodiffErrorKind::MissingGradient,
            format!(
                "no gradient path from loss shape {} to variable shape {}",
                format_shape(loss_shape.as_slice()),
                format_shape(&variable_shape)
            ),
        )
    })
}

fn node_value(nodes: &[Node], node: NodeId) -> Result<&Tensor, AutodiffError> {
    nodes.get(node).map(|entry| &entry.value).ok_or_else(|| {
        AutodiffError::new(
            AutodiffErrorKind::Runtime,
            "gradient graph contains an invalid parent node",
        )
    })
}

fn accumulate(
    gradients: &mut HashMap<NodeId, Tensor>,
    node: NodeId,
    contribution: Tensor,
) -> Result<(), AutodiffError> {
    if let Some(previous) = gradients.remove(&node) {
        let combined = tensor_binary(
            &previous,
            &contribution,
            Span::default(),
            "autodiff gradient accumulation",
            |left, right| left + right,
        )?;
        gradients.insert(node, combined);
    } else {
        gradients.insert(node, contribution);
    }
    Ok(())
}

fn negate_tensor(tensor: &Tensor) -> Result<Tensor, AutodiffError> {
    Tensor::from_data(
        tensor.shape().to_vec(),
        tensor.data().iter().map(|value| -value).collect(),
        Span::default(),
    )
    .map_err(AutodiffError::from)
}

fn transpose_tensor(tensor: &Tensor) -> Result<Tensor, AutodiffError> {
    if tensor.shape().len() != 2 {
        return Err(AutodiffError::new(
            AutodiffErrorKind::ShapeMismatch,
            format!(
                "autodiff transpose expects rank-2 tensor, got shape {}",
                format_shape(tensor.shape())
            ),
        ));
    }
    let rows = tensor.shape()[0];
    let cols = tensor.shape()[1];
    let mut data = vec![0.0; tensor.data().len()];
    for row in 0..rows {
        for col in 0..cols {
            data[col * rows + row] = tensor.data()[row * cols + col];
        }
    }
    Tensor::from_data(vec![cols, rows], data, Span::default()).map_err(AutodiffError::from)
}

fn sum_axis_tensor(tensor: &Tensor, axis: usize) -> Result<Tensor, AutodiffError> {
    if axis >= tensor.shape().len() {
        return Err(AutodiffError::new(
            AutodiffErrorKind::InvalidAxis,
            format!(
                "sum_axis axis {} is out of bounds for shape {}",
                axis,
                format_shape(tensor.shape())
            ),
        ));
    }
    let output_shape = tensor
        .shape()
        .iter()
        .enumerate()
        .filter_map(|(index, dim)| (index != axis).then_some(*dim))
        .collect::<Vec<_>>();
    let output_len = element_count(&output_shape, Span::default()).map_err(AutodiffError::from)?;
    let input_strides = strides(tensor.shape())?;
    let output_strides = strides(&output_shape)?;
    let mut data = vec![0.0; output_len];
    for linear in 0..tensor.data().len() {
        let input_index = unravel_index(linear, tensor.shape(), &input_strides);
        let mut output_linear = 0;
        let mut output_axis = 0;
        for (index, value) in input_index.iter().enumerate() {
            if index != axis {
                output_linear += value * output_strides[output_axis];
                output_axis += 1;
            }
        }
        data[output_linear] += tensor.data()[linear];
    }
    Tensor::from_data(output_shape, data, Span::default()).map_err(AutodiffError::from)
}

fn reduce_to_shape(
    gradient: &Tensor,
    target_shape: &[usize],
    operation: &str,
) -> Result<Tensor, AutodiffError> {
    let result_shape = gradient.shape();
    if target_shape.len() > result_shape.len() {
        return Err(shape_error(
            operation,
            result_shape,
            target_shape,
            "target rank is larger than gradient rank",
        ));
    }
    let offset = result_shape.len() - target_shape.len();
    for (index, target_dim) in target_shape.iter().enumerate() {
        let result_dim = result_shape[offset + index];
        if *target_dim != result_dim && *target_dim != 1 {
            return Err(shape_error(
                operation,
                result_shape,
                target_shape,
                "target is not broadcast-compatible with gradient",
            ));
        }
    }

    let target_len = element_count(target_shape, Span::default()).map_err(AutodiffError::from)?;
    let result_strides = strides(result_shape)?;
    let target_strides = strides(target_shape)?;
    let mut data = vec![0.0; target_len];
    for linear in 0..gradient.data().len() {
        let result_index = unravel_index(linear, result_shape, &result_strides);
        let mut target_linear = 0;
        for (index, target_dim) in target_shape.iter().enumerate() {
            let source_index = result_index[offset + index];
            let target_index = if *target_dim == 1 { 0 } else { source_index };
            target_linear += target_index * target_strides[index];
        }
        data[target_linear] += gradient.data()[linear];
    }
    Tensor::from_data(target_shape.to_vec(), data, Span::default()).map_err(AutodiffError::from)
}

fn broadcast_to_shape(
    value: &Tensor,
    target_shape: &[usize],
    operation: &str,
) -> Result<Tensor, AutodiffError> {
    let source_shape = value.shape();
    if source_shape.len() > target_shape.len() {
        return Err(shape_error(
            operation,
            source_shape,
            target_shape,
            "source rank is larger than target rank",
        ));
    }
    let offset = target_shape.len() - source_shape.len();
    for (index, source_dim) in source_shape.iter().enumerate() {
        let target_dim = target_shape[offset + index];
        if *source_dim != target_dim && *source_dim != 1 {
            return Err(shape_error(
                operation,
                source_shape,
                target_shape,
                "source is not broadcast-compatible with target",
            ));
        }
    }

    let target_len = element_count(target_shape, Span::default()).map_err(AutodiffError::from)?;
    let target_strides = strides(target_shape)?;
    let source_strides = strides(source_shape)?;
    let mut data = vec![0.0; target_len];
    for (linear, output) in data.iter_mut().enumerate() {
        let target_index = unravel_index(linear, target_shape, &target_strides);
        let mut source_linear = 0;
        for (index, source_dim) in source_shape.iter().enumerate() {
            let target_index = target_index[offset + index];
            let source_index = if *source_dim == 1 { 0 } else { target_index };
            source_linear += source_index * source_strides[index];
        }
        *output = value.data()[source_linear];
    }
    Tensor::from_data(target_shape.to_vec(), data, Span::default()).map_err(AutodiffError::from)
}

fn broadcast_after_axis(
    value: &Tensor,
    target_shape: &[usize],
    axis: usize,
) -> Result<Tensor, AutodiffError> {
    if axis >= target_shape.len() {
        return Err(AutodiffError::new(
            AutodiffErrorKind::InvalidAxis,
            format!(
                "sum_axis gradient axis {} is out of bounds for shape {}",
                axis,
                format_shape(target_shape)
            ),
        ));
    }
    let expected_shape = target_shape
        .iter()
        .enumerate()
        .filter_map(|(index, dim)| (index != axis).then_some(*dim))
        .collect::<Vec<_>>();
    if value.shape() != expected_shape.as_slice() {
        return Err(shape_error(
            "sum_axis gradient",
            value.shape(),
            target_shape,
            "reduced gradient has an unexpected shape",
        ));
    }
    let mut expanded_shape = value.shape().to_vec();
    expanded_shape.insert(axis, 1);
    let expanded = Tensor::from_data(expanded_shape, value.data().to_vec(), Span::default())
        .map_err(AutodiffError::from)?;
    broadcast_to_shape(&expanded, target_shape, "sum_axis gradient")
}

fn strides(shape: &[usize]) -> Result<Vec<usize>, AutodiffError> {
    let mut strides = vec![1usize; shape.len()];
    for index in (1..shape.len()).rev() {
        strides[index - 1] = strides[index].checked_mul(shape[index]).ok_or_else(|| {
            AutodiffError::new(
                AutodiffErrorKind::ShapeMismatch,
                format!("tensor shape {} is too large", format_shape(shape)),
            )
        })?;
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

fn shape_error(
    operation: &str,
    left_shape: &[usize],
    right_shape: &[usize],
    reason: &str,
) -> AutodiffError {
    AutodiffError::new(
        AutodiffErrorKind::ShapeMismatch,
        format!(
            "{} shape mismatch: {} and {} ({})",
            operation,
            format_shape(left_shape),
            format_shape(right_shape),
            reason
        ),
    )
}

#[cfg(test)]
mod tests {
    use std::rc::Rc;

    use super::{Tape, grad};
    use crate::tensor::Tensor;

    #[test]
    fn graph_is_reclaimed_after_all_handles_drop() {
        let weak;
        {
            let tape = Tape::new();
            weak = Rc::downgrade(&tape.graph);
            let x = tape.variable(Tensor::scalar(2.0));
            let _loss = x.mul(&x).expect("square").sum().expect("sum");
            assert_eq!(tape.node_count(), 3);
            assert!(weak.upgrade().is_some());
        }
        assert!(weak.upgrade().is_none());
    }

    #[test]
    fn scalar_gradient_is_computed_without_vm_execution() {
        let tape = Tape::new();
        let x = tape.variable(Tensor::scalar(3.0));
        let loss = x.mul(&x).expect("square").sum().expect("sum");
        let derivative = grad(&loss, &x).expect("gradient");

        assert_eq!(derivative.shape(), &[]);
        assert_eq!(derivative.data(), &[6.0]);
    }
}
