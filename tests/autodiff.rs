use muninn::autodiff::{AutodiffErrorKind, Tape, TensorExpr, grad};
use muninn::span::Span;
use muninn::tensor::Tensor;
use muninn::vm::{Vm, VmOptions};
use muninn::{Value, analyze_document, compile_and_run, compile_to_bytecode};

fn tensor(shape: &[usize], data: &[f64]) -> Tensor {
    Tensor::from_data(shape.to_vec(), data.to_vec(), Span::default()).expect("valid tensor")
}

fn assert_close(actual: &[f64], expected: &[f64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "expected {expected}, got {actual}"
        );
    }
}

#[test]
fn reverse_mode_reduces_broadcast_axes_for_each_operand() {
    let tape = Tape::new();
    let values = tape.variable(tensor(&[2, 1], &[1.0, 2.0]));
    let bias = tape.variable(tensor(&[1, 3], &[0.5, 1.0, 1.5]));
    let loss = values
        .add(&bias)
        .expect("broadcast add")
        .sum()
        .expect("sum");

    let values_gradient = grad(&loss, &values).expect("values gradient");
    let bias_gradient = grad(&loss, &bias).expect("bias gradient");

    assert_close(values_gradient.data(), &[3.0, 3.0], 1e-12);
    assert_close(bias_gradient.data(), &[2.0, 2.0, 2.0], 1e-12);
}

#[test]
fn reverse_mode_handles_matmul_and_axis_reduction() {
    let tape = Tape::new();
    let left = tape.variable(tensor(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
    let right = tape.variable(tensor(&[3, 2], &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]));
    let loss = left.matmul(&right).expect("matmul").sum().expect("sum");

    let left_gradient = grad(&loss, &left).expect("left gradient");
    let right_gradient = grad(&loss, &right).expect("right gradient");

    assert_close(
        left_gradient.data(),
        &[15.0, 19.0, 23.0, 15.0, 19.0, 23.0],
        1e-12,
    );
    assert_close(
        right_gradient.data(),
        &[5.0, 5.0, 7.0, 7.0, 9.0, 9.0],
        1e-12,
    );

    let reduced = left.sum_axis(1).expect("axis reduction");
    assert_eq!(reduced.shape(), vec![2]);
    let reduced_loss = reduced.sum().expect("scalar reduction");
    let reduced_gradient = grad(&reduced_loss, &left).expect("axis gradient");
    assert_close(reduced_gradient.data(), &[1.0; 6], 1e-12);
}

fn curve_loss(
    tape: &Tape,
    x: Tensor,
    observed: Tensor,
    slope: f64,
    intercept: f64,
) -> Result<(TensorExpr, TensorExpr, TensorExpr), muninn::AutodiffError> {
    let input = tape.constant(x);
    let target = tape.constant(observed);
    let slope = tape.variable(Tensor::scalar(slope));
    let intercept = tape.variable(Tensor::scalar(intercept));
    let prediction = input.mul(&slope)?.add(&intercept)?;
    let residual = prediction.sub(&target)?;
    let loss = residual.mul(&residual)?.sum()?;
    Ok((loss, slope, intercept))
}

fn curve_loss_value(x: Tensor, observed: Tensor, slope: f64, intercept: f64) -> f64 {
    let tape = Tape::new();
    let (loss, _, _) = curve_loss(&tape, x, observed, slope, intercept).expect("curve loss");
    loss.value().data()[0]
}

#[test]
fn curve_fit_gradients_agree_with_central_finite_difference() {
    let x = tensor(&[4], &[0.0, 1.0, 2.0, 3.0]);
    let observed = tensor(&[4], &[1.0, 3.0, 5.0, 7.0]);
    let slope = 1.25;
    let intercept = -0.4;
    let tape = Tape::new();
    let (loss, slope_variable, intercept_variable) =
        curve_loss(&tape, x.clone(), observed.clone(), slope, intercept).expect("curve loss");
    let slope_gradient = grad(&loss, &slope_variable).expect("slope gradient").data()[0];
    let intercept_gradient = grad(&loss, &intercept_variable)
        .expect("intercept gradient")
        .data()[0];

    let epsilon = 1e-6;
    let slope_finite_difference =
        (curve_loss_value(x.clone(), observed.clone(), slope + epsilon, intercept)
            - curve_loss_value(x.clone(), observed.clone(), slope - epsilon, intercept))
            / (2.0 * epsilon);
    let intercept_finite_difference =
        (curve_loss_value(x.clone(), observed.clone(), slope, intercept + epsilon)
            - curve_loss_value(x, observed, slope, intercept - epsilon))
            / (2.0 * epsilon);

    assert!((slope_gradient - slope_finite_difference).abs() < 1e-5);
    assert!((intercept_gradient - intercept_finite_difference).abs() < 1e-5);
}

#[test]
fn gradient_reports_scalar_and_shape_contracts() {
    let tape = Tape::new();
    let vector = tape.variable(tensor(&[2], &[1.0, 2.0]));
    let non_scalar = grad(&vector, &vector).expect_err("non-scalar loss");
    assert_eq!(non_scalar.kind, AutodiffErrorKind::NonScalarLoss);
    assert!(non_scalar.message.contains("shape [2]"));

    let other = tape.variable(Tensor::scalar(3.0));
    let missing = grad(&other.sum().expect("sum"), &vector).expect_err("missing gradient");
    assert_eq!(missing.kind, AutodiffErrorKind::MissingGradient);
    assert!(missing.message.contains("loss shape []"));

    let constant = tape.scalar(2.0);
    let invalid_target = grad(&constant, &constant).expect_err("constant target");
    assert_eq!(invalid_target.kind, AutodiffErrorKind::InvalidTarget);
}

#[test]
fn gradient_reports_broadcast_and_axis_shape_errors() {
    let tape = Tape::new();
    let left = tape.variable(tensor(&[2, 2], &[1.0, 2.0, 3.0, 4.0]));
    let right = tape.variable(tensor(&[3, 1], &[1.0, 2.0, 3.0]));
    let shape_error = left.add(&right).expect_err("incompatible broadcast");
    assert_eq!(shape_error.kind, AutodiffErrorKind::ShapeMismatch);
    assert!(shape_error.message.contains("[2, 2]") && shape_error.message.contains("[3, 1]"));

    let axis_error = left.sum_axis(2).expect_err("invalid axis");
    assert_eq!(axis_error.kind, AutodiffErrorKind::InvalidAxis);
    assert!(axis_error.message.contains("axis 2") && axis_error.message.contains("[2, 2]"));

    let rank_one = tape.variable(tensor(&[2], &[1.0, 2.0]));
    let matmul_error = left.matmul(&rank_one).expect_err("invalid matmul rank");
    assert_eq!(matmul_error.kind, AutodiffErrorKind::ShapeMismatch);
    assert!(matmul_error.message.contains("rank-2"));
}

#[test]
fn rejects_zero_dimension_shapes_at_the_tensor_boundary() {
    for source in ["tensor_zeros(0);", "tensor_fill(2, 0, 1.0);"] {
        let errors = compile_and_run(source).expect_err("zero-sized tensor");

        assert_eq!(errors[0].phase, "vm");
        assert!(errors[0].message.contains("dimensions must be positive"));
    }
}

#[test]
fn rejects_cross_tape_operations_and_gradient_targets() {
    let left_tape = Tape::new();
    let right_tape = Tape::new();
    let left = left_tape.variable(Tensor::scalar(2.0));
    let right = right_tape.variable(Tensor::scalar(3.0));

    let add_error = left.add(&right).expect_err("cross-tape add");
    assert_eq!(add_error.kind, AutodiffErrorKind::DifferentGraphs);

    let matmul_error = left.matmul(&right).expect_err("cross-tape matmul");
    assert_eq!(matmul_error.kind, AutodiffErrorKind::DifferentGraphs);

    let loss = left.sum().expect("left loss");
    let grad_error = grad(&loss, &right).expect_err("cross-tape gradient target");
    assert_eq!(grad_error.kind, AutodiffErrorKind::DifferentGraphs);
}

#[test]
fn reverse_mode_reduces_broadcasted_multiply_and_subtract_gradients() {
    let tape = Tape::new();
    let values = tape.variable(tensor(&[2, 1], &[2.0, 3.0]));
    let weights = tape.variable(tensor(&[1, 3], &[10.0, 20.0, 30.0]));
    let product_loss = values
        .mul(&weights)
        .expect("broadcast multiply")
        .sum()
        .expect("sum");

    let values_gradient = grad(&product_loss, &values).expect("values gradient");
    let weights_gradient = grad(&product_loss, &weights).expect("weights gradient");
    assert_close(values_gradient.data(), &[60.0, 60.0], 1e-12);
    assert_close(weights_gradient.data(), &[5.0, 5.0, 5.0], 1e-12);

    let bias = tape.variable(tensor(&[1, 3], &[1.0, 2.0, 3.0]));
    let difference_loss = values
        .sub(&bias)
        .expect("broadcast subtract")
        .sum()
        .expect("sum");
    let bias_gradient = grad(&difference_loss, &bias).expect("bias gradient");
    assert_close(bias_gradient.data(), &[-2.0, -2.0, -2.0], 1e-12);
}

#[test]
fn reverse_mode_handles_axis_zero_and_middle_reductions() {
    let tape = Tape::new();
    let input = tape.variable(tensor(
        &[2, 3, 2],
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
    ));

    let axis_zero = input.sum_axis(0).expect("axis-zero reduction");
    assert_eq!(axis_zero.shape(), vec![3, 2]);
    assert_close(
        axis_zero.value().data(),
        &[8.0, 10.0, 12.0, 14.0, 16.0, 18.0],
        1e-12,
    );
    let axis_zero_gradient =
        grad(&axis_zero.sum().expect("axis-zero loss"), &input).expect("axis-zero gradient");
    assert_close(axis_zero_gradient.data(), &[1.0; 12], 1e-12);

    let middle = input.sum_axis(1).expect("middle-axis reduction");
    assert_eq!(middle.shape(), vec![2, 2]);
    assert_close(middle.value().data(), &[9.0, 12.0, 27.0, 30.0], 1e-12);
    let middle_gradient =
        grad(&middle.sum().expect("middle-axis loss"), &input).expect("middle-axis gradient");
    assert_close(middle_gradient.data(), &[1.0; 12], 1e-12);
}

#[test]
fn graph_operations_do_not_change_existing_interpreter_jit_or_lsp_paths() {
    let source = r#"
fn count() -> Int {
    let mut value: Int = 0;
    while (value < 4) {
        value = value + 1;
    }
    return value;
}

count();
"#;
    let interpreted = compile_and_run(source).expect("interpreter");
    assert!(matches!(interpreted, Value::Int(4)));

    let module = compile_to_bytecode(source).expect("bytecode");
    let mut vm = Vm::new_with_options(
        module,
        VmOptions {
            jit_enabled: true,
            hot_loop_threshold: 1,
        },
    );
    assert!(matches!(vm.run().expect("vm"), Value::Int(4)));
    assert_eq!(vm.jit_stats().expect("jit stats").traces_rejected, 0);

    let analysis = analyze_document("let value: Int = 4; value;");
    assert!(analysis.is_ok());
}

fn sigmoid_scalar(value: f64) -> f64 {
    1.0 / (1.0 + (-value).exp())
}

#[test]
fn sigmoid_value_and_gradient_match_closed_form() {
    let tape = Tape::new();
    let input = tape.variable(tensor(&[3], &[-1.0, 0.0, 2.0]));
    let output = input.sigmoid().expect("sigmoid");
    let expected = [-1.0, 0.0, 2.0].map(sigmoid_scalar);
    assert_close(output.value().data(), &expected, 1e-12);

    let loss = output.sum().expect("sum");
    let gradient = grad(&loss, &input).expect("sigmoid gradient");
    let expected_grad = expected.map(|p| p * (1.0 - p));
    assert_close(gradient.data(), &expected_grad, 1e-12);
}

#[test]
fn bce_with_logits_value_and_gradient_match_finite_difference() {
    let tape = Tape::new();
    let logits = tape.variable(tensor(&[4], &[-1.5, -0.2, 0.7, 2.0]));
    let targets = tape.constant(tensor(&[4], &[0.0, 1.0, 1.0, 0.0]));
    let loss = logits.bce_with_logits(&targets).expect("bce");
    assert!(loss.shape().is_empty());

    let stable = |x: f64, z: f64| x.max(0.0) - x * z + (1.0 + (-x.abs()).exp()).ln();
    let expected =
        (stable(-1.5, 0.0) + stable(-0.2, 1.0) + stable(0.7, 1.0) + stable(2.0, 0.0)) / 4.0;
    assert_close(loss.value().data(), &[expected], 1e-12);

    // d/dx mean(stable_bce) = (sigmoid(x) - z) / n.
    let gradient = grad(&loss, &logits).expect("bce gradient");
    let expected_grad = [-1.5, -0.2, 0.7, 2.0]
        .iter()
        .zip([0.0, 1.0, 1.0, 0.0])
        .map(|(x, z)| (sigmoid_scalar(*x) - z) / 4.0)
        .collect::<Vec<_>>();
    assert_close(gradient.data(), &expected_grad, 1e-12);

    let mismatched = tape.constant(tensor(&[2], &[0.0, 1.0]));
    let shape_error = logits
        .bce_with_logits(&mismatched)
        .expect_err("shape contract");
    assert_eq!(shape_error.kind, AutodiffErrorKind::ShapeMismatch);
}

#[test]
fn cross_entropy_value_and_gradient_match_finite_difference() {
    let tape = Tape::new();
    let logits = tape.variable(tensor(&[2, 3], &[1.0, 2.0, 0.5, -1.0, 0.0, 1.5]));
    let targets = tape.constant(tensor(&[2], &[1.0, 2.0]));
    let loss = logits.cross_entropy_logits(&targets).expect("ce");
    assert!(loss.shape().is_empty());

    let row_loss = |row: &[f64], class: usize| {
        let max = row.iter().fold(f64::NEG_INFINITY, |best, v| best.max(*v));
        let denom: f64 = row.iter().map(|v| (v - max).exp()).sum();
        max + denom.ln() - row[class]
    };
    let expected = (row_loss(&[1.0, 2.0, 0.5], 1) + row_loss(&[-1.0, 0.0, 1.5], 2)) / 2.0;
    assert_close(loss.value().data(), &[expected], 1e-12);

    // d/dx = (softmax(x) - onehot) / batch.
    let softmax = |row: &[f64]| {
        let max = row.iter().fold(f64::NEG_INFINITY, |best, v| best.max(*v));
        let denom: f64 = row.iter().map(|v| (v - max).exp()).sum();
        row.iter()
            .map(|v| (v - max).exp() / denom)
            .collect::<Vec<_>>()
    };
    let mut expected_grad = Vec::new();
    for (row, class) in [vec![1.0, 2.0, 0.5], vec![-1.0, 0.0, 1.5]]
        .iter()
        .zip([1, 2])
    {
        for (col, prob) in softmax(row).iter().enumerate() {
            expected_grad.push((prob - f64::from(col == class)) / 2.0);
        }
    }
    let gradient = grad(&loss, &logits).expect("ce gradient");
    assert_close(gradient.data(), &expected_grad, 1e-12);

    let bad_targets = tape.constant(tensor(&[3], &[0.0, 1.0, 2.0]));
    let shape_error = logits
        .cross_entropy_logits(&bad_targets)
        .expect_err("batch contract");
    assert_eq!(shape_error.kind, AutodiffErrorKind::ShapeMismatch);
}

#[test]
fn new_graph_ops_agree_with_central_finite_difference() {
    // Sigmoid path: loss = sum(sigmoid(x)).
    let tape = Tape::new();
    let x = tape.variable(tensor(&[2], &[0.3, -0.8]));
    let loss = x.sigmoid().expect("sigmoid").sum().expect("sum");
    let analytic = grad(&loss, &x).expect("gradient").data().to_vec();
    let epsilon = 1e-6;
    for (index, point) in [0.3, -0.8].iter().enumerate() {
        let mut plus = [0.3, -0.8];
        let mut minus = [0.3, -0.8];
        plus[index] += epsilon;
        minus[index] -= epsilon;
        let value = |data: [f64; 2]| data.iter().map(|v| sigmoid_scalar(*v)).sum::<f64>();
        let numeric = (value(plus) - value(minus)) / (2.0 * epsilon);
        assert!(
            (analytic[index] - numeric).abs() < 1e-6,
            "sigmoid grad {} vs {numeric}",
            analytic[index]
        );
        let _ = point;
    }
}
