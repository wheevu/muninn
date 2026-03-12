use muninn::{analyze_document, compile_and_run};

#[test]
fn runs_small_typed_script() {
    let src = r#"
fn add(a: Int, b: Int) -> Int {
    return a + b;
}

let mut total: Int = 0;
while (total < 3) {
    total = add(total, 1);
}

total;
"#;

    let value = compile_and_run(src).expect("value");
    assert_eq!(value.to_string(), "3");
}

#[test]
fn supports_string_concatenation() {
    let src = r#"
let greeting: String = "Mun" + "inn";
greeting;
"#;

    let value = compile_and_run(src).expect("value");
    assert_eq!(value.to_string(), "Muninn");
}

#[test]
fn rejects_type_mismatch() {
    let src = r#"
let x: Int = true;
"#;

    let analysis = analyze_document(src);
    assert!(!analysis.diagnostics.is_empty());
    assert!(analysis.diagnostics[0]
        .message
        .contains("expected initializer of type Int"));
}

#[test]
fn rejects_assignment_to_immutable_binding() {
    let src = r#"
let x: Int = 1;
x = 2;
"#;

    let analysis = analyze_document(src);
    assert!(analysis
        .diagnostics
        .iter()
        .any(|error| error.message.contains("not mutable")));
}

#[test]
fn rejects_missing_non_void_return_paths() {
    let src = r#"
fn maybe(flag: Bool) -> Int {
    if (flag) {
        return 1;
    }
}

let x: Int = maybe(false);
x + 1;
"#;

    let analysis = analyze_document(src);
    assert!(analysis.diagnostics.iter().any(|error| error
        .message
        .contains("may fall through without returning Int")));
}

#[test]
fn reports_runtime_builtin_assert_errors_with_spans() {
    let src = r#"
assert(false);
1;
"#;

    let errors = compile_and_run(src).expect_err("runtime error");
    assert_eq!(errors[0].phase, "vm");
    assert!(errors[0].message.contains("assertion failed"));
    assert!(errors[0].span.line > 0);
}

#[test]
fn reports_runtime_division_by_zero_for_ints() {
    let src = "1 / 0;";
    let errors = compile_and_run(src).expect_err("runtime error");
    assert_eq!(errors[0].phase, "vm");
    assert!(errors[0].message.contains("division by zero"));
}

#[test]
fn reports_runtime_integer_overflow_for_addition() {
    let src = r#"
let max: Int = 9223372036854775807;
max + 1;
"#;
    let errors = compile_and_run(src).expect_err("runtime error");
    assert_eq!(errors[0].phase, "vm");
    assert!(errors[0].message.contains("integer overflow in addition"));
}

#[test]
fn reports_runtime_integer_overflow_for_multiplication() {
    let src = r#"
let big: Int = 3037000500;
big * big;
"#;
    let errors = compile_and_run(src).expect_err("runtime error");
    assert_eq!(errors[0].phase, "vm");
    assert!(errors[0]
        .message
        .contains("integer overflow in multiplication"));
}

#[test]
fn supports_expression_valued_blocks_and_if() {
    let src = r#"
let offset: Float = { let base: Float = 1.5; base + 0.5 };
let result: Float = if (offset > 1.0) {
    offset * 2.0
} else {
    0.0
};
result;
"#;

    let value = compile_and_run(src).expect("value");
    assert_eq!(value.to_string(), "4.0");
}

#[test]
fn runs_tensor_pipeline_with_broadcast_and_matmul() {
    let src = r#"
let base: Tensor = tensor_fill(2, 2, 1.5);
let bias: Tensor = tensor_fill(1, 2, 0.5);
let weights: Tensor = tensor_fill(2, 2, 2.0);
let normalized: Tensor = if (tensor_sum(base) > 0.0) {
    base + bias
} else {
    base
};
let logits: Tensor = tensor_matmul(normalized, weights);
let total: Float = tensor_sum(logits);
total;
"#;

    let value = compile_and_run(src).expect("value");
    assert_eq!(value.to_string(), "32.0");
}

#[test]
fn reports_tensor_broadcast_shape_errors_with_spans() {
    let src = r#"
let left: Tensor = tensor_fill(2, 2, 1.0);
let right: Tensor = tensor_fill(3, 1, 2.0);
left + right;
"#;

    let errors = compile_and_run(src).expect_err("runtime error");
    assert_eq!(errors[0].phase, "vm");
    assert!(errors[0].message.contains("cannot be broadcast"));
    assert!(errors[0].span.line > 0);
}
