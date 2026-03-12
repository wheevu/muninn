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
