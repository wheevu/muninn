use muninn::{analyze_document, compile_and_run};

#[test]
fn rejects_duplicate_bindings_in_same_scope() {
    let source = r#"
let x: Int = 1;
let x: Int = 2;
"#;

    let analysis = analyze_document(source);
    assert!(analysis
        .diagnostics
        .iter()
        .any(|error| error.message.contains("already defined in this scope")));
}

#[test]
fn allows_shadowing_in_inner_scope() {
    let source = r#"
let x: Int = 1;
if (true) {
    let x: Int = 2;
    print(x);
}
x;
"#;

    let result = compile_and_run(source).expect("run");
    assert_eq!(result.to_string(), "1");
}

#[test]
fn rejects_builtin_shadowing_at_global_scope() {
    let source = r#"
let print: Int = 1;
"#;

    let analysis = analyze_document(source);
    assert!(analysis
        .diagnostics
        .iter()
        .any(|error| error.message.contains("already defined in this scope")));
}

#[test]
fn resolves_local_before_global() {
    let source = r#"
let x: Int = 1;

fn pick() -> Int {
    let x: Int = 2;
    return x;
}

pick();
"#;

    let result = compile_and_run(source).expect("run");
    assert_eq!(result.to_string(), "2");
}

#[test]
fn rejects_duplicate_function_parameters() {
    let source = r#"
fn bad(x: Int, x: Int) -> Int {
    return x;
}

bad(1, 2);
"#;

    let analysis = analyze_document(source);
    assert!(analysis
        .diagnostics
        .iter()
        .any(|error| error.message.contains("already defined in this scope")));
}

#[test]
fn allows_shadowing_parameter_inside_function_body() {
    let source = r#"
fn wrap(x: Int) -> Int {
    let x: Int = x + 1;
    return x;
}

wrap(2);
"#;

    let result = compile_and_run(source).expect("run");
    assert_eq!(result.to_string(), "3");
}

#[test]
fn rejects_assignment_to_function_name() {
    let source = r#"
fn value() -> Int {
    return 1;
}

value = 3;
"#;

    let analysis = analyze_document(source);
    assert!(analysis
        .diagnostics
        .iter()
        .any(|error| error.message.contains("cannot assign to 'value'")));
}

#[test]
fn rejects_assignment_to_builtin_name() {
    let source = r#"
print = 1;
"#;

    let analysis = analyze_document(source);
    assert!(analysis
        .diagnostics
        .iter()
        .any(|error| error.message.contains("cannot assign to 'print'")));
}
