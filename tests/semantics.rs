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
