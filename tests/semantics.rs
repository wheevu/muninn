use muninn::{analyze_document, compile_and_run};

#[test]
fn rejects_duplicate_bindings_in_same_scope() {
    let source = r#"
let x: Int = 1;
let x: Int = 2;
"#;

    let analysis = analyze_document(source);
    assert!(
        analysis
            .diagnostics
            .iter()
            .any(|error| error.message.contains("already defined in this scope"))
    );
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
    assert!(
        analysis
            .diagnostics
            .iter()
            .any(|error| error.message.contains("already defined in this scope"))
    );
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
    assert!(
        analysis
            .diagnostics
            .iter()
            .any(|error| error.message.contains("already defined in this scope"))
    );
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
    assert!(
        analysis
            .diagnostics
            .iter()
            .any(|error| error.message.contains("cannot assign to 'value'"))
    );
}

#[test]
fn rejects_assignment_to_builtin_name() {
    let source = r#"
print = 1;
"#;

    let analysis = analyze_document(source);
    assert!(
        analysis
            .diagnostics
            .iter()
            .any(|error| error.message.contains("cannot assign to 'print'"))
    );
}

#[test]
fn assigning_to_function_reports_a_single_diagnostic() {
    let source = r#"
fn value() -> Int {
    return 1;
}

value = 3;
"#;

    let analysis = analyze_document(source);
    assert_eq!(analysis.diagnostics.len(), 1);
    assert!(
        analysis.diagnostics[0]
            .message
            .contains("cannot assign to 'value'")
    );
}

#[test]
fn flags_statements_after_top_level_infinite_loop() {
    let source = r#"
while (true) {
}
let y: Int = 1;
"#;

    let analysis = analyze_document(source);
    assert!(
        analysis
            .diagnostics
            .iter()
            .any(|error| error.message.contains("unreachable statement"))
    );
}

#[test]
fn allows_code_after_top_level_conditional_loop() {
    let source = r#"
let mut done: Bool = false;
while (done) {
    done = true;
}
done;
"#;

    let analysis = analyze_document(source);
    assert!(analysis.diagnostics.is_empty());
}

#[test]
fn native_hint_names_shadowed_builtin_without_stray_space() {
    let source = r#"
fn f() -> Int {
    let print: Int = 1;
    print(2);
    return 0;
}
"#;

    let analysis = analyze_document(source);
    assert!(analysis.diagnostics.iter().any(|error| {
        error
            .message
            .contains("did you mean native function 'print'?)")
    }));
    assert!(
        analysis
            .diagnostics
            .iter()
            .all(|error| !error.message.contains("'? )"))
    );
}

#[test]
fn symbol_lookup_at_file_start_ignores_synthetic_native_spans() {
    let source = "let alpha: Int = 1;\nalpha;\n";

    let analysis = analyze_document(source);
    assert!(analysis.diagnostics.is_empty());
    // Offset 0 is the `let` keyword, not a symbol. Native builtins carry a
    // synthetic line-0 span that must not win lookup here.
    assert!(analysis.symbol_at_offset(0).is_none());

    let use_offset = source.find("alpha;").expect("use site") + 2;
    let symbol = analysis
        .definition_at_offset(use_offset)
        .expect("use site resolves");
    assert_eq!(symbol.name, "alpha");
}
