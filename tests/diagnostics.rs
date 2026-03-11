use muninn::{analyze_document, compile_and_run};

#[test]
fn renders_golden_parser_error() {
    let source = "let x: Int = ;";
    let analysis = analyze_document(source);
    let error = analysis.diagnostics.first().expect("parser diagnostic");
    assert_eq!(error.phase, "parser");

    let rendered = error.render_with_source(source);
    let expected = "parser error: expected expression\n --> 1:14\n  |\n  1 | let x: Int = ;\n  |              ^";
    assert_eq!(rendered, expected);
}

#[test]
fn renders_golden_type_error() {
    let source = "let total: Int = true;";
    let analysis = analyze_document(source);
    let error = analysis.diagnostics.first().expect("typecheck diagnostic");
    assert_eq!(error.phase, "typecheck");

    let rendered = error.render_with_source(source);
    let expected = "typecheck error: expected initializer of type Int, got Bool\n --> 1:18\n  |\n  1 | let total: Int = true;\n  |                  ^^^^";
    assert_eq!(rendered, expected);
}

#[test]
fn renders_golden_runtime_error() {
    let source = "assert(false);\n1;\n";
    let errors = compile_and_run(source).expect_err("vm error");
    let error = &errors[0];
    assert_eq!(error.phase, "vm");

    let rendered = error.render_with_source(source);
    let expected = "vm error: assertion failed\n --> 1:1\n  |\n  1 | assert(false);\n  | ^^^^^^^^^^^^^";
    assert_eq!(rendered, expected);
}
