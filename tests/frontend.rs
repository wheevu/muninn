use muninn::{analyze_document, parse_document};

#[test]
fn frontend_exposes_semantics_for_successful_programs() {
    let analysis = analyze_document(
        r#"
fn add(a: Int, b: Int) -> Int {
    return a + b;
}

let result: Int = add(1, 2);
"#,
    );
    assert!(analysis.is_ok());
    assert!(analysis.parsed.is_some());
    assert!(analysis.semantics.is_some());
}

#[test]
fn frontend_keeps_parse_tree_ids_stable_for_type_queries() {
    let program = parse_document("let x: Int = 1 + 2;").expect("program");
    let analysis = analyze_document("let x: Int = 1 + 2;");
    let semantics = analysis.semantics.expect("semantics");
    let expr_id = match &program.statements[0].kind {
        muninn::ast::StmtKind::Let { initializer, .. } => initializer.id,
        _ => panic!("expected let"),
    };
    assert!(semantics.ty_for_expr(expr_id).is_some());
}
