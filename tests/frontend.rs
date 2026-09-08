use muninn::{analyze_document, is_rename_identifier, parse_document, references_to_target};

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

#[test]
fn frontend_exposes_symbol_queries_by_offset() {
    let source = "let value: Int = 1;\nvalue;\n";
    let analysis = analyze_document(source);
    assert!(!analysis.has_errors());

    let offset = source.find("value;\n").expect("value use");
    let symbol = analysis
        .definition_at_offset(offset)
        .expect("definition at offset");
    assert_eq!(symbol.name, "value");
}

#[test]
fn reference_index_distinguishes_same_named_locals_per_function() {
    let source = "fn first(value: Int) -> Int {\n    return value + 1;\n}\nfn second(value: Int) -> Int {\n    return value + 2;\n}\nfirst(1);\nsecond(2);\n";
    let analysis = analyze_document(source);
    assert!(analysis.is_ok());

    // Cursor on the `value` use inside `first` resolves to first's parameter.
    let first_use = source.find("value + 1").expect("first use");
    let first_id = analysis
        .definition_at_offset(first_use)
        .expect("first definition")
        .id;
    let first_spans = references_to_target(&analysis, first_id);
    // Definition plus one use; nothing from `second`.
    assert_eq!(first_spans.len(), 2, "spans: {first_spans:?}");
    for span in &first_spans {
        assert!(
            source[span.offset..span.end_offset].contains("value"),
            "span covers the name: {span:?}"
        );
    }
    let second_use = source.find("value + 2").expect("second use");
    let second_id = analysis
        .definition_at_offset(second_use)
        .expect("second definition")
        .id;
    assert_ne!(first_id, second_id, "no text matching across functions");
    assert_eq!(references_to_target(&analysis, second_id).len(), 2);
}

#[test]
fn rename_identifier_validation_rejects_non_identifiers() {
    assert!(is_rename_identifier("total"));
    assert!(is_rename_identifier("_x1"));
    assert!(!is_rename_identifier("has space"));
    assert!(!is_rename_identifier("1abc"));
    assert!(!is_rename_identifier(""));
    assert!(!is_rename_identifier("a-b"));
}
