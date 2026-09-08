//! Admission-gate coverage for nominal records: syntax, semantic and
//! type rules, diagnostics, runtime representation, and tooling impact.
//!
//! The contract under test:
//! `record Point { x: Int, y: Int }` declares a nominal type; `Point {
//! x: 1, y: 2 }` constructs it; `p.x` reads a field. Construction
//! requires every field exactly once with matching types. Field
//! assignment is out of scope: whole-record replacement through `Assign`
//! covers mutation.

use muninn::{analyze_document, compile_and_run, compile_to_bytecode, format_source};

fn run(source: &str) -> String {
    compile_and_run(source)
        .expect("record program runs")
        .to_string()
}

fn check_fails_with(source: &str, message: &str) {
    let errors = compile_to_bytecode(source).expect_err("must not compile");
    assert!(
        errors.iter().any(|error| error.message.contains(message)),
        "expected '{message}' in {errors:?}"
    );
}

#[test]
fn constructs_reads_and_compares_records() {
    let value = run("record Point { x: Int, y: Int }\n\
         let p: Point = Point { x: 3, y: 4 };\n\
         assert(p.x == 3);\n\
         assert(p == Point { x: 3, y: 4 });\n\
         assert(p != Point { x: 3, y: 5 });\n\
         p.x + p.y;");
    assert_eq!(value, "7");
}

#[test]
fn infers_record_types_and_passes_them_through_functions() {
    let value = run("record Point { x: Int, y: Int }\n\
         fn origin() -> Point { return Point { x: 0, y: 0 }; }\n\
         fn get_x(point: Point) -> Int { return point.x; }\n\
         let q = origin();\n\
         get_x(q);");
    assert_eq!(value, "0");
}

#[test]
fn nests_records_and_replaces_whole_values() {
    let value = run("record Inner { v: Int }\n\
         record Outer { inner: Inner, tag: String }\n\
         let mut box: Outer = Outer { inner: Inner { v: 1 }, tag: \"a\" };\n\
         assert(box.inner.v == 1);\n\
         box = Outer { inner: Inner { v: 41 }, tag: \"b\" };\n\
         print(box);\n\
         box.inner.v + 1;");
    assert_eq!(value, "42");
}

#[test]
fn records_survive_globals_and_reload() {
    use muninn::vm::Vm;
    let v1 = "record Point { x: Int }\nlet mut p: Point = Point { x: 1 };\np;";
    let v2 = "record Point { x: Int }\nlet mut p: Point = Point { x: 1 };\np.x + 41;";
    let mut vm = Vm::new(compile_to_bytecode(v1).expect("v1"));
    assert_eq!(vm.run().expect("v1 runs").to_string(), "{x: 1}");
    vm.request_reload(compile_to_bytecode(v2).expect("v2"))
        .expect("reload requested");
    vm.apply_pending_reload().expect("reload applies");
    assert_eq!(vm.run().expect("v2 runs").to_string(), "42");
}

#[test]
fn rejects_unknown_record_types() {
    check_fails_with("let p: Nope = Nope { };", "unknown record type 'Nope'");
}

#[test]
fn rejects_unknown_missing_and_mistyped_fields() {
    check_fails_with(
        "record P { x: Int }\nlet p: P = P { y: 1 };",
        "record 'P' has no field 'y'",
    );
    check_fails_with(
        "record P { x: Int }\nlet p: P = P { };",
        "'P' construction is missing field 'x'",
    );
    check_fails_with(
        "record P { x: Int }\nlet p: P = P { x: 1, x: 2 };",
        "duplicate field 'x'",
    );
    check_fails_with(
        "record P { x: Int }\nlet p: P = P { x: true };",
        "field 'x' of 'P' expects Int, got Bool",
    );
    check_fails_with(
        "record P { x: Int }\nlet p: P = P { x: 1 };\np.z;",
        "record 'P' has no field 'z'",
    );
    check_fails_with(
        "record P { x: Int }\n1.x;",
        "value of type Int has no fields",
    );
}

#[test]
fn rejects_duplicate_records_and_fields() {
    check_fails_with(
        "record P { x: Int }\nrecord P { y: Int }",
        "record 'P' is already defined",
    );
    check_fails_with("record P { x: Int, x: Bool }", "duplicate field 'x'");
}

#[test]
fn rejects_mismatched_record_assignment_and_comparison() {
    check_fails_with(
        "record A { x: Int }\nrecord B { x: Int }\nlet a: A = A { x: 1 };\nlet b: B = B { x: 1 };\na == b;",
        "comparison expects",
    );
    check_fails_with(
        "record A { x: Int }\nrecord B { x: Int }\nlet mut a: A = A { x: 1 };\na = B { x: 1 };",
        "cannot assign",
    );
}

#[test]
fn constructor_name_resolves_to_the_record_declaration() {
    let source = "record Point { x: Int }\nlet p: Point = Point { x: 1 };\np;";
    let analysis = analyze_document(source);
    assert!(analysis.is_ok());
    let offset = source.find("Point { x: 1 }").expect("constructor");
    let symbol = analysis
        .definition_at_offset(offset)
        .expect("constructor resolves");
    assert_eq!(symbol.name, "Point");
    assert!(matches!(symbol.kind, muninn::typecheck::SymbolKind::Record));
}

#[test]
fn formatter_keeps_record_syntax_stable() {
    let source = "record Point { x: Int, y: Int }\nlet p: Point = Point { x: 1, y: 2 };\np.x;\n";
    let once = format_source(source);
    assert_eq!(format_source(&once), once);
    compile_to_bytecode(&once).expect("formatted records compile");
}
