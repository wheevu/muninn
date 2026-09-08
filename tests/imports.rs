//! Import loader coverage: file-relative resolution, cycle chains,
//! diamond dedupe, specifier validation, and per-file diagnostics.
//!
//! The loader expands imports before analysis, so the language core is
//! untouched: no new types, no new opcodes, no bytecode bump. Failures
//! name the owning file with the full import chain, never a stack
//! overflow and never silent inclusion.

use muninn::vm::VmOptions;
use muninn::{HostPolicy, compile_project, compile_to_bytecode, load_project};
use std::fs;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

static TEMP_COUNTER: AtomicUsize = AtomicUsize::new(0);

fn write_project(files: &[(&str, &str)]) -> std::path::PathBuf {
    let mut dir = std::env::temp_dir();
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("time")
        .as_nanos();
    let count = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    dir.push(format!(
        "muninn-imports-{}-{stamp}-{count}",
        std::process::id()
    ));
    fs::create_dir_all(&dir).expect("create project dir");
    for (name, contents) in files {
        let path = dir.join(name);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).expect("create subdir");
        }
        fs::write(&path, contents).expect("write project file");
    }
    dir
}

fn run_entry(entry: &std::path::Path) -> Result<String, String> {
    let (module, _) = compile_project(entry).map_err(|errors| {
        errors
            .into_iter()
            .map(|error| error.render())
            .collect::<Vec<_>>()
            .join("\n")
    })?;
    let mut policy = HostPolicy::default();
    if let Some(parent) = entry.parent() {
        policy.set_fs_root(parent.to_path_buf());
    }
    muninn::run_bytecode_module_with_policy(module, policy, VmOptions::default())
        .map(|value| value.to_string())
        .map_err(|errors| {
            errors
                .into_iter()
                .map(|error| error.to_string())
                .collect::<Vec<_>>()
                .join("\n")
        })
}

#[test]
fn single_file_compilation_rejects_imports_actionably() {
    let errors = compile_to_bytecode("import \"./util.mun\";\n1;")
        .expect_err("single-file import must fail");
    assert!(
        errors
            .iter()
            .any(|error| error.message.contains("requires file-based loading")),
        "actionable message, got: {errors:?}"
    );
}

#[test]
fn sibling_import_runs() {
    let dir = write_project(&[
        (
            "util.mun",
            "fn double(n: Int) -> Int {\n    return n * 2;\n}\n",
        ),
        (
            "main.mun",
            "import \"./util.mun\";\nlet out: Int = double(21);\nassert(out == 42);\nout;\n",
        ),
    ]);
    let out = run_entry(&dir.join("main.mun")).expect("runs");
    assert_eq!(out, "42");
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn parent_import_runs() {
    let dir = write_project(&[
        ("shared.mun", "fn one() -> Int {\n    return 1;\n}\n"),
        ("sub/main.mun", "import \"../shared.mun\";\none() + 1;\n"),
    ]);
    let out = run_entry(&dir.join("sub/main.mun")).expect("runs");
    assert_eq!(out, "2");
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn diamond_import_loads_shared_code_once() {
    let dir = write_project(&[
        ("shared.mun", "let value: Int = 7;\n"),
        (
            "left.mun",
            "import \"./shared.mun\";\nfn left() -> Int {\n    return value + 1;\n}\n",
        ),
        (
            "right.mun",
            "import \"./shared.mun\";\nfn right() -> Int {\n    return value + 2;\n}\n",
        ),
        (
            "main.mun",
            "import \"./left.mun\";\nimport \"./right.mun\";\nleft() + right();\n",
        ),
    ]);
    let out = run_entry(&dir.join("main.mun")).expect("diamond runs");
    assert_eq!(out, "17");
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn self_import_reports_a_cycle() {
    let dir = write_project(&[("main.mun", "import \"./main.mun\";\n1;\n")]);
    let errors = compile_project(&dir.join("main.mun")).expect_err("cycle");
    assert!(
        errors
            .iter()
            .any(|error| error.error.message.contains("circular import")
                && error.error.message.contains("main.mun -> main.mun")),
        "self-cycle chain, got: {}",
        errors
            .iter()
            .map(|error| error.error.message.clone())
            .collect::<Vec<_>>()
            .join("; ")
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn two_file_cycle_names_the_full_chain() {
    let dir = write_project(&[
        ("a.mun", "import \"./b.mun\";\n1;\n"),
        ("b.mun", "import \"./a.mun\";\n2;\n"),
    ]);
    let errors = compile_project(&dir.join("a.mun")).expect_err("cycle");
    assert!(
        errors
            .iter()
            .any(|error| error.error.message == "circular import: a.mun -> b.mun -> a.mun"),
        "full chain, got: {}",
        errors
            .iter()
            .map(|error| error.error.message.clone())
            .collect::<Vec<_>>()
            .join("; ")
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn three_file_cycle_names_every_link() {
    let dir = write_project(&[
        ("a.mun", "import \"./b.mun\";\n1;\n"),
        ("b.mun", "import \"./c.mun\";\n2;\n"),
        ("c.mun", "import \"./a.mun\";\n3;\n"),
    ]);
    let errors = compile_project(&dir.join("a.mun")).expect_err("cycle");
    assert!(
        errors.iter().any(|error| error.error.message
            == "circular import: a.mun -> b.mun -> c.mun -> a.mun"),
        "got: {}",
        errors
            .iter()
            .map(|error| error.error.message.clone())
            .collect::<Vec<_>>()
            .join("; ")
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn missing_import_file_is_a_loud_load_error() {
    let dir = write_project(&[("main.mun", "import \"./gone.mun\";\n1;\n")]);
    let errors = compile_project(&dir.join("main.mun")).expect_err("missing file");
    assert!(
        errors
            .iter()
            .any(|error| error.error.message.contains("cannot")),
        "load failure, got: {}",
        errors
            .iter()
            .map(|error| error.error.message.clone())
            .collect::<Vec<_>>()
            .join("; ")
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn import_specifiers_are_validated_before_any_io() {
    for (spec, fragment) in [
        ("/etc/hostname", "absolute import paths are not allowed"),
        ("util.mun", "must be file-relative"),
        ("./util.txt", "must end with '.mun'"),
        ("@stdlib/util.mun", "reserved for future package aliases"),
    ] {
        let dir = write_project(&[("main.mun", &format!("import \"{spec}\";\n1;\n"))]);
        let errors = compile_project(&dir.join("main.mun")).expect_err("bad specifier");
        assert!(
            errors
                .iter()
                .any(|error| error.error.message.contains(fragment)),
            "specifier {spec} rejected with '{fragment}', got: {}",
            errors
                .iter()
                .map(|error| error.error.message.clone())
                .collect::<Vec<_>>()
                .join("; ")
        );
        let _ = fs::remove_dir_all(&dir);
    }
}

#[test]
fn nested_imports_are_rejected_by_the_parser() {
    let dir = write_project(&[(
        "main.mun",
        "fn f() -> Int {\n    import \"./util.mun\";\n    return 1;\n}\nf();\n",
    )]);
    let errors = compile_project(&dir.join("main.mun")).expect_err("nested import");
    assert!(
        errors
            .iter()
            .any(|error| error.error.message.contains("must be top-level")),
        "got: {}",
        errors
            .iter()
            .map(|error| error.error.message.clone())
            .collect::<Vec<_>>()
            .join("; ")
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn type_errors_name_the_dependency_file() {
    let dir = write_project(&[
        ("util.mun", "let broken: Int = true;\n"),
        ("main.mun", "import \"./util.mun\";\n1;\n"),
    ]);
    let errors = compile_project(&dir.join("main.mun")).expect_err("dep type error");
    assert!(
        errors.iter().any(|error| error
            .path
            .file_name()
            .is_some_and(|name| name == "util.mun")),
        "error attributed to util.mun, got: {}",
        errors
            .iter()
            .map(|error| error.path.display().to_string())
            .collect::<Vec<_>>()
            .join("; ")
    );
    assert!(
        errors.iter().any(|error| error
            .render()
            .contains("expected initializer of type Int, got Bool")),
        "original message survives mapping"
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn duplicate_globals_across_files_stay_an_error() {
    let dir = write_project(&[
        ("util.mun", "let value: Int = 1;\n"),
        (
            "main.mun",
            "import \"./util.mun\";\nlet value: Int = 2;\nvalue;\n",
        ),
    ]);
    let errors = compile_project(&dir.join("main.mun")).expect_err("duplicate");
    assert!(
        errors
            .iter()
            .any(|error| error.error.message.contains("already defined")),
        "got: {}",
        errors
            .iter()
            .map(|error| error.error.message.clone())
            .collect::<Vec<_>>()
            .join("; ")
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn loader_orders_dependencies_before_the_entry() {
    let dir = write_project(&[
        ("util.mun", "fn f() -> Int {\n    return 9;\n}\n"),
        ("main.mun", "import \"./util.mun\";\nf();\n"),
    ]);
    let project = load_project(&dir.join("main.mun")).expect("loads");
    assert_eq!(project.files.len(), 2);
    assert!(
        project.files[0]
            .path
            .file_name()
            .is_some_and(|name| name == "util.mun")
    );
    assert!(
        project.files[1]
            .path
            .file_name()
            .is_some_and(|name| name == "main.mun")
    );
    let _ = fs::remove_dir_all(&dir);
}
