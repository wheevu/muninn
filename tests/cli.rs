use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

static TEMP_COUNTER: AtomicUsize = AtomicUsize::new(0);

fn write_temp_source(contents: &str) -> PathBuf {
    let mut path = std::env::temp_dir();
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("time")
        .as_nanos();
    let count = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    path.push(format!(
        "muninn-cli-{}-{stamp}-{count}.mun",
        std::process::id()
    ));
    fs::write(&path, contents).expect("write temp source");
    path
}

fn temp_bytecode_path() -> PathBuf {
    let mut path = std::env::temp_dir();
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("time")
        .as_nanos();
    let count = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    path.push(format!(
        "muninn-cli-{}-{stamp}-{count}.mubc",
        std::process::id()
    ));
    path
}

#[test]
fn check_command_reports_ok_for_valid_program() {
    let source = r#"
fn add(a: Int, b: Int) -> Int {
    return a + b;
}

let value: Int = add(1, 2);
"#;
    let path = write_temp_source(source);

    let output = Command::new(env!("CARGO_BIN_EXE_muninn"))
        .arg("check")
        .arg(&path)
        .output()
        .expect("run muninn check");

    let _ = fs::remove_file(&path);

    assert!(output.status.success());
    assert_eq!(String::from_utf8_lossy(&output.stdout).trim(), "ok");
}

#[test]
fn unknown_command_fails_with_helpful_message() {
    let source = "let x: Int = 1;";
    let path = write_temp_source(source);

    let output = Command::new(env!("CARGO_BIN_EXE_muninn"))
        .arg("lint")
        .arg(&path)
        .output()
        .expect("run muninn lint");

    let _ = fs::remove_file(&path);

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("unknown command 'lint'."));
    assert!(stderr.contains("Usage:"));
}

#[test]
fn build_command_emits_mubc_artifact() {
    let source = r#"
let value: Int = 1 + 2;
value;
"#;
    let source_path = write_temp_source(source);
    let output_path = temp_bytecode_path();

    let output = Command::new(env!("CARGO_BIN_EXE_muninn"))
        .arg("build")
        .arg(&source_path)
        .arg("-o")
        .arg(&output_path)
        .output()
        .expect("run muninn build");

    let _ = fs::remove_file(&source_path);
    let _ = fs::remove_file(&output_path);

    assert!(output.status.success());
}

#[test]
fn run_bc_command_executes_compiled_artifact() {
    let source = r#"
let value: Int = 4;
value + 3;
"#;
    let source_path = write_temp_source(source);
    let output_path = temp_bytecode_path();

    let build = Command::new(env!("CARGO_BIN_EXE_muninn"))
        .arg("build")
        .arg(&source_path)
        .arg("-o")
        .arg(&output_path)
        .output()
        .expect("run muninn build");
    assert!(build.status.success());

    let run = Command::new(env!("CARGO_BIN_EXE_muninn"))
        .arg("run-bc")
        .arg(&output_path)
        .output()
        .expect("run muninn run-bc");

    let _ = fs::remove_file(&source_path);
    let _ = fs::remove_file(&output_path);

    assert!(run.status.success());
    assert!(String::from_utf8_lossy(&run.stdout).contains("=> 7"));
}

#[test]
fn run_command_exits_non_zero_on_type_errors() {
    let source = "let x: Int = true;";
    let path = write_temp_source(source);

    let output = Command::new(env!("CARGO_BIN_EXE_muninn"))
        .arg("run")
        .arg(&path)
        .output()
        .expect("run muninn run");

    let _ = fs::remove_file(&path);

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("typecheck error"));
    assert!(stderr.contains("expected initializer of type Int, got Bool"));
}

#[test]
fn help_flag_prints_usage() {
    let output = Command::new(env!("CARGO_BIN_EXE_muninn"))
        .arg("--help")
        .output()
        .expect("run muninn --help");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Usage:"));
    assert!(stdout.contains("muninn run [--jit]"));
}

#[test]
fn run_command_accepts_jit_flag() {
    let source = r#"
fn count() -> Int {
    let mut total: Int = 0;
    while (total < 3) {
        total = total + 1;
    }
    return total;
}
count();
"#;
    let path = write_temp_source(source);

    let output = Command::new(env!("CARGO_BIN_EXE_muninn"))
        .arg("run")
        .arg("--jit")
        .arg("--jit-threshold")
        .arg("1")
        .arg(&path)
        .output()
        .expect("run muninn run --jit");

    let _ = fs::remove_file(&path);

    assert!(output.status.success());
    assert!(String::from_utf8_lossy(&output.stdout).contains("=> 3"));
}

#[test]
fn check_command_without_path_fails() {
    let output = Command::new(env!("CARGO_BIN_EXE_muninn"))
        .arg("check")
        .output()
        .expect("run muninn check");

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("missing source file for command 'check'"));
    assert!(stderr.contains("Usage:"));
}

#[test]
fn run_bc_rejects_mutated_artifact() {
    let source = r#"
let value: Int = 4;
value + 3;
"#;
    let source_path = write_temp_source(source);
    let output_path = temp_bytecode_path();

    let build = Command::new(env!("CARGO_BIN_EXE_muninn"))
        .arg("build")
        .arg(&source_path)
        .arg("-o")
        .arg(&output_path)
        .output()
        .expect("run muninn build");
    assert!(build.status.success());

    // Corrupt the magic header. Artifacts are validated on load, never
    // trusted, so any mutation must fail instead of executing.
    let mut bytes = fs::read(&output_path).expect("read artifact");
    bytes[0] = b'X';
    fs::write(&output_path, bytes).expect("write mutated artifact");

    let run = Command::new(env!("CARGO_BIN_EXE_muninn"))
        .arg("run-bc")
        .arg(&output_path)
        .output()
        .expect("run muninn run-bc");

    let _ = fs::remove_file(&source_path);
    let _ = fs::remove_file(&output_path);

    assert!(!run.status.success());
    assert!(String::from_utf8_lossy(&run.stderr).contains("bytecode error"));
}

#[test]
fn fmt_reformats_in_place_and_check_passes_after() {
    let path = write_temp_source("let x: Int = 1;   \n");

    let format = Command::new(env!("CARGO_BIN_EXE_muninn"))
        .arg("fmt")
        .arg(&path)
        .output()
        .expect("run muninn fmt");
    assert!(format.status.success());
    assert_eq!(
        fs::read_to_string(&path).expect("read back"),
        "let x: Int = 1;\n"
    );

    let check = Command::new(env!("CARGO_BIN_EXE_muninn"))
        .arg("fmt")
        .arg("--check")
        .arg(&path)
        .output()
        .expect("run muninn fmt --check");
    let _ = fs::remove_file(&path);
    assert!(check.status.success());
}

#[test]
fn fmt_check_fails_on_unformatted_source() {
    let path = write_temp_source("let x: Int = 1;   \n");

    let check = Command::new(env!("CARGO_BIN_EXE_muninn"))
        .arg("fmt")
        .arg("--check")
        .arg(&path)
        .output()
        .expect("run muninn fmt --check");
    let _ = fs::remove_file(&path);

    assert!(!check.status.success());
    assert!(String::from_utf8_lossy(&check.stderr).contains("would reformat"));
}

#[test]
fn run_passes_script_argv_after_the_entry_file() {
    let source = "assert(args_len() == 2);\nassert(args_get(0) == \"hello\");\nargs_get(1);\n";
    let path = write_temp_source(source);

    let output = Command::new(env!("CARGO_BIN_EXE_muninn"))
        .arg("run")
        .arg(&path)
        .arg("hello")
        .arg("--")
        .arg("--not-a-cli-flag")
        .output()
        .expect("run muninn run with argv");

    let _ = fs::remove_file(&path);

    assert!(output.status.success());
    assert!(String::from_utf8_lossy(&output.stdout).contains("=> --not-a-cli-flag"));
}

#[test]
fn run_propagates_script_exit_codes() {
    let path = write_temp_source("exit(3);\n");

    let output = Command::new(env!("CARGO_BIN_EXE_muninn"))
        .arg("run")
        .arg(&path)
        .output()
        .expect("run muninn run exit");

    let _ = fs::remove_file(&path);
    assert_eq!(output.status.code(), Some(3));

    // Out-of-range codes clamp to 1: 126/127/128+N stay reserved for
    // the shell and signals, never emitted by the VM.
    let path = write_temp_source("exit(200);\n");
    let output = Command::new(env!("CARGO_BIN_EXE_muninn"))
        .arg("run")
        .arg(&path)
        .output()
        .expect("run muninn run exit clamp");
    let _ = fs::remove_file(&path);
    assert_eq!(output.status.code(), Some(1));
}

#[test]
fn run_type_error_exits_with_code_2() {
    let path = write_temp_source("let x: Int = true;");

    let output = Command::new(env!("CARGO_BIN_EXE_muninn"))
        .arg("run")
        .arg(&path)
        .output()
        .expect("run muninn run");

    let _ = fs::remove_file(&path);

    assert_eq!(output.status.code(), Some(2));
}

#[test]
fn run_no_fs_denies_filesystem_access() {
    let path = write_temp_source("fs_read(\"data.txt\");\n");

    let output = Command::new(env!("CARGO_BIN_EXE_muninn"))
        .arg("run")
        .arg("--no-fs")
        .arg(&path)
        .output()
        .expect("run muninn run --no-fs");

    let _ = fs::remove_file(&path);

    assert_eq!(output.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&output.stderr).contains("no fs root"));
}

#[test]
fn run_piped_stdin_reaches_the_script() {
    let path = write_temp_source("stdin_read();\n");

    let mut child = Command::new(env!("CARGO_BIN_EXE_muninn"))
        .arg("run")
        .arg(&path)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("spawn muninn run");
    child
        .stdin
        .as_mut()
        .expect("stdin")
        .write_all(b"piped words")
        .expect("write stdin");
    let output = child.wait_with_output().expect("wait");

    let _ = fs::remove_file(&path);

    assert!(output.status.success());
    assert!(String::from_utf8_lossy(&output.stdout).contains("piped words"));
}

#[test]
fn run_twice_with_cache_stays_correct() {
    let path = write_temp_source("let value: Int = 40 + 2;\nvalue;\n");

    for _ in 0..2 {
        let output = Command::new(env!("CARGO_BIN_EXE_muninn"))
            .arg("run")
            .arg(&path)
            .output()
            .expect("run muninn run cached");
        assert!(output.status.success());
        assert!(String::from_utf8_lossy(&output.stdout).contains("=> 42"));
    }

    let _ = fs::remove_file(&path);
}

#[test]
fn check_reports_import_cycles_with_code_2() {
    let mut dir = std::env::temp_dir();
    dir.push(format!("muninn-cli-cycle-{}", std::process::id()));
    let _ = fs::create_dir_all(&dir);
    fs::write(dir.join("a.mun"), "import \"./b.mun\";\n1;\n").expect("write a");
    fs::write(dir.join("b.mun"), "import \"./a.mun\";\n2;\n").expect("write b");

    let output = Command::new(env!("CARGO_BIN_EXE_muninn"))
        .arg("check")
        .arg(dir.join("a.mun"))
        .output()
        .expect("run muninn check cycle");

    let _ = fs::remove_dir_all(&dir);

    assert_eq!(output.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&output.stderr).contains("circular import"));
}

#[test]
fn run_bc_passes_argv_to_the_artifact() {
    let source_path = write_temp_source("args_get(0);\n");
    let output_path = temp_bytecode_path();

    let build = Command::new(env!("CARGO_BIN_EXE_muninn"))
        .arg("build")
        .arg(&source_path)
        .arg("-o")
        .arg(&output_path)
        .output()
        .expect("run muninn build");
    assert!(build.status.success());

    let run = Command::new(env!("CARGO_BIN_EXE_muninn"))
        .arg("run-bc")
        .arg(&output_path)
        .arg("--")
        .arg("from-cache")
        .output()
        .expect("run muninn run-bc with argv");

    let _ = fs::remove_file(&source_path);
    let _ = fs::remove_file(&output_path);

    assert!(run.status.success());
    assert!(String::from_utf8_lossy(&run.stdout).contains("=> from-cache"));
}

#[test]
fn run_demo_project_with_imports_args_and_fs() {
    // Living documentation: the checked-in demo project exercises the
    // whole runtime slice (sibling import, argv, scoped fs, env probe).
    let output = Command::new(env!("CARGO_BIN_EXE_muninn"))
        .arg("run")
        .arg("examples/project/main.mun")
        .arg("--")
        .arg("from-demo")
        .output()
        .expect("run demo project");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("hello!"));
    assert!(stdout.contains("from-demo"));
    assert!(stdout.contains("project-data"));
    assert!(stdout.contains("=> hello!"));
}
