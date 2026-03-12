use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

fn write_temp_source(contents: &str) -> PathBuf {
    let mut path = std::env::temp_dir();
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("time")
        .as_nanos();
    path.push(format!("muninn-cli-{stamp}.mun"));
    fs::write(&path, contents).expect("write temp source");
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
    assert!(stdout.contains("muninn run <file>"));
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
