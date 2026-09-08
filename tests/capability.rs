//! Capability-IO coverage for the deny-by-default host policy.
//!
//! New IO surface fails closed: argv is empty, env is denied, the
//! filesystem has no root, and stdin is empty unless the host grants
//! them. Every refusal is an actionable vm-phase error, never ambient
//! authority and never a panic. Probes (`args_len`, `env_has`,
//! `fs_exists`) let scripts branch before trapping accessors.

use muninn::native::{NativeFunctionKind, parse_exit_code};
use muninn::vm::VmOptions;
use muninn::{HostPolicy, compile_to_bytecode, run_bytecode_module_with_policy};
use std::fs;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

static TEMP_COUNTER: AtomicUsize = AtomicUsize::new(0);

fn run_with(source: &str, policy: HostPolicy) -> Result<String, String> {
    let module = compile_to_bytecode(source)
        .map_err(|errors| format!("unexpected compile failure: {errors:?}"))?;
    run_bytecode_module_with_policy(module, policy, VmOptions::default())
        .map(|value| value.to_string())
        .map_err(|errors| {
            errors
                .into_iter()
                .map(|error| error.to_string())
                .collect::<Vec<_>>()
                .join("\n")
        })
}

fn allow_io(policy: &mut HostPolicy) {
    for kind in [
        NativeFunctionKind::Print,
        NativeFunctionKind::Assert,
        NativeFunctionKind::ArgsLen,
        NativeFunctionKind::ArgsGet,
        NativeFunctionKind::EnvGet,
        NativeFunctionKind::EnvHas,
        NativeFunctionKind::FsRead,
        NativeFunctionKind::FsExists,
        NativeFunctionKind::FsWrite,
        NativeFunctionKind::ClockMs,
        NativeFunctionKind::StdinRead,
        NativeFunctionKind::Eprint,
        NativeFunctionKind::Exit,
    ] {
        policy.allow(kind);
    }
}

fn temp_root() -> std::path::PathBuf {
    let mut dir = std::env::temp_dir();
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("time")
        .as_nanos();
    let count = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    dir.push(format!("muninn-cap-{}-{stamp}-{count}", std::process::id()));
    fs::create_dir_all(&dir).expect("create temp root");
    dir
}

#[test]
fn argv_is_empty_by_default() {
    assert_eq!(
        run_with("args_len();", HostPolicy::default()),
        Ok("0".to_string())
    );
}

#[test]
fn argv_roundtrips_host_granted_args() {
    let mut policy = HostPolicy::default();
    policy.set_argv(vec!["first".to_string(), "second".to_string()]);
    let out = run_with(
        "assert(args_len() == 2);\nassert(args_get(0) == \"first\");\nargs_get(1);",
        policy,
    )
    .expect("runs");
    assert_eq!(out, "second");
}

#[test]
fn argv_out_of_range_traps_loudly() {
    let mut policy = HostPolicy::default();
    policy.set_argv(vec!["only".to_string()]);
    for source in ["args_get(1);", "args_get(-1);"] {
        let error = run_with(source, policy.clone()).expect_err("must trap");
        assert!(
            error.contains("out of range") && error.contains("argc 1"),
            "names index and argc, got: {error}"
        );
    }
}

#[test]
fn disabled_io_native_traps_with_actionable_name() {
    let policy = HostPolicy::sandboxed();
    for (source, name) in [
        ("args_len();", "args_len"),
        ("env_has(\"PATH\");", "env_has"),
        ("fs_exists(\"x\");", "fs_exists"),
        ("clock_ms();", "clock_ms"),
        ("stdin_read();", "stdin_read"),
        ("eprint(1);", "eprint"),
        ("exit(0);", "exit"),
    ] {
        let error = run_with(source, policy.clone()).expect_err("must trap");
        assert!(
            error.contains("disabled by host policy") && error.contains(name),
            "names the builtin, got: {error}"
        );
    }
}

#[test]
fn env_allow_all_by_default_but_missing_traps() {
    // PATH exists on every supported platform; the unset name is
    // namespaced to this suite so parallel tests cannot collide.
    let out = run_with(
        "assert(env_has(\"PATH\"));\nassert(env_has(\"MUNINN_DEFINITELY_UNSET_XYZ\") == false);\n1;",
        HostPolicy::default(),
    )
    .expect("runs");
    assert_eq!(out, "1");
    let error = run_with(
        "env_get(\"MUNINN_DEFINITELY_UNSET_XYZ\");",
        HostPolicy::default(),
    )
    .expect_err("must trap");
    assert!(
        error.contains("is not set") && error.contains("env_has"),
        "points at the probe, got: {error}"
    );
}

#[test]
fn env_allowlist_denies_unlisted_names() {
    let mut policy = HostPolicy::sandboxed();
    allow_io(&mut policy);
    policy.allow_env("PATH");
    let out = run_with("env_has(\"PATH\");", policy.clone()).expect("listed runs");
    assert_eq!(out, "true");
    let error = run_with("env_has(\"HOME\");", policy).expect_err("must trap");
    assert!(
        error.contains("disabled by host policy") && error.contains("HOME"),
        "names the denied var, got: {error}"
    );
}

#[test]
fn fs_is_denied_without_a_root_even_by_default() {
    // The IO surface is new, so there is no legacy behavior to preserve:
    // fail closed everywhere until the host scopes a root. The default
    // policy allows the native but grants no root; the sandboxed policy
    // refuses earlier at the allowlist gate.
    let error = run_with("fs_read(\"x.txt\");", HostPolicy::default()).expect_err("must trap");
    assert!(
        error.contains("disabled by host policy") && error.contains("no fs root"),
        "fail-closed message, got: {error}"
    );
    let error = run_with("fs_read(\"x.txt\");", HostPolicy::sandboxed()).expect_err("must trap");
    assert!(
        error.contains("disabled by host policy") && error.contains("fs_read"),
        "allowlist gate names the builtin, got: {error}"
    );
}

#[test]
fn fs_roundtrips_inside_the_granted_root() {
    let root = temp_root();
    fs::write(root.join("seed.txt"), "seed-data").expect("seed file");
    let mut policy = HostPolicy::default();
    allow_io(&mut policy);
    policy.set_fs_root(root.clone());
    let out = run_with(
        "assert(fs_exists(\"seed.txt\"));\nassert(fs_exists(\"missing.txt\") == false);\nfs_write(\"out.txt\", fs_read(\"seed.txt\"));\nfs_read(\"out.txt\");",
        policy,
    )
    .expect("runs");
    assert_eq!(out, "seed-data");
    assert_eq!(
        fs::read_to_string(root.join("out.txt")).expect("host reads back"),
        "seed-data"
    );
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn fs_rejects_escapes_missing_files_and_missing_parents() {
    let root = temp_root();
    let mut policy = HostPolicy::default();
    allow_io(&mut policy);
    policy.set_fs_root(root.clone());

    let error = run_with("fs_read(\"../outside.txt\");", policy.clone()).expect_err("escape");
    assert!(
        error.contains("escapes the granted fs root"),
        "got: {error}"
    );

    let error = run_with("fs_read(\"/etc/hostname\");", policy.clone()).expect_err("absolute");
    assert!(
        error.contains("absolute paths are not allowed"),
        "got: {error}"
    );

    let error = run_with("fs_read(\"gone.txt\");", policy.clone()).expect_err("missing");
    assert!(
        error.contains("file not found") && error.contains("fs_exists"),
        "points at the probe, got: {error}"
    );

    let error = run_with("fs_write(\"no/such/dir/out.txt\", \"x\");", policy.clone())
        .expect_err("missing parent");
    assert!(error.contains("parent must exist"), "got: {error}");

    let error = run_with("fs_write(\"../out.txt\", \"x\");", policy).expect_err("write escape");
    assert!(
        error.contains("escapes the granted fs root"),
        "got: {error}"
    );
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn clock_is_denied_by_default_and_monotonic_when_allowed() {
    let mut denied = HostPolicy::sandboxed();
    denied.allow(NativeFunctionKind::ClockMs);
    // Sandboxed + explicitly allowed still works: denial comes from the
    // allowlist, and clock needs no extra grant once allowed.
    let first: i64 = run_with("clock_ms();", denied)
        .expect("runs")
        .parse()
        .expect("integer millis");
    assert!(first > 0);
    let mut policy = HostPolicy::default();
    policy.set_argv(Vec::new());
    let second: i64 = run_with("clock_ms();", policy)
        .expect("runs")
        .parse()
        .expect("integer millis");
    assert!(second >= first);
}

#[test]
fn stdin_is_empty_unless_the_host_seeds_it() {
    assert_eq!(
        run_with("stdin_read();", HostPolicy::default()),
        Ok(String::new())
    );
    let mut policy = HostPolicy::default();
    policy.set_stdin("piped input".to_string());
    assert_eq!(
        run_with("stdin_read();", policy),
        Ok("piped input".to_string())
    );
}

#[test]
fn eprint_runs_and_returns_void() {
    let mut policy = HostPolicy::default();
    policy.allow(NativeFunctionKind::Eprint);
    let out = run_with("eprint(\"to stderr\");\n42;", policy).expect("runs");
    assert_eq!(out, "42");
}

#[test]
fn exit_traps_with_a_machine_readable_code() {
    let mut policy = HostPolicy::default();
    policy.allow(NativeFunctionKind::Exit);
    let error = run_with("exit(3);", policy).expect_err("must trap");
    assert!(error.contains("exit code 3"), "got: {error}");
    assert_eq!(parse_exit_code("exit code 3"), Some(3));
    assert_eq!(parse_exit_code("exit code -1"), Some(-1));
    assert_eq!(parse_exit_code("assertion failed"), None);
}

#[test]
fn revoke_fs_drops_access_mid_policy_lifetime() {
    let root = temp_root();
    fs::write(root.join("a.txt"), "a").expect("seed");
    let mut policy = HostPolicy::default();
    allow_io(&mut policy);
    policy.set_fs_root(root.clone());
    assert_eq!(
        run_with("fs_exists(\"a.txt\");", policy.clone()),
        Ok("true".to_string())
    );
    policy.revoke_fs();
    let error = run_with("fs_exists(\"a.txt\");", policy).expect_err("must trap");
    assert!(error.contains("no fs root"), "got: {error}");
    let _ = fs::remove_dir_all(&root);
}
