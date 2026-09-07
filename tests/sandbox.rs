//! Hostile-script coverage for the deny-by-default host policy.
//!
//! Every case asserts a loud, actionable vm-phase error: no panics, no
//! hangs, no cross-VM leakage. The wall-clock test uses a generous timeout
//! and an infinite loop so it cannot flake on fast machines.

use muninn::native::NativeFunctionKind;
use muninn::vm::VmOptions;
use muninn::{HostPolicy, Vm, compile_to_bytecode, run_bytecode_module_with_policy};

fn sandboxed() -> HostPolicy {
    HostPolicy::sandboxed()
}

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

#[test]
fn infinite_loop_traps_on_fuel() {
    let mut policy = sandboxed();
    policy.max_steps = Some(1_000);
    let error = run_with("let mut run: Bool = true;\nwhile (run) {\n}\nrun;", policy)
        .expect_err("must trap");
    assert!(
        error.contains("fuel exhausted"),
        "loud fuel error, got: {error}"
    );
}

#[test]
fn fuel_exhaustion_is_deterministic() {
    let source =
        "let mut total: Int = 0;\nwhile (total < 100000) {\n    total = total + 1;\n}\ntotal;";
    let mut first = sandboxed();
    first.max_steps = Some(5_000);
    let mut second = sandboxed();
    second.max_steps = Some(5_000);
    assert_eq!(
        run_with(source, first).expect_err("must trap"),
        run_with(source, second).expect_err("must trap")
    );
}

#[test]
fn oversized_tensor_traps_before_allocating() {
    let mut policy = sandboxed();
    policy.allow(NativeFunctionKind::TensorFill);
    policy.allow(NativeFunctionKind::TensorSum);
    policy.max_tensor_elements = 100;
    let error = run_with(
        "let base: Tensor = tensor_fill(64, 64, 1.0);\ntensor_sum(base);",
        policy,
    )
    .expect_err("must trap");
    assert!(
        error.contains("maximum is 100"),
        "host cap in message, got: {error}"
    );
}

#[test]
fn disabled_builtin_traps_with_actionable_name() {
    let policy = sandboxed();
    let error = run_with("print(1);", policy).expect_err("must trap");
    assert!(
        error.contains("disabled by host policy") && error.contains("print"),
        "names the builtin, got: {error}"
    );
}

#[test]
fn allowed_builtins_still_run() {
    let mut policy = sandboxed();
    policy.allow(NativeFunctionKind::Print);
    policy.allow(NativeFunctionKind::Assert);
    policy.max_steps = None;
    let out = run_with("assert(1 == 1);\nprint(7);\n7;", policy).expect("runs");
    assert_eq!(out, "7");
}

#[test]
fn readonly_globals_reject_mutation() {
    let mut policy = sandboxed();
    policy.max_steps = None;
    policy.readonly_globals = true;
    let error = run_with("let mut x: Int = 1;\nx = 2;\nx;", policy).expect_err("must trap");
    assert!(
        error.contains("readonly"),
        "readonly in message, got: {error}"
    );
}

#[test]
fn separate_vms_do_not_share_globals() {
    let first = compile_to_bytecode("let secret: Int = 41;\nsecret;").expect("compiles");
    let mut vm1 = Vm::new_with_policy(first, sandboxed());
    vm1.run().expect("first runs");
    assert!(vm1.global("secret").is_some());

    let second = compile_to_bytecode("1;").expect("compiles");
    let mut vm2 = Vm::new_with_policy(second, sandboxed());
    vm2.run().expect("second runs");
    assert!(
        vm2.global("secret").is_none(),
        "globals live per-VM: nothing leaks across guests"
    );
}

#[test]
fn wall_clock_timeout_traps_infinite_loop() {
    let mut policy = sandboxed();
    policy.max_steps = None;
    policy.wall_clock_timeout_ms = Some(50);
    let error = run_with("let mut run: Bool = true;\nwhile (run) {\n}\nrun;", policy)
        .expect_err("must trap");
    assert!(
        error.contains("wall-clock timeout"),
        "timeout error, got: {error}"
    );
}
