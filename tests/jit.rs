use muninn::vm::{Vm, VmOptions};
use muninn::{Value, compile_to_bytecode};

#[test]
fn jit_is_disabled_by_default() {
    let module = compile_to_bytecode(local_scalar_loop_source()).expect("module");
    let vm = Vm::new(module);

    assert!(vm.jit_stats().is_none());
}

#[test]
fn hot_local_int_loop_compiles_and_matches_interpreter_result() {
    let module = compile_to_bytecode(local_scalar_loop_source()).expect("module");
    let mut vm = Vm::new_with_options(
        module,
        VmOptions {
            jit_enabled: true,
            hot_loop_threshold: 1,
        },
    );

    let value = vm.run().expect("run");
    let stats = vm.jit_stats().expect("jit stats");

    assert!(matches!(value, Value::Int(5)));
    assert_eq!(stats.traces_compiled, 1);
    assert!(stats.interpreted_trace_runs + stats.native_trace_runs > 0);
}

#[test]
fn unsupported_global_loop_is_rejected_without_changing_result() {
    let source = r#"
let mut total: Int = 0;
while (total < 5) {
    total = total + 1;
}
total;
"#;
    let module = compile_to_bytecode(source).expect("module");
    let mut vm = Vm::new_with_options(
        module,
        VmOptions {
            jit_enabled: true,
            hot_loop_threshold: 1,
        },
    );

    let value = vm.run().expect("run");
    let stats = vm.jit_stats().expect("jit stats");

    assert!(matches!(value, Value::Int(5)));
    assert_eq!(stats.traces_compiled, 0);
    assert_eq!(stats.traces_rejected, 1);
}

#[test]
fn trace_interpreter_preserves_runtime_errors() {
    let source = r#"
fn crash() -> Int {
    let mut value: Int = 1;
    while (value < 3) {
        value = value / 0;
    }
    return value;
}

crash();
"#;
    let module = compile_to_bytecode(source).expect("module");
    let mut vm = Vm::new_with_options(
        module,
        VmOptions {
            jit_enabled: true,
            hot_loop_threshold: 1,
        },
    );

    let error = vm.run().expect_err("division by zero");

    assert!(error.message.contains("division by zero"));
    assert!(error.span.line > 0);
}

#[test]
fn jit_preserves_integer_overflow_errors() {
    let source = r#"
fn crash() -> Int {
    let mut value: Int = 9223372036854775807;
    while (value > 0) {
        value = value + 1;
    }
    return value;
}

crash();
"#;
    let module = compile_to_bytecode(source).expect("module");
    let mut vm = Vm::new_with_options(
        module,
        VmOptions {
            jit_enabled: true,
            hot_loop_threshold: 1,
        },
    );

    let error = vm.run().expect_err("integer overflow");

    assert!(error.message.contains("integer overflow in addition"));
    assert!(error.span.line > 0);
}

#[test]
fn reload_request_invalidates_compiled_traces() {
    let module = compile_to_bytecode(local_scalar_loop_source()).expect("module");
    let mut vm = Vm::new_with_options(
        module,
        VmOptions {
            jit_enabled: true,
            hot_loop_threshold: 1,
        },
    );

    while vm.jit_stats().expect("jit stats").traces_compiled == 0 {
        assert!(vm.step_instruction().expect("step").is_none());
    }

    vm.request_reload(compile_to_bytecode(local_scalar_loop_source()).expect("reload"))
        .expect("request reload");

    let stats = vm.jit_stats().expect("jit stats");
    assert_eq!(stats.traces_compiled, 1);
    assert_eq!(stats.interpreted_trace_runs + stats.native_trace_runs, 0);
}

fn local_scalar_loop_source() -> &'static str {
    r#"
fn count() -> Int {
    let mut total: Int = 0;
    while (total < 5) {
        total = total + 1;
    }
    return total;
}

count();
"#
}
