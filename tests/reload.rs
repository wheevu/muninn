use muninn::{compile_to_bytecode, vm::{ReloadStatus, Vm}};

#[test]
fn reload_swaps_module_and_preserves_globals_at_safe_point() {
    let source_v1 = r#"
let mut counter: Int = 0;
fn step() -> Int {
    counter = counter + 1;
    return counter;
}
while (counter < 3) {
    step();
}
counter;
"#;

    let source_v2 = r#"
let mut counter: Int = 0;
fn step() -> Int {
    counter = counter + 10;
    return counter;
}
while (counter < 12) {
    step();
}
counter;
"#;

    let mut vm = Vm::new(compile_to_bytecode(source_v1).expect("v1"));
    vm.reserve_runtime_capacity(128, 32);

    while vm.frame_depth() <= 1 {
        assert!(vm.step_instruction().expect("step").is_none());
    }

    vm.request_reload(compile_to_bytecode(source_v2).expect("v2"))
        .expect("request reload");
    assert_eq!(vm.poll_safe_point(), ReloadStatus::Pending);

    while vm.poll_safe_point() != ReloadStatus::Ready {
        assert!(vm.step_instruction().expect("step").is_none());
    }

    vm.apply_pending_reload().expect("apply reload");
    let value = vm.run().expect("run reloaded vm");

    assert_eq!(value.to_string(), "21");
    assert_eq!(vm.global("counter").expect("counter").to_string(), "21");
}

#[test]
fn incompatible_reload_is_rejected_without_corrupting_state() {
    let source_v1 = r#"
let mut counter: Int = 0;
counter = counter + 1;
counter;
"#;

    let source_bad = r#"
let mut value: Int = 0;
value;
"#;

    let mut vm = Vm::new(compile_to_bytecode(source_v1).expect("v1"));
    vm.reserve_runtime_capacity(64, 8);
    while vm.global("counter").is_none() {
        assert!(vm.step_instruction().expect("step").is_none());
    }

    vm.request_reload(compile_to_bytecode(source_bad).expect("bad"))
        .expect("request reload");
    assert_eq!(vm.poll_safe_point(), ReloadStatus::Ready);

    let error = vm.apply_pending_reload().expect_err("reload error");
    assert!(error.message.contains("missing"));

    let value = vm.run().expect("run original vm");
    assert_eq!(value.to_string(), "1");
    assert_eq!(vm.global("counter").expect("counter").to_string(), "1");
}
