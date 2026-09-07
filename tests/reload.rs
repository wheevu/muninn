use muninn::{
    compile_to_bytecode,
    vm::{HostPolicy, ReloadStatus, Vm},
};

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

#[test]
fn reload_rejects_global_kind_change_without_corrupting_state() {
    let source_v1 = r#"
let mut counter: Int = 0;
counter;
"#;

    let source_bad = r#"
let mut counter: String = "nope";
counter;
"#;

    let mut vm = Vm::new(compile_to_bytecode(source_v1).expect("v1"));
    while vm.global("counter").is_none() {
        assert!(vm.step_instruction().expect("step").is_none());
    }

    vm.request_reload(compile_to_bytecode(source_bad).expect("bad"))
        .expect("request reload");
    assert_eq!(vm.poll_safe_point(), ReloadStatus::Ready);

    let error = vm.apply_pending_reload().expect_err("reload error");
    assert!(error.message.contains("counter"));
    assert!(error.message.contains("changed kind"));

    let value = vm.run().expect("run original vm");
    assert_eq!(value.to_string(), "0");
}

#[test]
fn second_reload_request_replaces_the_pending_module() {
    let source_v1 = r#"
let mut counter: Int = 0;
counter;
"#;
    let source_a = r#"
let mut counter: Int = 0;
counter = 111;
counter;
"#;
    let source_b = r#"
let mut counter: Int = 0;
counter = 222;
counter;
"#;

    let mut vm = Vm::new(compile_to_bytecode(source_v1).expect("v1"));
    vm.request_reload(compile_to_bytecode(source_a).expect("a"))
        .expect("request reload a");
    vm.request_reload(compile_to_bytecode(source_b).expect("reload b"))
        .expect("request reload b");

    while vm.poll_safe_point() == ReloadStatus::Pending {
        assert!(vm.step_instruction().expect("step").is_none());
    }
    vm.apply_pending_reload().expect("apply reload");

    assert_eq!(vm.run().expect("run").to_string(), "222");
}

#[test]
fn top_level_loop_is_a_safe_point_but_nested_loop_is_not() {
    let spinning = r#"
fn spin() -> Void {
    while (true) {
    }
}
spin();
"#;
    let mut vm = Vm::new(compile_to_bytecode(spinning).expect("spinning"));
    for _ in 0..2000 {
        assert!(vm.step_instruction().expect("step").is_none());
    }
    assert!(vm.frame_depth() > 1);

    let calmer = r#"
fn spin() -> Void {
    while (false) {
    }
}
spin();
"#;
    vm.request_reload(compile_to_bytecode(calmer).expect("calmer"))
        .expect("request reload");
    for _ in 0..5000 {
        assert!(vm.step_instruction().expect("step").is_none());
    }
    // The nested loop never unwinds to the entry frame, so the staged
    // reload waits forever. There is no execution budget; callers must
    // not stage reloads against non-terminating nested calls.
    assert_eq!(vm.poll_safe_point(), ReloadStatus::Pending);

    let flat = r#"
let mut i: Int = 0;
while (i < 100000) {
    i = i + 1;
}
i;
"#;
    let mut vm = Vm::new(compile_to_bytecode(flat).expect("flat"));
    for _ in 0..50 {
        assert!(vm.step_instruction().expect("step").is_none());
    }
    let patched = r#"
let mut i: Int = 0;
i = 99;
i;
"#;
    vm.request_reload(compile_to_bytecode(patched).expect("patched"))
        .expect("request reload");
    assert_eq!(vm.poll_safe_point(), ReloadStatus::Ready);
}

#[test]
fn fuel_budget_resets_on_successful_reload() {
    let source_v1 = r#"
let mut i: Int = 0;
while (i < 100000) {
    i = i + 1;
}
i;
"#;
    let source_v2 = r#"
let mut i: Int = 0;
i = 7;
i;
"#;

    let mut policy = HostPolicy::sandboxed();
    policy.max_steps = Some(1_000);
    let mut vm = Vm::new_with_policy(compile_to_bytecode(source_v1).expect("v1"), policy);
    let error = vm.run().expect_err("v1 exhausts its fuel");
    assert!(error.to_string().contains("fuel exhausted"));

    vm.request_reload(compile_to_bytecode(source_v2).expect("v2"))
        .expect("request reload");
    vm.apply_pending_reload().expect("apply reload");
    // A fresh program starts a fresh budget: the reloaded module runs clean.
    assert_eq!(vm.run().expect("v2 runs").to_string(), "7");
}
