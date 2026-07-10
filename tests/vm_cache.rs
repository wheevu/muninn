use muninn::{compile_to_bytecode, vm::Vm};

#[test]
fn global_cache_records_hit_after_initial_miss() {
    let source = r#"
let value: Int = 41;
value;
value;
"#;
    let mut vm = Vm::new(compile_to_bytecode(source).expect("module"));

    let result = vm.run().expect("run");
    let stats = vm.global_cache_stats();

    assert_eq!(result.to_string(), "41");
    assert_eq!(stats.misses, 1);
    assert_eq!(stats.hits, 1);
    assert_eq!(stats.invalidations, 1);
}

#[test]
fn global_cache_is_invalidated_when_global_mutates() {
    let source = r#"
let mut value: Int = 1;
value;
value = 2;
value;
"#;
    let mut vm = Vm::new(compile_to_bytecode(source).expect("module"));

    let result = vm.run().expect("run");
    let stats = vm.global_cache_stats();

    assert_eq!(result.to_string(), "2");
    assert_eq!(stats.misses, 2);
    assert_eq!(stats.hits, 0);
    assert_eq!(stats.invalidations, 2);
}

#[test]
fn reload_request_bumps_invalidation_count_when_cache_is_populated() {
    let source = r#"
let mut value: Int = 1;
value;
value;
"#;
    let mut vm = Vm::new(compile_to_bytecode(source).expect("module"));

    while vm.global_cache_stats().hits == 0 {
        assert!(vm.step_instruction().expect("step").is_none());
    }

    vm.request_reload(compile_to_bytecode(source).expect("reload"))
        .expect("request reload");

    let stats = vm.global_cache_stats();
    assert_eq!(stats.misses, 1);
    assert_eq!(stats.hits, 1);
    assert_eq!(stats.invalidations, 2);
}

#[test]
fn reload_request_bumps_invalidation_count_when_cache_is_empty() {
    let source = r#"
let value: Int = 1;
value;
"#;
    let mut vm = Vm::new(compile_to_bytecode(source).expect("module"));

    while vm.global("value").is_none() {
        assert!(vm.step_instruction().expect("step").is_none());
    }
    assert_eq!(vm.global_cache_stats().invalidations, 1);

    vm.request_reload(compile_to_bytecode(source).expect("reload"))
        .expect("request reload");

    let stats = vm.global_cache_stats();
    assert_eq!(stats.misses, 0);
    assert_eq!(stats.hits, 0);
    assert_eq!(stats.invalidations, 2);
}
