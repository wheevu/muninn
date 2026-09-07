//! Long-running host demo: a tick counter survives a code reload.
//!
//! Version 1 counts by ones. The host reloads version 2, which counts by
//! tens, while the live `counter` global is preserved at the VM safe point.
//! An incompatible version 3 is rejected and the running program keeps its
//! state, which is the snapshot-and-rollback story in miniature.

use std::time::Instant;

use muninn::compile_to_bytecode;
use muninn::vm::Vm;

const V1: &str = r#"
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

const V2: &str = r#"
let mut counter: Int = 0;
fn step() -> Int {
    counter = counter + 10;
    return counter;
}
while (counter < 33) {
    step();
}
counter;
"#;

const V3_BAD: &str = r#"
let mut renamed: Int = 0;
renamed;
"#;

fn main() {
    let mut vm = Vm::new(compile_to_bytecode(V1).expect("v1 compiles"));
    let first = vm.run().expect("v1 runs");
    println!("v1 counter: {first}");

    let started = Instant::now();
    vm.request_reload(compile_to_bytecode(V2).expect("v2 compiles"))
        .expect("reload requested");
    vm.apply_pending_reload().expect("reload applies");
    let reload_us = started.elapsed().as_micros();
    let second = vm.run().expect("v2 runs");
    println!("v2 counter after reload: {second} (reload took {reload_us} us)");

    let rejected = vm
        .request_reload(compile_to_bytecode(V3_BAD).expect("v3 compiles"))
        .is_ok()
        .then(|| vm.apply_pending_reload())
        .expect("reload staged");
    match rejected {
        Ok(()) => println!("unexpected: bad reload applied"),
        Err(error) => println!("bad reload rejected, state kept: {error}"),
    }
    println!(
        "counter still: {}",
        vm.global("counter").expect("counter survives")
    );
}
