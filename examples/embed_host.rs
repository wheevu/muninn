//! Minimal host embedding an untrusted Muninn script.
//!
//! One `Vm` per guest is the isolation boundary: globals, fuel, and clocks
//! are never shared across VMs. This example runs an untrusted loop under a
//! deny-by-default policy, prints the trapped error with its span, then
//! shows a second VM is unaffected by the first.

use muninn::native::NativeFunctionKind;
use muninn::{HostPolicy, compile_to_bytecode, run_bytecode_module_with_policy, vm::VmOptions};

const UNTRUSTED: &str = r#"
let mut total: Int = 0;
let mut run: Bool = true;
while (run) {
    total = total + 1;
}
total;
"#;

fn main() {
    let mut policy = HostPolicy::sandboxed();
    policy.allow(NativeFunctionKind::Print);
    policy.allow(NativeFunctionKind::Assert);
    policy.max_steps = Some(10_000);

    let module = compile_to_bytecode(UNTRUSTED).expect("untrusted script compiles");
    match run_bytecode_module_with_policy(module, policy, VmOptions::default()) {
        Ok(value) => println!("guest returned: {value}"),
        Err(errors) => {
            for error in &errors {
                println!("guest trapped: {error}");
            }
        }
    }

    // A second VM starts clean: nothing leaks across guests.
    let clean = compile_to_bytecode("1;").expect("clean script compiles");
    let value =
        run_bytecode_module_with_policy(clean, HostPolicy::sandboxed(), VmOptions::default())
            .expect("clean guest runs");
    println!("second guest returned: {value}");
}
