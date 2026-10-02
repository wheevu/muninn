//! Host-side views of a running VM: captured output, the value stack, and
//! call frames.
//!
//! These are what an embedder reads while a script runs. The browser
//! playground in `site/` is built on them, so the three claims pinned here are
//! the ones that page depends on: captured text arrives in the order it was
//! printed, a stepped frame reports the locals its compiler reserved, and a
//! bounded capture keeps whole lines while reporting what it dropped. A
//! printed tensor line is marked by the interpreter, never guessed from text.

use muninn::vm::VmOptions;
use muninn::{HostPolicy, Vm, compile_to_bytecode};

fn vm_for(source: &str) -> Vm {
    let module = compile_to_bytecode(source).expect("source should compile");
    Vm::new_with_policy_and_options(module, HostPolicy::default(), VmOptions::default())
}

#[test]
fn captured_output_arrives_in_print_order() {
    let mut vm = vm_for("print(1); eprint(\"to stderr\"); print(2);");
    vm.capture_output();
    vm.run().expect("run should finish");

    let (out, err) = vm.take_output();
    assert_eq!(out.text, "1\n2\n");
    assert_eq!(err.text, "to stderr\n");
    assert!(out.tensor_lines.is_empty());
    assert!(err.tensor_lines.is_empty());
}

#[test]
fn taking_output_empties_the_capture_without_stopping_it() {
    let mut vm = vm_for("print(\"first\"); print(\"second\");");
    vm.capture_output();

    let mut collected = String::new();
    let mut takes = 0;
    loop {
        let finished = vm.step_instruction().expect("step should not fault");
        let (out, _) = vm.take_output();
        if !out.text.is_empty() {
            takes += 1;
        }
        collected.push_str(&out.text);
        if finished.is_some() {
            break;
        }
    }

    assert!(takes >= 2, "each print should be readable as it happens");
    assert_eq!(collected, "first\nsecond\n");
}

#[test]
fn a_capture_starts_empty_even_when_text_was_printed_before_it() {
    let mut vm = vm_for("print(\"before\");");
    vm.capture_output();
    assert_eq!(vm.take_output().0.text, "");
    vm.run().expect("run should finish");
    assert_eq!(vm.take_output().0.text, "before\n");
}

#[test]
fn an_inherited_stream_reports_nothing_because_nothing_was_held_back() {
    let mut vm = vm_for("print(\"to the terminal\");");
    vm.run().expect("run should finish");
    let (out, err) = vm.take_output();
    assert!(out.text.is_empty());
    assert!(err.text.is_empty());
    assert_eq!(out.dropped_lines, 0);
    assert_eq!(err.dropped_lines, 0);
}

#[test]
fn a_bounded_capture_keeps_whole_lines_and_reports_what_it_dropped() {
    let mut vm = vm_for("print(\"ab\"); print(\"cd\"); print(\"ef\");");
    vm.capture_output_bounded(7);
    vm.run().expect("run should finish");

    let (out, err) = vm.take_output();
    assert_eq!(out.text, "ab\ncd\n");
    assert_eq!(out.dropped_lines, 1);
    assert!(out.text.ends_with('\n'), "a kept line is never half a line");
    assert!(out.tensor_lines.is_empty());
    assert!(err.text.is_empty());

    let (out, _) = vm.take_output();
    assert!(out.text.is_empty());
    assert_eq!(out.dropped_lines, 0, "taking a drain resets its drop count");
}

#[test]
fn a_printed_tensor_is_marked_by_the_interpreter_not_guessed_from_text() {
    let mut vm = vm_for(
        "print(tensor_zeros(2)); eprint(tensor_zeros(2)); print(\"tensor(shape=[1], data=[0])\");",
    );
    vm.capture_output();
    vm.run().expect("run should finish");

    let (out, err) = vm.take_output();
    assert_eq!(out.tensor_lines, vec![1]);
    assert!(
        err.tensor_lines.is_empty(),
        "eprint always records a plain line"
    );
}

#[test]
fn a_stepped_frame_reports_its_locals_and_instruction_pointer() {
    let source = "fn twice(value: Int) -> Int { return value * 2; }\n\
                  let start: Int = 21;\n\
                  print(twice(start));\n";
    let module = compile_to_bytecode(source).expect("source should compile");
    let mut vm = Vm::new_with_policy_and_options(
        module.clone(),
        HostPolicy::default(),
        VmOptions::default(),
    );
    vm.capture_output();

    let mut saw_locals = false;
    let mut last_ip = 0;
    let mut steps = 0;
    loop {
        let finished = vm.step_instruction().expect("step should not fault");
        let frames = vm.frames();
        if let Some(frame) = frames.last()
            && module.functions[frame.function_id].name == "twice"
        {
            assert!(frame.ip >= last_ip, "a frame's pointer only moves forward");
            last_ip = frame.ip;
            let locals = vm.frame_locals(frames.len() - 1);
            if locals.len() == 1 && locals[0].stringify() == "21" {
                saw_locals = true;
            }
        }
        if finished.is_some() {
            break;
        }
        steps += 1;
        assert!(steps < 1000, "the program should terminate");
    }

    assert!(last_ip > 0, "twice should have run its own instructions");
    assert!(saw_locals, "its parameter should read back as 21");
    assert_eq!(vm.take_output().0.text, "42\n");
    assert!(
        vm.frames().is_empty(),
        "frames unwind when the program returns"
    );
}
