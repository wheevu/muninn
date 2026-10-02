//! The bytecode listing a debugger shows must describe the same stream the
//! VM executes.
//!
//! `disassemble` re-implements the VM's operand walk in the display
//! direction, so the claim under test is that both cover exactly the same
//! bytes: every walk lands on instruction starts, and its last instruction
//! ends where the chunk does. A new opcode that forgets its operands fails
//! `operands_for` at compile time; one that lies about them fails here.

use muninn::bytecode::{Constant, Operands, disassemble, operands_for};
use muninn::{BytecodeModule, compile_to_bytecode, decode_bytecode_module, encode_bytecode_module};

const EXAMPLES: [&str; 4] = [
    "examples/dsa_euclid.mun",
    "examples/perceptron.mun",
    "examples/records.mun",
    "examples/tensor_pipeline.mun",
];

fn module_for(path: &str) -> BytecodeModule {
    let source = std::fs::read_to_string(path).expect("example source");
    compile_to_bytecode(&source)
        .unwrap_or_else(|errors| panic!("{path} should compile: {errors:?}"))
}

fn width_of(op: muninn::bytecode::OpCode) -> usize {
    match operands_for(op) {
        Operands::None => 1,
        Operands::U8 => 2,
        Operands::U16 | Operands::Name => 3,
    }
}

#[test]
fn every_walk_covers_its_chunk_exactly() {
    for path in EXAMPLES {
        let module = module_for(path);
        for function in &module.functions {
            let listing = disassemble(&function.chunk);
            assert!(
                !listing.is_empty(),
                "{path} function {} has no instructions",
                function.name
            );
            let last = listing.last().expect("non-empty listing");
            assert_eq!(
                last.ip + width_of(last.op),
                function.chunk.code.len(),
                "{path} function {} walked {} bytes of {}",
                function.name,
                last.ip + width_of(last.op),
                function.chunk.code.len()
            );
            for pair in listing.windows(2) {
                assert_eq!(
                    pair[0].ip + width_of(pair[0].op),
                    pair[1].ip,
                    "{path} function {} has a gap or overlap at {}",
                    function.name,
                    pair[1].ip
                );
            }
        }
    }
}

#[test]
fn a_listing_survives_the_bytecode_round_trip() {
    for path in EXAMPLES {
        let module = module_for(path);
        let bytes = encode_bytecode_module(&module).expect("encode");
        let decoded = decode_bytecode_module(&bytes).expect("decode");
        assert_eq!(
            module.functions.len(),
            decoded.functions.len(),
            "{path} lost functions"
        );
        for (original, restored) in module.functions.iter().zip(&decoded.functions) {
            assert_eq!(
                disassemble(&original.chunk),
                disassemble(&restored.chunk),
                "{path} function {} lists differently after a round trip",
                original.name
            );
        }
    }
}

#[test]
fn operands_carry_the_value_the_vm_would_read() {
    let module = compile_to_bytecode("let answer: Int = 42;\nanswer;\n").expect("compiles");
    let listing = &disassemble(&module.functions[module.entry_function].chunk);

    let constant = listing
        .iter()
        .find(|instruction| instruction.name == "constant")
        .expect("the literal is loaded as a constant");
    assert_eq!(
        constant.constant,
        Some(Constant::Int(42)),
        "a constant operand is a constant index, and the page shows what it points at"
    );

    let define = listing
        .iter()
        .find(|instruction| instruction.name == "define_global")
        .expect("the binding defines a global");
    assert_eq!(
        define.constant,
        Some(Constant::String("answer".to_string())),
        "a name operand resolves to the name the VM reads"
    );

    assert!(
        listing
            .iter()
            .any(|instruction| instruction.name == "get_global"
                && instruction.constant == Some(Constant::String("answer".to_string()))),
        "the tail expression reads the binding back"
    );
}

#[test]
fn a_truncated_chunk_stops_instead_of_panicking() {
    let module = compile_to_bytecode("let a: Int = 1;\n").expect("compiles");
    let chunk = &module.functions[module.entry_function].chunk;
    let mut broken = chunk.clone();
    broken.code.pop();
    broken.code.push(0xfe);
    let listing = disassemble(&broken);
    assert!(
        listing
            .iter()
            .all(|instruction| instruction.ip < broken.code.len()),
        "the walk stops before the invalid byte"
    );
}
