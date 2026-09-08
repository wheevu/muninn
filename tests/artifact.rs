//! Bytecode artifact differential: validated-on-load, roundtrip-exact,
//! mutation-rejecting.
//!
//! The contract under test: compiler output always validates; encoding then
//! decoding then validating accepts it; any corruption (truncation, version
//! bump, trailing bytes, bit flips) fails decode or validation and never
//! reaches the VM. This is the checked-in seed corpus for future
//! structure-aware fuzzing: real compiled programs, not random bytes.

use muninn::{
    BytecodeModule, compile_and_run, compile_to_bytecode, decode_bytecode_module,
    encode_bytecode_module, run_bytecode_module,
};
use std::fs;

fn seeds() -> Vec<(String, String)> {
    let mut seeds = Vec::new();
    let mut entries: Vec<_> = fs::read_dir("examples")
        .expect("examples dir")
        .filter_map(Result::ok)
        .filter(|entry| entry.path().extension().is_some_and(|ext| ext == "mun"))
        .collect();
    entries.sort_by_key(|entry| entry.path());
    for entry in entries {
        let path = entry.path().to_string_lossy().into_owned();
        let source = fs::read_to_string(entry.path()).expect("read seed");
        seeds.push((path, source));
    }
    assert!(!seeds.is_empty(), "seed corpus must not be empty");
    seeds
}

fn encode(module: &BytecodeModule) -> Vec<u8> {
    encode_bytecode_module(module).expect("encode seed module")
}

#[test]
fn compiler_output_always_validates_and_runs() {
    for (path, source) in seeds() {
        let module = compile_to_bytecode(&source).expect("seed compiles");
        muninn::bytecode::validate_module(&module)
            .unwrap_or_else(|_| panic!("compiler output validates: {path}"));
        let direct = compile_and_run(&source).expect("seed runs");
        let replayed = run_bytecode_module(module).expect("validated module runs");
        assert_eq!(
            direct.to_string(),
            replayed.to_string(),
            "same value direct vs bytecode: {path}"
        );
    }
}

#[test]
fn encode_decode_roundtrip_is_exact_and_rerunnable() {
    for (path, source) in seeds() {
        let module = compile_to_bytecode(&source).expect("seed compiles");
        let bytes = encode(&module);
        let decoded =
            decode_bytecode_module(&bytes).unwrap_or_else(|_| panic!("roundtrip decodes: {path}"));
        muninn::bytecode::validate_module(&decoded)
            .unwrap_or_else(|_| panic!("roundtrip validates: {path}"));
        let first = run_bytecode_module(module).expect("original runs");
        let second = run_bytecode_module(decoded).expect("roundtrip runs");
        assert_eq!(
            first.to_string(),
            second.to_string(),
            "roundtrip reruns: {path}"
        );
    }
}

#[test]
fn truncated_and_versioned_and_suffixed_artifacts_fail() {
    let (_, source) = seeds().into_iter().next().expect("seed");
    let bytes = encode(&compile_to_bytecode(&source).expect("seed compiles"));
    assert!(bytes.len() > 16, "seed artifact has substance");

    for end in [0, 4, 8, bytes.len() / 2, bytes.len() - 1] {
        assert!(
            decode_bytecode_module(&bytes[..end]).is_err(),
            "truncation at {end} fails"
        );
    }

    let mut suffixed = bytes.clone();
    suffixed.push(0x00);
    assert!(
        decode_bytecode_module(&suffixed).is_err(),
        "trailing byte fails"
    );

    // Version lives at bytes 4..6 (magic MUBC, then u16 LE version).
    let mut bumped = bytes.clone();
    bumped[4] = bumped[4].wrapping_add(1);
    assert!(
        decode_bytecode_module(&bumped).is_err(),
        "version bump fails"
    );
}

#[test]
fn bit_flips_never_produce_a_runnable_corrupt_module() {
    // Smallest seed keeps the sweep fast; fuel caps any mutant that still
    // validates but loops forever.
    let source = fs::read_to_string("examples/perceptron.mun").expect("perceptron seed");
    let bytes = encode(&compile_to_bytecode(&source).expect("seed compiles"));
    let policy = muninn::HostPolicy {
        max_steps: Some(10_000),
        ..muninn::HostPolicy::default()
    };
    // VM faults that validation claims are unreachable: if the validator
    // accepts a mutant, running it under fuel must never hit these.
    const DIVERGENT: [&str; 6] = [
        "stack underflow",
        "instruction pointer out of range",
        "invalid opcode",
        "invalid constant index",
        "invalid local slot",
        "invalid function id",
    ];
    let mut accepted = 0;
    let mut rejected = 0;
    // Deterministic sweep: flip one bit per byte position.
    for (index, byte) in bytes.iter().enumerate() {
        let mut mutated = bytes.clone();
        mutated[index] = byte ^ 0x01;
        match decode_bytecode_module(&mutated) {
            Err(_) => rejected += 1,
            Ok(module) => {
                if muninn::bytecode::validate_module(&module).is_err() {
                    rejected += 1;
                } else {
                    accepted += 1;
                    if let Err(errors) = muninn::run_bytecode_module_with_policy(
                        module,
                        policy.clone(),
                        muninn::vm::VmOptions::default(),
                    ) {
                        for error in &errors {
                            assert!(
                                !DIVERGENT
                                    .iter()
                                    .any(|marker| error.message.contains(marker)),
                                "accept-then-fault divergence at byte {index}: {error:?}"
                            );
                        }
                    }
                }
            }
        }
    }
    assert!(rejected > 0, "mutations get rejected");
    assert!(accepted > 0, "single-bit flips mostly stay benign");
}
