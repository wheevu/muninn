//! Formatter idempotence over the real example corpus: formatting twice
//! yields the same text, and formatted output still compiles.

use muninn::{compile_to_bytecode, format_source};
use std::fs;

fn mun_sources() -> Vec<(String, String)> {
    let mut entries: Vec<_> = fs::read_dir("examples")
        .expect("examples dir")
        .filter_map(Result::ok)
        .filter(|entry| entry.path().extension().is_some_and(|ext| ext == "mun"))
        .collect();
    entries.sort_by_key(|entry| entry.path());
    entries
        .into_iter()
        .map(|entry| {
            let path = entry.path().to_string_lossy().into_owned();
            let source = fs::read_to_string(entry.path()).expect("read example");
            (path, source)
        })
        .collect()
}

#[test]
fn formatter_is_idempotent_on_the_example_corpus() {
    for (path, source) in mun_sources() {
        let once = format_source(&source);
        let twice = format_source(&once);
        assert_eq!(once, twice, "idempotent: {path}");
    }
}

#[test]
fn formatted_examples_still_compile() {
    for (path, source) in mun_sources() {
        let formatted = format_source(&source);
        compile_to_bytecode(&formatted)
            .unwrap_or_else(|_| panic!("formatted output compiles: {path}"));
    }
}
