use std::fs;

use muninn::{analyze_document, compile_and_run};

#[test]
fn all_examples_parse_check_and_run() {
    for path in [
        "examples/donut.mun",
        "examples/feature_tour.mun",
        "examples/perceptron.mun",
    ] {
        let source = fs::read_to_string(path).expect("example source");
        let analysis = analyze_document(&source);
        assert!(
            analysis.diagnostics.is_empty(),
            "expected no diagnostics for {path}, got: {:?}",
            analysis
                .diagnostics
                .iter()
                .map(|error| error.message.clone())
                .collect::<Vec<_>>()
        );
        compile_and_run(&source).expect("example run");
    }
}

#[test]
fn readme_example_runs() {
    let readme = fs::read_to_string("README.md").expect("readme");
    let start = readme
        .find("```muninn")
        .expect("README has muninn code block");
    let block = &readme[start + "```muninn".len()..];
    let end = block.find("```\n").expect("code block end");
    let source = block[..end].trim();

    let analysis = analyze_document(source);
    assert!(analysis.diagnostics.is_empty());
    compile_and_run(source).expect("README example run");
}
