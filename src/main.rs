use std::{env, fs};

use muninn::{analyze_document, compile_and_run};

const DEMO_PROGRAM: &str = r#"
fn add(a: Int, b: Int) -> Int {
    return a + b;
}

let mut total: Int = 0;
while (total < 3) {
    total = add(total, 1);
}

print("done");
print(total);
"#;

fn main() {
    let mut args = env::args().skip(1).collect::<Vec<_>>();
    let (command, source) = match args.len() {
        0 => ("run".to_string(), DEMO_PROGRAM.to_string()),
        1 => {
            let path = args.remove(0);
            ("run".to_string(), read_source(&path))
        }
        _ => {
            let command = args.remove(0);
            let path = args.remove(0);
            (command, read_source(&path))
        }
    };

    match command.as_str() {
        "run" => run_source(&source),
        "check" => check_source(&source),
        other => {
            eprintln!("unknown command '{other}'. use 'run <file>' or 'check <file>'");
            std::process::exit(1);
        }
    }
}

fn read_source(path: &str) -> String {
    match fs::read_to_string(path) {
        Ok(contents) => contents,
        Err(error) => {
            eprintln!("failed to read '{path}': {error}");
            std::process::exit(1);
        }
    }
}

fn run_source(source: &str) {
    match compile_and_run(source) {
        Ok(value) => println!("=> {}", value),
        Err(errors) => {
            for error in errors {
                eprintln!("{}", error.render_with_source(source));
            }
            std::process::exit(1);
        }
    }
}

fn check_source(source: &str) {
    let analysis = analyze_document(source);
    if analysis.diagnostics.is_empty() {
        println!("ok");
        return;
    }

    for error in analysis.diagnostics {
        eprintln!("{}", error.render_with_source(source));
    }
    std::process::exit(1);
}
