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

const USAGE: &str =
    "Usage:\n  muninn run <file>\n  muninn check <file>\n  muninn <file>\n  muninn --help";

fn main() {
    let mut args = env::args().skip(1).collect::<Vec<_>>();
    if matches!(args.first().map(String::as_str), Some("--help" | "-h")) {
        println!("{}", USAGE);
        return;
    }

    let (command, source) = match args.len() {
        0 => ("run".to_string(), DEMO_PROGRAM.to_string()),
        1 => {
            let arg = args.remove(0);
            match arg.as_str() {
                "run" | "check" => {
                    eprintln!("missing source file for command '{}'", arg);
                    eprintln!("{}", USAGE);
                    std::process::exit(1);
                }
                _ => ("run".to_string(), read_source(&arg)),
            }
        }
        2 => {
            let command = args.remove(0);
            let path = args.remove(0);
            match command.as_str() {
                "run" | "check" => (command, read_source(&path)),
                _ => {
                    eprintln!("unknown command '{}'.", command);
                    eprintln!("{}", USAGE);
                    std::process::exit(1);
                }
            }
        }
        _ => {
            eprintln!("too many arguments");
            eprintln!("{}", USAGE);
            std::process::exit(1);
        }
    };

    match command.as_str() {
        "run" => run_source(&source),
        "check" => check_source(&source),
        other => {
            eprintln!("unknown command '{}'.", other);
            eprintln!("{}", USAGE);
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
