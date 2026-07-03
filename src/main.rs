use std::path::{Path, PathBuf};
use std::{env, fs};

use muninn::{
    BytecodeDecodeError, analyze_document, compile_to_bytecode, decode_bytecode_module,
    encode_bytecode_module, run_bytecode_module, run_bytecode_module_with_options, vm::VmOptions,
};

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

const USAGE: &str = "Usage:
  muninn run [--jit] [--jit-threshold N] <file>
  muninn check <file>
  muninn build <file> [-o output.mubc]
  muninn run-bc [--jit] [--jit-threshold N] <file.mubc>
  muninn [--jit] [--jit-threshold N] <file>
  muninn --help";

fn main() {
    let mut args = env::args().skip(1).collect::<Vec<_>>();
    if matches!(args.first().map(String::as_str), Some("--help" | "-h")) {
        println!("{}", USAGE);
        return;
    }

    if args.is_empty() {
        run_source(DEMO_PROGRAM, VmOptions::default());
        return;
    }

    let command = args.remove(0);
    match command.as_str() {
        "run" => {
            let (options, paths) = parse_run_options(&args, "run");
            let path = expect_single_path(&paths, "run");
            run_source(&read_source(path), options);
        }
        "check" => {
            let path = expect_single_path(&args, "check");
            check_source(&read_source(path));
        }
        "build" => build_source(&args),
        "run-bc" => {
            let (options, paths) = parse_run_options(&args, "run-bc");
            let path = expect_single_path(&paths, "run-bc");
            run_bytecode_path(path, options);
        }
        "--jit" | "--jit-threshold" => {
            let mut run_args = vec![command];
            run_args.extend(args);
            let (options, paths) = parse_run_options(&run_args, "run");
            let path = expect_single_path(&paths, "run");
            run_source(&read_source(path), options);
        }
        other if args.is_empty() => run_source(&read_source(other), VmOptions::default()),
        other => {
            eprintln!("unknown command '{}'.", other);
            eprintln!("{}", USAGE);
            std::process::exit(1);
        }
    }
}

fn expect_single_path<'a>(args: &'a [String], command: &str) -> &'a str {
    if args.len() != 1 {
        eprintln!("missing source file for command '{}'", command);
        eprintln!("{}", USAGE);
        std::process::exit(1);
    }
    &args[0]
}

fn parse_run_options(args: &[String], command: &str) -> (VmOptions, Vec<String>) {
    let mut options = VmOptions::default();
    let mut paths = Vec::new();
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--jit" => {
                options.jit_enabled = true;
                index += 1;
            }
            "--jit-threshold" => {
                let Some(value) = args.get(index + 1) else {
                    eprintln!("missing value for --jit-threshold");
                    eprintln!("{}", USAGE);
                    std::process::exit(1);
                };
                options.hot_loop_threshold = value.parse().unwrap_or_else(|_| {
                    eprintln!("invalid --jit-threshold value '{}'", value);
                    eprintln!("{}", USAGE);
                    std::process::exit(1);
                });
                index += 2;
            }
            value if value.starts_with("--") => {
                eprintln!("unknown option '{}' for command '{}'", value, command);
                eprintln!("{}", USAGE);
                std::process::exit(1);
            }
            value => {
                paths.push(value.to_string());
                index += 1;
            }
        }
    }
    (options, paths)
}

fn build_source(args: &[String]) {
    if args.is_empty() {
        eprintln!("missing source file for command 'build'");
        eprintln!("{}", USAGE);
        std::process::exit(1);
    }

    let source_path = &args[0];
    let output_path = match args {
        [_] => default_bytecode_output_path(source_path),
        [_, flag, output] if flag == "-o" => PathBuf::from(output),
        _ => {
            eprintln!("invalid arguments for command 'build'");
            eprintln!("{}", USAGE);
            std::process::exit(1);
        }
    };

    let source = read_source(source_path);
    match compile_to_bytecode(&source) {
        Ok(module) => {
            let bytes = encode_bytecode_module(&module);
            if let Err(error) = fs::write(&output_path, bytes) {
                eprintln!(
                    "failed to write bytecode '{}': {}",
                    output_path.display(),
                    error
                );
                std::process::exit(1);
            }
            println!("{}", output_path.display());
        }
        Err(errors) => {
            for error in errors {
                eprintln!("{}", error.render_with_source(&source));
            }
            std::process::exit(1);
        }
    }
}

fn default_bytecode_output_path(source_path: &str) -> PathBuf {
    let path = Path::new(source_path);
    let mut output = path.to_path_buf();
    output.set_extension("mubc");
    output
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

fn read_bytecode(path: &str) -> Vec<u8> {
    match fs::read(path) {
        Ok(bytes) => bytes,
        Err(error) => {
            eprintln!("failed to read '{path}': {error}");
            std::process::exit(1);
        }
    }
}

fn run_source(source: &str, options: VmOptions) {
    match compile_to_bytecode(source)
        .and_then(|module| run_bytecode_module_with_options(module, options))
    {
        Ok(value) => println!("=> {}", value),
        Err(errors) => {
            for error in errors {
                eprintln!("{}", error.render_with_source(source));
            }
            std::process::exit(1);
        }
    }
}

fn run_bytecode_path(path: &str, options: VmOptions) {
    let bytes = read_bytecode(path);
    match decode_and_run_bytecode(&bytes, options) {
        Ok(value) => println!("=> {}", value),
        Err(error) => {
            eprintln!("{}", error);
            std::process::exit(1);
        }
    }
}

fn decode_and_run_bytecode(bytes: &[u8], options: VmOptions) -> Result<muninn::Value, String> {
    let module = decode_bytecode_module(bytes).map_err(render_decode_error)?;
    if options == VmOptions::default() {
        run_bytecode_module(module)
    } else {
        run_bytecode_module_with_options(module, options)
    }
    .map_err(|errors| {
        errors
            .into_iter()
            .map(|error| error.to_string())
            .collect::<Vec<_>>()
            .join("\n")
    })
}

fn render_decode_error(error: BytecodeDecodeError) -> String {
    format!("bytecode error: {}", error)
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
