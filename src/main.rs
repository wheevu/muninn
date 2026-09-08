use std::io::{IsTerminal, Read};
use std::path::{Path, PathBuf};
use std::{env, fs};

use muninn::{
    BytecodeDecodeError, FileDiagnostic, HostPolicy, LoadedProject, MUBC_VERSION,
    compile_loaded_project, compile_to_bytecode, decode_bytecode_module, encode_bytecode_module,
    format_source, load_project, parse_exit_code, run_bytecode_module_with_options,
    run_bytecode_module_with_policy, vm::VmOptions,
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
  muninn run [--jit] [--jit-threshold N] [--allow-fs ROOT | --no-fs] [--allow-env NAME]... <file> [-- args...]
  muninn check <file>
  muninn build <file> [-o output.mubc]
  muninn run-bc [--jit] [--jit-threshold N] [--allow-fs ROOT | --no-fs] [--allow-env NAME]... <file.mubc> [-- args...]
  muninn fmt [--check] <file>
  muninn [--jit] [--jit-threshold N] <file>
  muninn --help

Exit codes: 0 success, 1 runtime trap, 2 usage/load/typecheck/compile/bytecode
error, 0-125 script exit(code). 126/127/128+N are never emitted by the VM.";

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
            let request = parse_file_command(&args, "run");
            run_file(&request);
        }
        "check" => {
            let path = expect_single_path(&args, "check");
            check_file(path);
        }
        "build" => build_source(&args),
        "fmt" => format_source_command(&args),
        "run-bc" => {
            let request = parse_file_command(&args, "run-bc");
            run_bc_file(&request);
        }
        "--jit" | "--jit-threshold" => {
            let mut run_args = vec![command];
            run_args.extend(args);
            let request = parse_file_command(&run_args, "run");
            run_file(&request);
        }
        other if args.is_empty() => {
            // Bare `muninn <file>`: trusted-CLI defaults, no script args.
            let request = FileRequest {
                options: VmOptions::default(),
                fs: FsGrant::Default,
                env_names: None,
                entry: other.to_string(),
                argv: Vec::new(),
            };
            run_file(&request);
        }
        other => {
            eprintln!("unknown command '{}'.", other);
            eprintln!("{}", USAGE);
            std::process::exit(2);
        }
    }
}

/// A file command after CLI parsing: the entry artifact, the VM options,
/// the capability grants, and the script arguments.
struct FileRequest {
    options: VmOptions,
    fs: FsGrant,
    env_names: Option<Vec<String>>,
    entry: String,
    argv: Vec<String>,
}

#[derive(Debug, Clone)]
enum FsGrant {
    /// Scope filesystem access under the entry file's directory.
    Default,
    /// Scope filesystem access under an explicit root.
    Root(PathBuf),
    /// Deny all filesystem access.
    Deny,
}

/// Parses `<options> <entry> [-- script-args...]`. Tokens after the entry
/// file (or after `--`) are script arguments, never CLI options, so user
/// flags starting with `-` survive via `--`.
fn parse_file_command(args: &[String], command: &str) -> FileRequest {
    let (left, mut argv) = split_script_args(args);
    let mut options = VmOptions::default();
    let mut fs = FsGrant::Default;
    let mut env_names: Option<Vec<String>> = None;
    let mut positional: Vec<String> = Vec::new();
    let mut index = 0;
    while index < left.len() {
        match left[index].as_str() {
            "--jit" => {
                options.jit_enabled = true;
                index += 1;
            }
            "--jit-threshold" => {
                let Some(value) = left.get(index + 1) else {
                    eprintln!("missing value for --jit-threshold");
                    eprintln!("{}", USAGE);
                    std::process::exit(2);
                };
                options.hot_loop_threshold = value.parse().unwrap_or_else(|_| {
                    eprintln!("invalid --jit-threshold value '{}'", value);
                    eprintln!("{}", USAGE);
                    std::process::exit(2);
                });
                index += 2;
            }
            "--allow-fs" => {
                let Some(root) = left.get(index + 1) else {
                    eprintln!("missing value for --allow-fs");
                    eprintln!("{}", USAGE);
                    std::process::exit(2);
                };
                fs = FsGrant::Root(PathBuf::from(root));
                index += 2;
            }
            "--no-fs" => {
                fs = FsGrant::Deny;
                index += 1;
            }
            "--allow-env" => {
                let Some(name) = left.get(index + 1) else {
                    eprintln!("missing value for --allow-env");
                    eprintln!("{}", USAGE);
                    std::process::exit(2);
                };
                env_names.get_or_insert_with(Vec::new).push(name.clone());
                index += 2;
            }
            value if value.starts_with("--") => {
                eprintln!("unknown option '{}' for command '{}'", value, command);
                eprintln!("{}", USAGE);
                std::process::exit(2);
            }
            value => {
                positional.push(value.to_string());
                index += 1;
            }
        }
    }
    if positional.is_empty() {
        eprintln!("missing source file for command '{}'", command);
        eprintln!("{}", USAGE);
        std::process::exit(2);
    }
    let entry = positional.remove(0);
    // Bare tokens after the entry are script arguments too.
    argv.splice(0..0, positional);
    FileRequest {
        options,
        fs,
        env_names,
        entry,
        argv,
    }
}

/// Splits CLI tokens at the first `--`: everything after is script argv.
fn split_script_args(args: &[String]) -> (Vec<String>, Vec<String>) {
    match args.iter().position(|arg| arg == "--") {
        Some(at) => (args[..at].to_vec(), args[at + 1..].to_vec()),
        None => (args.to_vec(), Vec::new()),
    }
}

fn expect_single_path<'a>(args: &'a [String], command: &str) -> &'a str {
    if args.len() != 1 {
        eprintln!("missing source file for command '{}'", command);
        eprintln!("{}", USAGE);
        std::process::exit(2);
    }
    &args[0]
}

/// Builds the trusted-CLI host policy: script argv, piped stdin, env
/// (allow-all unless `--allow-env` restricted it), and filesystem scoped
/// under `base_dir` unless overridden or denied.
fn cli_policy(
    base_dir: &Path,
    argv: Vec<String>,
    fs: &FsGrant,
    env_names: Option<Vec<String>>,
) -> HostPolicy {
    let mut policy = HostPolicy::default();
    policy.set_argv(argv);
    if let Some(names) = env_names {
        policy.allowed_env = Some(names.into_iter().collect());
    }
    match fs {
        FsGrant::Default => policy.set_fs_root(base_dir.to_path_buf()),
        FsGrant::Root(root) => policy.set_fs_root(root.clone()),
        FsGrant::Deny => {}
    }
    policy.set_stdin(read_piped_stdin());
    policy
}

/// Reads piped stdin fully. A terminal yields an empty string instead of
/// blocking: scripts probe with `stdin_read() == ""` for "no input".
fn read_piped_stdin() -> String {
    if std::io::stdin().is_terminal() {
        return String::new();
    }
    let mut data = String::new();
    if std::io::stdin().read_to_string(&mut data).is_err() {
        return String::new();
    }
    data
}

fn base_dir_for(entry: &Path, project: &LoadedProject) -> PathBuf {
    project
        .entry
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| {
            entry
                .parent()
                .map(Path::to_path_buf)
                .unwrap_or_else(|| PathBuf::from("."))
        })
}

fn run_file(request: &FileRequest) {
    let entry = Path::new(&request.entry);
    let project = load_project(entry).unwrap_or_else(|errors| exit_with_project_errors(errors));
    let policy = cli_policy(
        &base_dir_for(entry, &project),
        request.argv.clone(),
        &request.fs,
        request.env_names.clone(),
    );
    // Content-addressed bytecode cache: the key is a hash of the version
    // plus the combined source, so a hit can only serve identical code.
    // Anything the validator rejects falls back to a fresh compile.
    if let Some(module) = cached_module(&project) {
        let value = run_bytecode_module_with_policy(module, policy, request.options)
            .map_err(|errors| project.map_errors(errors))
            .unwrap_or_else(|errors| exit_with_project_errors(errors));
        println!("=> {}", value);
        return;
    }
    let module =
        compile_loaded_project(&project).unwrap_or_else(|errors| exit_with_project_errors(errors));
    store_cached_module(&project, &module);
    let value = run_bytecode_module_with_policy(module, policy, request.options)
        .map_err(|errors| project.map_errors(errors))
        .unwrap_or_else(|errors| exit_with_project_errors(errors));
    println!("=> {}", value);
}

/// FNV-1a 64-bit over the artifact version plus the combined source.
/// Deterministic across runs with no new dependencies.
fn project_digest(project: &LoadedProject) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;
    let mut hash = FNV_OFFSET;
    for byte in MUBC_VERSION
        .to_le_bytes()
        .iter()
        .chain(env!("CARGO_PKG_VERSION").as_bytes())
        .chain(project.combined.as_bytes())
    {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

fn cache_path(digest: u64) -> PathBuf {
    let mut dir = std::env::temp_dir();
    dir.push("muninn-cache");
    dir.push(format!("{:016x}.mubc", digest));
    dir
}

fn cached_module(project: &LoadedProject) -> Option<muninn::BytecodeModule> {
    let bytes = fs::read(cache_path(project_digest(project))).ok()?;
    decode_bytecode_module(&bytes).ok()
}

fn store_cached_module(project: &LoadedProject, module: &muninn::BytecodeModule) {
    let bytes = match encode_bytecode_module(module) {
        Ok(bytes) => bytes,
        Err(_) => return,
    };
    let path = cache_path(project_digest(project));
    if let Some(parent) = path.parent()
        && fs::create_dir_all(parent).is_err()
    {
        return;
    }
    // Same-dir temp plus rename: readers never see a partial artifact.
    // Cache failures are silent: `run` always works without the cache.
    let temp = path.with_extension(format!("tmp-{}", std::process::id()));
    if fs::write(&temp, &bytes).is_err() {
        return;
    }
    if fs::rename(&temp, &path).is_err() {
        let _ = fs::remove_file(&temp);
    }
}

fn run_bc_file(request: &FileRequest) {
    let path = Path::new(&request.entry);
    let base = path
        .canonicalize()
        .ok()
        .and_then(|full| full.parent().map(Path::to_path_buf))
        .or_else(|| {
            path.parent()
                .map(Path::to_path_buf)
                .or_else(|| Some(PathBuf::from(".")))
        })
        .expect("base dir");
    let policy = cli_policy(
        &base,
        request.argv.clone(),
        &request.fs,
        request.env_names.clone(),
    );
    let bytes = read_bytecode(&request.entry);
    match decode_and_run_bytecode(&bytes, request.options, policy) {
        Ok(value) => println!("=> {}", value),
        Err(message) => {
            if let Some(code) = exit_code_from_joined(&message) {
                std::process::exit(code);
            }
            eprintln!("{}", message);
            std::process::exit(1);
        }
    }
}

/// Scans joined runtime output for an `exit(code)` trap message.
fn exit_code_from_joined(message: &str) -> Option<i32> {
    let at = message.find("exit code ")?;
    let code: i64 = message[at + "exit code ".len()..]
        .split(|ch: char| !ch.is_ascii_digit() && ch != '-')
        .next()?
        .parse()
        .ok()?;
    Some(clamp_exit_code(code))
}

/// Clamps script exit codes to 0-125. Anything outside (and any trap
/// that is not `exit`) becomes 1; 126/127/128+N stay reserved for the
/// shell and signals, never emitted by the VM.
fn clamp_exit_code(code: i64) -> i32 {
    if (0..=125).contains(&code) {
        code as i32
    } else {
        1
    }
}

fn exit_with_project_errors(errors: Vec<FileDiagnostic>) -> ! {
    // A lone `exit(code)` trap becomes the process status.
    if errors.len() == 1
        && let Some(code) = parse_exit_code(&errors[0].error.message)
    {
        std::process::exit(clamp_exit_code(code));
    }
    for error in &errors {
        eprintln!("{}", error.render());
    }
    // Typecheck/parser failures (actionable source errors) are usage
    // class; vm-phase traps exit 1. The phase tag decides.
    let usage = errors.iter().any(|error| error.error.phase != "vm");
    std::process::exit(if usage { 2 } else { 1 });
}

fn build_source(args: &[String]) {
    if args.is_empty() {
        eprintln!("missing source file for command 'build'");
        eprintln!("{}", USAGE);
        std::process::exit(2);
    }

    let source_path = &args[0];
    let output_path = match args {
        [_] => default_bytecode_output_path(source_path),
        [_, flag, output] if flag == "-o" => PathBuf::from(output),
        _ => {
            eprintln!("invalid arguments for command 'build'");
            eprintln!("{}", USAGE);
            std::process::exit(2);
        }
    };

    let entry = Path::new(source_path);
    let project = load_project(entry).unwrap_or_else(|errors| exit_with_project_errors(errors));
    let module =
        compile_loaded_project(&project).unwrap_or_else(|errors| exit_with_project_errors(errors));
    let bytes = match encode_bytecode_module(&module) {
        Ok(bytes) => bytes,
        Err(error) => {
            eprintln!("failed to encode bytecode: {error}");
            std::process::exit(1);
        }
    };
    if let Err(error) = atomic_write(&output_path, &bytes) {
        eprintln!(
            "failed to write bytecode '{}': {}",
            output_path.display(),
            error
        );
        std::process::exit(1);
    }
    println!("{}", output_path.display());
}

/// Same-dir temp plus rename: readers never see a partial artifact.
fn atomic_write(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }
    let temp = path.with_extension(format!("tmp-{}", std::process::id()));
    fs::write(&temp, bytes)?;
    match fs::rename(&temp, path) {
        Ok(()) => Ok(()),
        Err(error) => {
            let _ = fs::remove_file(&temp);
            Err(error)
        }
    }
}

fn default_bytecode_output_path(source_path: &str) -> PathBuf {
    let path = Path::new(source_path);
    let mut output = path.to_path_buf();
    output.set_extension("mubc");
    output
}

/// Formats a source file in place. With `--check`, reports whether the
/// file is already formatted instead of writing.
fn format_source_command(args: &[String]) {
    let mut check = false;
    let mut paths = Vec::new();
    for arg in args {
        if arg == "--check" {
            check = true;
        } else if arg.starts_with("--") {
            eprintln!("unknown option '{}' for command 'fmt'", arg);
            eprintln!("{}", USAGE);
            std::process::exit(2);
        } else {
            paths.push(arg.clone());
        }
    }
    if paths.len() != 1 {
        eprintln!("missing source file for command 'fmt'");
        eprintln!("{}", USAGE);
        std::process::exit(2);
    }
    let source = read_source(&paths[0]);
    let formatted = format_source(&source);
    if formatted == source {
        println!("{}: already formatted", paths[0]);
        return;
    }
    if check {
        eprintln!("{}: would reformat", paths[0]);
        std::process::exit(1);
    }
    if let Err(error) = fs::write(&paths[0], formatted) {
        eprintln!("failed to write '{}': {}", paths[0], error);
        std::process::exit(1);
    }
    println!("{}", paths[0]);
}

fn read_source(path: &str) -> String {
    match fs::read_to_string(path) {
        Ok(contents) => contents,
        Err(error) => {
            eprintln!("failed to read '{path}': {error}");
            std::process::exit(2);
        }
    }
}

fn read_bytecode(path: &str) -> Vec<u8> {
    match fs::read(path) {
        Ok(bytes) => bytes,
        Err(error) => {
            eprintln!("failed to read '{path}': {error}");
            std::process::exit(2);
        }
    }
}

fn run_source(source: &str, options: VmOptions) {
    match compile_to_bytecode(source)
        .and_then(|module| run_bytecode_module_with_options(module, options))
    {
        Ok(value) => println!("=> {}", value),
        Err(errors) => {
            let usage = errors.iter().any(|error| error.phase != "vm");
            for error in errors {
                eprintln!("{}", error.render_with_source(source));
            }
            std::process::exit(if usage { 2 } else { 1 });
        }
    }
}

fn decode_and_run_bytecode(
    bytes: &[u8],
    options: VmOptions,
    policy: HostPolicy,
) -> Result<muninn::Value, String> {
    let module = decode_bytecode_module(bytes).map_err(render_decode_error)?;
    run_bytecode_module_with_policy(module, policy, options).map_err(|errors| {
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

fn check_file(source_path: &str) {
    let entry = Path::new(source_path);
    match load_project(entry).and_then(|project| compile_loaded_project(&project).map(|_| project))
    {
        Ok(_) => println!("ok"),
        Err(errors) => exit_with_project_errors(errors),
    }
}
