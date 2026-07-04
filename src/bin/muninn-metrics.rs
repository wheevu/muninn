use std::fs;
use std::io;
use std::time::{Duration, Instant};

use muninn::{compile_and_run, compile_to_bytecode, encode_bytecode_module, run_bytecode_module};

const SCALAR_LOOP: &str = r#"
fn count() -> Int {
    let mut total: Int = 0;
    while (total < 2000) {
        total = total + 1;
    }
    return total;
}
count();
"#;

fn main() -> io::Result<()> {
    let metrics = metrics_table()?;
    fs::create_dir_all("docs")?;
    fs::write("docs/metrics.md", &metrics)?;
    update_readme("README.md", &metrics)?;
    println!("wrote docs/metrics.md");
    Ok(())
}

fn metrics_table() -> io::Result<String> {
    let euclid = example_size("examples/dsa_euclid.mun")?;
    let tensor = example_size("examples/tensor_pipeline.mun")?;
    let perceptron = example_size("examples/perceptron.mun")?;
    let workspace_tests = count_workspace_tests()?;
    let benchmark_targets = count_benchmark_targets()?;
    let example_programs = count_example_programs()?;
    let compile_run = bench_compile_and_run(SCALAR_LOOP, 200);
    let vm_only = bench_vm_only(SCALAR_LOOP, 1_000);
    let tensor_source =
        fs::read_to_string("examples/tensor_pipeline.mun")?.replace("print(total);", "");
    let tensor_run = bench_compile_and_run(&tensor_source, 200);

    Ok(format!(
        "| metric | value |\n|---|---:|\n| workspace tests | {} |\n| benchmark targets | {} |\n| example programs | {} |\n| `dsa_euclid.mun` source | {} |\n| `dsa_euclid.mun` bytecode | {} |\n| `tensor_pipeline.mun` source | {} |\n| `tensor_pipeline.mun` bytecode | {} |\n| `perceptron.mun` source | {} |\n| `perceptron.mun` bytecode | {} |\n| scalar loop compile + run | {} µs/op |\n| scalar loop VM only | {} µs/op |\n| tensor pipeline compile + run | {} µs/op |\n",
        workspace_tests,
        benchmark_targets,
        example_programs,
        bytes(euclid.source_bytes),
        bytes(euclid.bytecode_bytes),
        bytes(tensor.source_bytes),
        bytes(tensor.bytecode_bytes),
        bytes(perceptron.source_bytes),
        bytes(perceptron.bytecode_bytes),
        micros_per_op(compile_run, 200),
        micros_per_op(vm_only, 1_000),
        micros_per_op(tensor_run, 200),
    ))
}

fn count_workspace_tests() -> io::Result<usize> {
    let mut count = 0;
    for root in ["src", "tests", "lsp/src"] {
        for path in rust_files(root)? {
            count += fs::read_to_string(path)?
                .lines()
                .filter(|line| line.trim() == "#[test]")
                .count();
        }
    }
    Ok(count)
}

fn count_benchmark_targets() -> io::Result<usize> {
    let mut count = 0;
    for path in rust_files("benches")? {
        count += fs::read_to_string(path)?.matches("bench_function(").count();
    }
    Ok(count)
}

fn count_example_programs() -> io::Result<usize> {
    Ok(fs::read_dir("examples")?
        .filter_map(Result::ok)
        .filter(|entry| {
            entry
                .path()
                .extension()
                .is_some_and(|extension| extension == "mun")
        })
        .count())
}

fn rust_files(root: &str) -> io::Result<Vec<std::path::PathBuf>> {
    let mut files = Vec::new();
    let mut pending = vec![std::path::PathBuf::from(root)];
    while let Some(path) = pending.pop() {
        if path.is_dir() {
            for entry in fs::read_dir(path)? {
                pending.push(entry?.path());
            }
        } else if path.extension().is_some_and(|extension| extension == "rs") {
            files.push(path);
        }
    }
    Ok(files)
}

fn example_size(path: &str) -> io::Result<ExampleSize> {
    let source = fs::read_to_string(path)?;
    let module = compile_to_bytecode(&source).map_err(render_errors)?;
    let bytecode = encode_bytecode_module(&module);
    Ok(ExampleSize {
        source_bytes: source.len(),
        bytecode_bytes: bytecode.len(),
    })
}

fn bench_compile_and_run(source: &str, iterations: usize) -> Duration {
    let started = Instant::now();
    for _ in 0..iterations {
        std::hint::black_box(compile_and_run(std::hint::black_box(source)).expect("program runs"));
    }
    started.elapsed()
}

fn bench_vm_only(source: &str, iterations: usize) -> Duration {
    let module = compile_to_bytecode(source).expect("program compiles");
    let started = Instant::now();
    for _ in 0..iterations {
        std::hint::black_box(
            run_bytecode_module(std::hint::black_box(module.clone())).expect("program runs"),
        );
    }
    started.elapsed()
}

fn micros_per_op(duration: Duration, iterations: usize) -> u128 {
    duration.as_micros() / iterations as u128
}

fn bytes(value: usize) -> String {
    if value >= 1024 {
        format!("{:.1} KB", value as f64 / 1024.0)
    } else {
        format!("{value} B")
    }
}

fn render_errors(errors: Vec<muninn::error::MuninnError>) -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        errors
            .into_iter()
            .map(|error| error.message)
            .collect::<Vec<_>>()
            .join("\n"),
    )
}

fn update_readme(path: &str, metrics: &str) -> io::Result<()> {
    let readme = fs::read_to_string(path)?;
    let updated = replace_section(&readme, "metrics", metrics);
    fs::write(path, updated)
}

fn replace_section(readme: &str, name: &str, content: &str) -> String {
    let start = format!("<!-- {name}:start -->");
    let end = format!("<!-- {name}:end -->");
    let Some(start_index) = readme.find(&start) else {
        return readme.to_string();
    };
    let Some(end_index) = readme.find(&end) else {
        return readme.to_string();
    };
    let before = &readme[..start_index + start.len()];
    let after = &readme[end_index..];
    format!("{before}\n{}\n{after}", content.trim_end())
}

struct ExampleSize {
    source_bytes: usize,
    bytecode_bytes: usize,
}
