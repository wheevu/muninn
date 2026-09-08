use std::time::Duration;

use criterion::measurement::WallTime;
use criterion::{
    BenchmarkGroup, Criterion, Throughput, black_box, criterion_group, criterion_main,
};
use muninn::vm::VmOptions;
use muninn::{
    compile_and_run, compile_to_bytecode, run_bytecode_module, run_bytecode_module_with_options,
};

/// Tuned harness: fast benches stay CI-friendly (sample 100, warmup 2s,
/// measurement 5s). Throughput is reported in elements processed per
/// iteration where the workload size is trivially known.
fn tune_fast(group: &mut BenchmarkGroup<'_, WallTime>) {
    group
        .sample_size(100)
        .warm_up_time(Duration::from_secs(2))
        .measurement_time(Duration::from_secs(5));
}

fn bench_scalar_loop(c: &mut Criterion) {
    let source = r#"
let mut total: Int = 0;
while (total < 2000) {
    total = total + 1;
}
total;
"#;
    let mut group = c.benchmark_group("scalar");
    tune_fast(&mut group);
    // 2000 loop iterations per measured op.
    group.throughput(Throughput::Elements(2000));
    group.bench_function("scalar_loop", |b| {
        b.iter(|| black_box(compile_and_run(black_box(source)).expect("scalar loop")))
    });
    group.finish();
}

fn bench_vm_only_scalar_loop(c: &mut Criterion) {
    let source = r#"
fn count() -> Int {
    let mut total: Int = 0;
    while (total < 2000) {
        total = total + 1;
    }
    return total;
}
count();
"#;
    let module = compile_to_bytecode(source).expect("scalar loop module");
    let mut group = c.benchmark_group("vm");
    tune_fast(&mut group);
    group.throughput(Throughput::Elements(2000));
    group.bench_function("vm_only_scalar_loop_interpreter", |b| {
        b.iter(|| {
            black_box(run_bytecode_module(black_box(module.clone())).expect("vm scalar loop"))
        })
    });
    group.bench_function("vm_only_scalar_loop_jit_cold", |b| {
        b.iter(|| {
            black_box(
                run_bytecode_module_with_options(
                    black_box(module.clone()),
                    VmOptions {
                        jit_enabled: true,
                        hot_loop_threshold: 8,
                    },
                )
                .expect("jit scalar loop"),
            )
        })
    });
    group.finish();
}

fn bench_native_calls(c: &mut Criterion) {
    let source = r#"
let base: Tensor = tensor_fill(64, 64, 1.0);
let total: Float = tensor_sum(base);
total;
"#;
    let mut group = c.benchmark_group("native");
    tune_fast(&mut group);
    group.throughput(Throughput::Elements(64 * 64));
    group.bench_function("native_call", |b| {
        b.iter(|| black_box(compile_and_run(black_box(source)).expect("native call")))
    });
    group.finish();
}

fn bench_tensor_elementwise(c: &mut Criterion) {
    let source = r#"
let left: Tensor = tensor_fill(64, 64, 1.5);
let right: Tensor = tensor_fill(1, 64, 0.5);
let combined: Tensor = left + right;
tensor_sum(combined);
"#;
    let mut group = c.benchmark_group("elementwise");
    tune_fast(&mut group);
    group.throughput(Throughput::Elements(64 * 64));
    group.bench_function("tensor_elementwise", |b| {
        b.iter(|| black_box(compile_and_run(black_box(source)).expect("tensor elementwise")))
    });
    group.finish();
}

fn bench_tensor_matmul(c: &mut Criterion) {
    let source = r#"
let left: Tensor = tensor_fill(32, 32, 1.0);
let right: Tensor = tensor_fill(32, 32, 2.0);
let product: Tensor = tensor_matmul(left, right);
tensor_sum(product);
"#;
    let mut group = c.benchmark_group("matmul");
    tune_fast(&mut group);
    // 32x32 matmul performs 2 * 32^3 multiply-adds.
    group.throughput(Throughput::Elements(2 * 32 * 32 * 32));
    group.bench_function("tensor_matmul", |b| {
        b.iter(|| black_box(compile_and_run(black_box(source)).expect("tensor matmul")))
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_scalar_loop,
    bench_vm_only_scalar_loop,
    bench_native_calls,
    bench_tensor_elementwise,
    bench_tensor_matmul
);
criterion_main!(benches);
