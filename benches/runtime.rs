use criterion::{Criterion, black_box, criterion_group, criterion_main};
use muninn::vm::VmOptions;
use muninn::{
    compile_and_run, compile_to_bytecode, run_bytecode_module, run_bytecode_module_with_options,
};

fn bench_scalar_loop(c: &mut Criterion) {
    let source = r#"
let mut total: Int = 0;
while (total < 2000) {
    total = total + 1;
}
total;
"#;
    c.bench_function("scalar_loop", |b| {
        b.iter(|| compile_and_run(black_box(source)).expect("scalar loop"))
    });
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
    c.bench_function("vm_only_scalar_loop_interpreter", |b| {
        b.iter(|| run_bytecode_module(black_box(module.clone())).expect("vm scalar loop"))
    });
    c.bench_function("vm_only_scalar_loop_jit_cold", |b| {
        b.iter(|| {
            run_bytecode_module_with_options(
                black_box(module.clone()),
                VmOptions {
                    jit_enabled: true,
                    hot_loop_threshold: 8,
                },
            )
            .expect("jit scalar loop")
        })
    });
}

fn bench_native_calls(c: &mut Criterion) {
    let source = r#"
let base: Tensor = tensor_fill(64, 64, 1.0);
let total: Float = tensor_sum(base);
total;
"#;
    c.bench_function("native_call", |b| {
        b.iter(|| compile_and_run(black_box(source)).expect("native call"))
    });
}

fn bench_tensor_elementwise(c: &mut Criterion) {
    let source = r#"
let left: Tensor = tensor_fill(64, 64, 1.5);
let right: Tensor = tensor_fill(1, 64, 0.5);
let combined: Tensor = left + right;
tensor_sum(combined);
"#;
    c.bench_function("tensor_elementwise", |b| {
        b.iter(|| compile_and_run(black_box(source)).expect("tensor elementwise"))
    });
}

fn bench_tensor_matmul(c: &mut Criterion) {
    let source = r#"
let left: Tensor = tensor_fill(32, 32, 1.0);
let right: Tensor = tensor_fill(32, 32, 2.0);
let product: Tensor = tensor_matmul(left, right);
tensor_sum(product);
"#;
    c.bench_function("tensor_matmul", |b| {
        b.iter(|| compile_and_run(black_box(source)).expect("tensor matmul"))
    });
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
