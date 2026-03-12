use criterion::{Criterion, black_box, criterion_group, criterion_main};
use muninn::compile_and_run;

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
    bench_native_calls,
    bench_tensor_elementwise,
    bench_tensor_matmul
);
criterion_main!(benches);
