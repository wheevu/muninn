use muninn::compile_and_run;

#[test]
fn runs_locked_syntax_surface() {
    let src = r#"
fn scale(x: Float, factor: Float) -> Float {
    x * factor
}

let mut total: Float = 0.0;
for i in 0..4 {
    total = scale(total + 1.0, 1.25);
}

let mut grid: Int[3, 3] = [0, 0, 0, 0, 0, 0, 0, 0, 0];
grid[1, 2] = 9;

let score: Float = total |> scale(1.0);
let msg: String = "score={score}, cell={grid[1, 2]}";
print(msg);
"#;

    assert!(compile_and_run(src).is_ok());
}

#[test]
fn runs_class_constructor_and_method() {
    let src = r#"
class Counter {
    let value: Int;

    fn init(start: Int) {
        self.value = start;
    }

    fn bump(delta: Int) -> Int {
        self.value = self.value + delta;
        self.value
    }
}

let counter: Counter = Counter(2);
let answer: Int = counter.bump(40);
print("answer={answer}");
"#;

    assert!(compile_and_run(src).is_ok());
}

#[test]
fn rejects_type_mismatch() {
    let src = r#"
let x: Int = 1;
let y: String = x;
"#;

    assert!(compile_and_run(src).is_err());
}

#[test]
fn runs_vectorized_array_ops_and_scalar_promotion() {
    let src = r#"
let a: Float[3] = [1.0, 2.0, 3.0];
let b: Float[3] = [0.5, 1.0, 1.5];
let c: Float[3] = a + b;
let d: Float[3] = c * 2.0;
let e: Float[3] = 0.5 * d;
print("vec={e[0]}, {e[1]}, {e[2]}");
"#;

    assert!(compile_and_run(src).is_ok());
}

#[test]
fn enforces_strict_scalar_promotion() {
    let src = r#"
let a: Float[3] = [10.0, 20.0, 30.0];
let b: Float[3] = a / 255;
"#;

    assert!(compile_and_run(src).is_err());
}

#[test]
fn propagates_none_with_try_operator() {
    let src = r#"
fn checked(v: Float) -> Option[Float] {
    if (v > 0.0) { __some(v) } else { __none }
}

fn probe(v: Float) -> Option[Float] {
    let x: Float = checked(v)?;
    __some(__unwrap(__none))
}

let out: Option[Float] = probe(-1.0);
print("out={out}");
"#;

    assert!(compile_and_run(src).is_ok());
}
