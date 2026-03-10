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
    if (v > 0.0) { some(v) } else { none }
}

fn probe(v: Float) -> Option[Float] {
    let x: Float = checked(v)?;
    some(unwrap(none))
}

let out: Option[Float] = probe(-1.0);
print("out={out}");
"#;

    assert!(compile_and_run(src).is_ok());
}

#[test]
fn rejects_intrinsic_shadowing() {
    let src = r#"
let none: Int = 1;
"#;

    assert!(compile_and_run(src).is_err());
}

#[test]
fn supports_string_relational_comparisons() {
    let src = r#"
if ("alpha" < "beta") { 1 } else { unwrap(none) };
if ("beta" > "alpha") { 1 } else { unwrap(none) };
"#;

    assert!(compile_and_run(src).is_ok());
}

#[test]
fn supports_ml_array_builtins() {
    let src = r#"
let a: Float[3] = [1.0, 2.0, 3.0];
let b: Float[3] = ones(3);
let c: Float[3] = a + b;
let d: Float = sum(c);
let e: Float = dot(c, [2.0, 2.0, 2.0]);
let n: Int = len(c);
print("d={d}, e={e}, n={n}");
"#;

    assert!(compile_and_run(src).is_ok());
}

#[test]
fn supports_grid_property_indexing() {
    let src = r#"
class GridHolder {
    let grid: Int[2, 2];

    fn init() {
        self.grid = [0, 0, 0, 0];
    }

    fn poke() {
        self.grid[1, 1] = 7;
        print(self.grid[1, 1]);
    }
}

let holder: GridHolder = GridHolder();
holder.poke();
"#;

    assert!(compile_and_run(src).is_ok());
}

#[test]
fn infers_local_binding_types() {
    let src = r#"
let x = 2;
let y = x + 3;
let z: Int = y;
print("z={z}");
"#;

    assert!(compile_and_run(src).is_ok());
}

#[test]
fn infers_constructor_type_for_local_binding() {
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

let counter = Counter(1);
let out: Int = counter.bump(2);
print("out={out}");
"#;

    assert!(compile_and_run(src).is_ok());
}

#[test]
fn supports_grid_property_indexing_with_inferred_instance_type() {
    let src = r#"
class GridHolder {
    let grid: Int[2, 2];

    fn init() {
        self.grid = [0, 0, 0, 0];
    }
}

let holder = GridHolder();
holder.grid[1, 0] = 5;
print(holder.grid[1, 0]);
"#;

    assert!(compile_and_run(src).is_ok());
}

#[test]
fn len_accepts_strings() {
    let src = r#"
let n: Int = len("muninn");
print(n);
"#;

    assert!(compile_and_run(src).is_ok());
}

#[test]
fn handles_deep_recursion_without_host_stack_overflow() {
    let src = r#"
fn down(n: Int) -> Int {
    if (n == 0) { 0 } else { down(n - 1) }
}

let out: Int = down(5000);
print(out);
"#;

    assert!(compile_and_run(src).is_ok());
}

#[test]
fn supports_nested_grid_property_indexing() {
    let src = r#"
class GridHolder {
    let grid: Int[2, 2];

    fn init() {
        self.grid = [0, 0, 0, 0];
    }
}

class Wrapper {
    let holder: GridHolder;

    fn init() {
        self.holder = GridHolder();
    }
}

let wrapper = Wrapper();
wrapper.holder.grid[1, 1] = 9;
print(wrapper.holder.grid[1, 1]);
"#;

    assert!(compile_and_run(src).is_ok());
}
