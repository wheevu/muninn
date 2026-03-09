# 🐦‍⬛ Muninn 

Muninn is a statically typed, expression-oriented language implemented in Rust.
It compiles to custom bytecode and runs on a custom stack VM.

Pipeline:

`source -> lexer -> parser -> desugar -> typecheck -> lower -> bytecode -> VM`

## Key features

- Mandatory typed declarations: `let x: Int = 5;`
- Classes with fields, methods, and `init`
- Expression-based `if`/blocks and `unless`
- Pipeline operator: `x |> f(y)`
- Native 2D grid syntax: `Int[5, 5]`, `grid[x, y]`
- Range loops: `for i in 0..10 { ... }`
- String interpolation: `"value={x}"`
- Option propagation: `Option[T]`, `expr?`
- Vectorized math for arrays:
  - same-shape array ops (`+`, `-`, `*`, `/`)
  - strict scalar promotion (`array * 2.0`, `2.0 * array`)

## Example

```muninn
fn checked_scale(scale: Float) -> Option[Float] {
    if (scale == 0.0) { __none } else { __some(scale) }
}

fn normalize(raw: Float[3], scale: Float) -> Option[Float[3]] {
    let s: Float = checked_scale(scale)?;
    __some(raw / s)
}

let features: Option[Float[3]] = normalize([210.0, 140.0, 70.0], 255.0);
print("features={features}");
```

## Quick start

```bash
cargo run
```

Run the ML demo:

```bash
cargo run -- examples/perceptron.mun
```

Run tests:

```bash
cargo test
```
