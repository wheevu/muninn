# Muninn 🐦‍⬛

Muninn is a statically typed, expression-oriented language implemented in Rust.
It compiles to bytecode and runs on a stack VM. 🤸🏻

Pipeline:

`source -> lexer -> parser -> desugar -> typecheck -> lower -> bytecode -> VM`

## Key features

- Typed declarations with optional local inference: `let x: Int = 5;`, `let y = 5;`
- Classes with fields, methods, and `init`
- Expression-based `if`/blocks and `unless`
- Pipeline operator: `x |> f(y)`
- Native 2D grid syntax: `Int[5, 5]`, `grid[x, y]`
- Range loops: `for i in 0..10 { ... }`
- String interpolation: `"value={x}"`
- Option propagation: `Option[T]`, `expr?`, `some(value)`, `none`
- Vectorized math for arrays:
  - same-shape array ops (`+`, `-`, `*`, `/`)
  - strict scalar promotion (`array * 2.0`, `2.0 * array`)

## Example

Example file: `examples/feature_tour.mun`

```muninn
fn checked_scale(scale: Float) -> Option[Float] {
    if (scale == 0.0) { none } else { some(scale) }
}

class Perceptron {
    let weights: Float[3];
    let bias: Float;

    fn init(weights: Float[3], bias: Float) {
        self.weights = weights;
        self.bias = bias;
    }

    fn forward(raw: Float[3], scale: Float) -> Option[Float] {
        let s: Float = checked_scale(scale)?;
        let normalized: Float[3] = raw / s;
        let weighted: Float[3] = normalized * self.weights;
        let shifted: Float[3] = 0.9 * (weighted + 0.05);
        let score: Float = shifted[0] + shifted[1] + shifted[2] + self.bias;
        some(unless (score > 0.0) { 0.0 } else { 1.0 })
    }
}

let mut grid: Int[2, 2] = [0, 0, 0, 0];
for i in 0..2 {
    grid[i, i] = i + 1;
}

let p: Perceptron = Perceptron([0.2, -0.5, 0.1], 0.3);
let output = [210.0, 140.0, 70.0] |> p.forward(255.0);
print("output={output}, grid00={grid[0, 0]}");
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
