# Muninn 🐦‍⬛
<p>
  <img src="https://img.shields.io/badge/Rust-000000?logo=rust&logoColor=white">
  <img src="https://img.shields.io/badge/Language-Statically%20Typed-111111">
  <img src="https://img.shields.io/badge/Type-Scripting%20Language-111111">
  <img src="https://img.shields.io/badge/License-MIT-111111">
</p>

Muninn is a small statically typed scripting language implemented in Rust.

## Features

- Primitive types: `Int`, `Float`, `Bool`, `String`, `Void`
- `let` bindings with optional local type inference
- Mutable bindings via `mut`
- Top-level functions with typed parameters and explicit return types
- `if` statements and `while` loops
- Assignment and function calls
- Arithmetic, comparison, and logical operators
- String concatenation with `+`
- Builtins: `print`, `assert`

## Example

```muninn
fn add(a: Int, b: Int) -> Int {
    return a + b;
}

let mut total: Int = 0;
while (total < 3) {
    total = add(total, 1);
}

print(total);
total;
```

## Commands

Run demo:

```bash
cargo run
```

Run a file:

```bash
cargo run -- run examples/feature_tour.mun
```

Type-check a file:

```bash
cargo run -- check examples/feature_tour.mun
```

Run tests:

```bash
cargo test --workspace
```
