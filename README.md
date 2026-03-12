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
fn abs_int(value: Int) -> Int {
    if (value < 0) {
        return -value;
    }
    return value;
}

fn gcd(a: Int, b: Int) -> Int {
    let mut x: Int = abs_int(a);
    let mut y: Int = abs_int(b);
    while (y != 0) {
        let quotient: Int = x / y;
        let remainder: Int = x - quotient * y;
        x = y;
        y = remainder;
    }
    return x;
}

fn lcm(a: Int, b: Int) -> Int {
    let divisor: Int = gcd(a, b);
    return (a / divisor) * b;
}

let divisor: Int = gcd(84, 30);
let multiple: Int = lcm(84, 30);
assert(divisor == 6);
assert(multiple == 420);
print(divisor);
print(multiple);
divisor;
```

## Commands

Run demo:

```bash
cargo run
```

Run a file:

```bash
cargo run -- run examples/dsa_euclid.mun
```

Type-check a file:

```bash
cargo run -- check examples/dsa_euclid.mun
```

Run tests:

```bash
cargo test --workspace
```
