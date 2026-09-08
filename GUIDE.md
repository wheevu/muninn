# Muninn 🐦‍⬛
<p>
  <img src="https://img.shields.io/badge/Rust-000000?logo=rust&logoColor=white">
  <img src="https://img.shields.io/badge/Language-Statically%20Typed-111111">
  <img src="https://img.shields.io/badge/Type-Scripting%20Language-111111">
  <img src="https://img.shields.io/badge/License-MIT-111111">
</p>

Muninn is a small statically typed scripting language implemented in Rust.

## Quickstart

Requirements: a stable Rust toolchain with Rust 2024 edition support.
Node.js 20 or newer and npm are needed only to develop the VS Code extension.

Run the demo, execute a source file, or run the test suite:

```bash
cargo run
cargo run -- run examples/dsa_euclid.mun
cargo test --workspace
```

![Muninn compiler pipeline](docs/muninn-pipeline.svg)

## Features

### Language
- Primitive types: `Int`, `Float`, `Bool`, `String`, `Tensor`, and `Void`
- Nominal records: `record Point { x: Int, y: Int }`, construction with `Point { x: 1, y: 2 }`, field reads with `point.x`
- `let` bindings with optional local type inference, plus `mut` for mutable variables
- Functions with typed parameters and explicit return types
- Control flow with `if`, `while`, expression-valued blocks, and `if/else`
- Assignment, function calls, arithmetic, comparison, and logical operators
- String concatenation with `+`
- Tensor arithmetic with broadcasting and matrix multiplication

### Runtime
- Built-in functions for printing (`print` to stdout, `eprint` to stderr) and assertions
- Tensor built-ins for creating, reshaping, multiplying, and reducing tensors
- Capability-gated IO for real tools: `args_len`/`args_get`, `env_has`/`env_get`,
  `fs_exists`/`fs_read`/`fs_write`, `clock_ms`, `stdin_read`, `exit`
- Probes before traps: `args_len`, `env_has`, and `fs_exists` let scripts
  branch on absence; the accessors trap with a span when input is missing
- Sibling imports: `import "./util.mun";` at the top level (file-relative,
  `.mun` extension, cycles fail with the full chain)

### Tooling
- Type-check source files and multi-file projects
- Compile programs to `.mubc` bytecode (atomic writes, never partial)
- Run source files or precompiled bytecode, with a content-addressed
  bytecode cache for `run`
- Script arguments after the entry file (or after `--`), piped stdin via
  `stdin_read`, and exit codes: 0 success, 1 runtime trap, 2 usage/load/
  typecheck/compile/bytecode error, 0-125 script `exit(code)`
- Hot reload with global preservation at VM safe points
- Format sources with `muninn fmt` (`--check` verifies without writing)
- Embed untrusted scripts under a deny-by-default `HostPolicy`
  (builtin allowlist, deterministic fuel, wall-clock timeout,
  tensor cap, readonly globals, plus argv, env allowlist, fs root, and
  stdin grants with `revoke_fs` to shrink the surface after startup);
  see `examples/embed_host.rs`

### Performance
- Capacity reservation mode for allocation-free interpreter hot paths
- Optional tracing JIT for small hot `Int` loops, backed by Cranelift
- Global lookup cache with reload-safe invalidation

## Tensor path

Muninn has a small tensor runtime for typed numerical scripts.

![Tensor pipeline](docs/tensor-rune.svg)

For the compiler/runtime shape and current boundaries, see [Architecture](docs/architecture.md).

Tensor shape contract: `[]` is the scalar shape, while every dimension of a
non-scalar tensor must be positive. Zero-element tensors are rejected at the
constructor and runtime boundaries instead of being represented as a
one-element buffer.

## Differentiable tensor expressions

Muninn also exposes a small eager Rust API for reverse-mode differentiation.
It is intentionally an interpreter-only fallback: `grad` evaluates a captured
tensor tape and does not change source compilation, bytecode execution, hot
reload, LSP, or the optional integer JIT.

```rust
use muninn::{Tape, Tensor, grad};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let tape = Tape::new();
    let x = tape.variable(Tensor::scalar(3.0));
    let loss = x.mul(&x)?.sum()?;
    let dx = grad(&loss, &x)?;
    assert_eq!(dx.data(), &[6.0]);
    Ok(())
}
```

The graph supports eager broadcasting, elementwise add/subtract/multiply,
rank-2 matrix multiplication, full sums, and axis sums. A loss must be a
scalar tensor; shape mismatches and missing paths return structured,
shape-aware errors. See `examples/curve_fit.rs` for a small curve-fitting
example without any external numerical runtime.

## File tools

Muninn scripts can take arguments, read the environment, touch the
filesystem, read piped stdin, and set the process exit code.
Everything is host-granted: the CLI grants argv, piped stdin, env, and a
filesystem root scoped to the entry file's directory, while the
embedding API (`HostPolicy`) denies everything until enabled.
See `examples/project/` for a runnable demo.

```text
args_len() -> Int            // probe: use before args_get
args_get(index: Int) -> String
env_has(name: String) -> Bool // probe: use before env_get
env_get(name: String) -> String
fs_exists(path: String) -> Bool // probe: use before fs_read
fs_read(path: String) -> String
fs_write(path: String, data: String) -> Void
clock_ms() -> Int
stdin_read() -> String        // "" when stdin is a terminal
eprint(value) -> Void         // stderr, same overloads as print
exit(code: Int) -> Void       // process status 0-125, else 1
```

Filesystem paths are relative to the granted root: absolute paths,
escapes above the root, and `@` prefixes trap with a span.
Denied env names trap as policy errors; allowed-but-missing names report
`false` from `env_has` and trap from `env_get`.
There is no network, no subprocess, and no FFI: those would be
allow-all by another name.

## Imports

Projects span files with one statement, top-level only:

```text
import "./util.mun";
```

Paths resolve against the importing file, must start with `./` or `../`,
and must end with `.mun`.
Absolute paths, bare names, and `@` prefixes (reserved for future
package aliases) fail with an actionable diagnostic.
Dependencies load before the entry file, diamonds load once, and cycles
fail with the full chain (`a.mun -> b.mun -> a.mun`) instead of
recursing.
Type errors name the owning file.
Single-file compilation (`compile_to_bytecode`, the LSP) rejects imports
and says to run a file instead.

## Runtime modes

The same source can be checked, run directly, compiled to bytecode, or run with the experimental JIT.

![Runtime modes](docs/runtime-modes.svg)

## Benchmarks

| benchmark | what it measures |
|---|---|
| `scalar_loop` | compile + run scalar loop |
| `vm_only_scalar_loop_interpreter` | interpreter loop execution |
| `vm_only_scalar_loop_jit_cold` | JIT-enabled loop execution, including trace warmup |
| `native_call` | tensor builtin call overhead |
| `tensor_elementwise` | tensor broadcasting and addition |
| `tensor_matmul` | matrix multiply builtin |

## Metrics

Generated by `cargo run --release --bin muninn-metrics`. Wall-clock numbers are local release-mode measurements.

<!-- metrics:start -->
| metric | value |
|---|---:|
| workspace tests | 233 |
| benchmark targets | 6 |
| example programs | 4 |
| `dsa_euclid.mun` source | 691 B |
| `dsa_euclid.mun` bytecode | 6.2 KB |
| `tensor_pipeline.mun` source | 367 B |
| `tensor_pipeline.mun` bytecode | 3.6 KB |
| `perceptron.mun` source | 204 B |
| `perceptron.mun` bytecode | 1.7 KB |
| scalar loop compile + run | 580 µs/op |
| scalar loop VM only | 562 µs/op |
| tensor pipeline compile + run | 55 µs/op |
<!-- metrics:end -->

## Example

![Muninn example code](docs/example-code.svg)

<details>
  <summary>Copy the source</summary>

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

</details>

<details>
  <summary> Commands</summary>

Run demo:

```bash
cargo run
```

Run a file:

```bash
cargo run -- run examples/dsa_euclid.mun
```

Run a file tool with script arguments (everything after the entry file,
or after `--`, belongs to the script):

```bash
cargo run -- run examples/project/main.mun -- from-demo
```

Scope or deny capabilities (filesystem defaults to the entry file's
directory; env defaults to allow-all; `--allow-env` restricts to names):

```bash
cargo run -- run --allow-fs ./data --allow-env HOME examples/project/main.mun
cargo run -- run --no-fs examples/dsa_euclid.mun
```

Run with the experimental JIT:

```bash
cargo run --features jit -- run --jit examples/dsa_euclid.mun
```

Lower the hot-loop threshold while experimenting:

```bash
cargo run --features jit -- run --jit --jit-threshold 1 examples/dsa_euclid.mun
```

Type-check a file:

```bash
cargo run -- check examples/dsa_euclid.mun
```

Compile a source file to bytecode:

```bash
cargo run -- build examples/dsa_euclid.mun -o examples/dsa_euclid.mubc
```

Run a precompiled bytecode artifact:

```bash
cargo run -- run-bc examples/dsa_euclid.mubc
```

Artifacts are validated on load before anything runs; a corrupted or
incompatible `.mubc` file fails with a bytecode error instead of
executing. See [Architecture](docs/architecture.md) for the contract.

Run tests:

```bash
cargo test --workspace
```

Run benchmarks:

```bash
cargo bench --bench runtime
```

Run benchmarks with JIT support compiled in:

```bash
cargo bench --features jit --bench runtime
```

## Documentation

- [`docs/architecture.md`](docs/architecture.md): compiler, runtime, and project boundaries.
- [`docs/versioning.md`](docs/versioning.md): bytecode artifact and cache policy.
- [`corpus/README.md`](corpus/README.md): expressiveness corpus and the data-shape question.
- [`docs/metrics.md`](docs/metrics.md): generated project metrics and benchmark inputs.
- [`docs/muninn-pipeline.svg`](docs/muninn-pipeline.svg): compiler pipeline.
- [`docs/runtime-modes.svg`](docs/runtime-modes.svg): source, bytecode, interpreter, and JIT paths.
- [`docs/tensor-rune.svg`](docs/tensor-rune.svg): tensor runtime path.
- [`docs/example-code.svg`](docs/example-code.svg): example program walkthrough.
