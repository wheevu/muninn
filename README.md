# Muninn

A small statically typed scripting language built in Rust.

![Muninn compiler pipeline](docs/muninn-pipeline.svg)

Muninn parses source, checks types, compiles bytecode, and runs it on a stack VM.
The same program can run through the interpreter, bytecode path, hot-reload runtime, or the experimental integer tracing JIT.

## Measured snapshot

| Metric | Result |
| --- | ---: |
| Workspace tests | 193 |
| Benchmark targets | 6 |
| Scalar loop, compile and run | 400 µs/op |
| Scalar loop, VM only | 354 µs/op |
| Tensor pipeline, compile and run | 32 µs/op |

Apple M1, Rust 1.97.1, release mode, commit `e154224` plus uncommitted working tree.
These are local wall-clock measurements from `muninn-metrics`, not portable latency claims.

![Muninn runtime modes](docs/runtime-modes.svg)

## Language shape

- Typed functions, bindings, mutation, blocks, loops, and conditionals
- Int, Float, Bool, String, Tensor, Void, and nominal records
- Tensor broadcasting, matrix multiplication, reductions, and eager gradients
- Source checks, bytecode builds, editor support, and runtime metrics

<table>
  <tr>
    <td><img src="docs/example-code.svg" alt="A Muninn program"></td>
    <td><img src="docs/tensor-rune.svg" alt="The Muninn tensor path"></td>
  </tr>
</table>

[Build, run, benchmark, and inspect every command](GUIDE.md).

- [Architecture](docs/architecture.md)
- [Metrics](docs/metrics.md)
