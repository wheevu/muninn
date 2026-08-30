# Muninn

A small statically typed scripting language built in Rust.

![Muninn compiler pipeline](docs/muninn-pipeline.svg)

Muninn parses source, checks types, compiles bytecode, and runs it on a stack VM.
The same program can run through the interpreter, bytecode path, hot-reload runtime, or the experimental integer tracing JIT.

![Muninn runtime modes](docs/runtime-modes.svg)

## Language shape

- Typed functions, bindings, mutation, blocks, loops, and conditionals
- Int, Float, Bool, String, Tensor, and Void
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
