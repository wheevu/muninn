# Architecture

Muninn is intentionally small. The goal is a coherent typed scripting runtime, not a broad language platform.

## Pipeline

```text
source -> lexer -> parser -> typecheck -> bytecode -> VM
```

- `src/lexer.rs` turns UTF-8 source into tokens with source spans.
- `src/parser.rs` builds the AST and assigns stable node IDs.
- `src/typecheck.rs` is the semantic source of truth for types, symbols, references, and diagnostics.
- `src/compiler.rs` lowers checked programs into bytecode.
- `src/bytecode.rs` encodes, decodes, and validates `.mubc` modules.
- `src/vm.rs` executes bytecode and reports span-carrying runtime errors.
- `src/frontend.rs` shares parser/typechecker results with CLI and LSP tooling.
- `lsp/` is deliberately thin: diagnostics, hover, and definition use the same semantic model as the CLI.

## Runtime boundaries

The VM treats bytecode as an input boundary. Decoded modules are validated before execution: opcodes, operand widths, local slots, jump targets, function references, and entry function bounds are checked before the VM runs.

Tensor allocation is capped so source programs cannot request arbitrarily large runtime buffers through tensor builtins. Compiler operands that must fit bytecode fields are checked before emission.

## JIT boundary

The JIT is experimental and feature-gated with `--features jit`. It only targets small hot loops over local `Int` values. Unsupported traces fall back to the interpreter.

The native path is deliberately narrow: traces are interpreted first by the trace engine and the Cranelift backend only handles a small integer subset with overflow bailouts. It is not a general optimizer.

## Non-goals for now

- classes, methods, enums, pattern matching, or generics
- broad standard-library expansion
- editor features that do not come from compiler semantics
- performance claims without release-mode measurement

The project is strongest when it stays boring at the boundaries and ambitious only where the behavior is tested.
