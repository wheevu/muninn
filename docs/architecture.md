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
- `src/vm.rs` executes bytecode, maintains VM-local global lookup cache stats, handles safe-point reloads, and reports span-carrying runtime errors.
- `src/frontend.rs` shares parser/typechecker results with CLI and LSP tooling. `references_to_target` exposes the exact reference index (definition plus reference spans per symbol id) behind LSP references, rename, and highlight, so same-named locals stay distinct without text matching.
- `lsp/` is deliberately thin: diagnostics, hover, and definition use the same semantic model as the CLI.
- `src/autodiff.rs` is an additive eager tensor-expression tape. Reverse-mode `grad` traverses it directly and is intentionally not part of bytecode, VM, JIT, hot-reload, or LSP execution.

## Runtime boundaries

The VM treats bytecode as an input boundary. Decoded modules are validated before execution: opcodes, operand widths, local slots, jump targets, function references, and entry function bounds are checked before the VM runs.

Accepted modules additionally guarantee three runtime properties: no operand-stack underflow on any reachable path, no fall off the end of a function, and agreement on stack height wherever control-flow paths join. Jump and loop targets must be instruction starts, not merely in-bounds offsets. Operand types are deliberately unchecked: the value stack is dynamically typed, so type errors stay runtime guards with source spans.

Hosts embed untrusted scripts through `HostPolicy`: a builtin allowlist that traps by name, deterministic per-instruction fuel, a best-effort wall-clock timeout checked every 1024 instructions, a host tensor element cap enforced before allocation, and optional readonly globals. One `Vm` per guest is the isolation boundary; budgets reset on fresh starts and successful reloads. The default policy preserves legacy behavior (everything allowed, no limits).

Tensor allocation is capped so source programs cannot request arbitrarily large runtime buffers through tensor builtins. The empty shape `[]` is a scalar; all non-scalar dimensions must be positive, and zero-element tensors are rejected consistently. Compiler operands that must fit bytecode fields are checked before emission.

## Bytecode artifact contract

`.mubc` files are validated on load, never trusted. Decoding rejects bad magic, unsupported versions, truncated payloads, oversized counts, unknown tags, and invalid UTF-8; the decoded module then passes the same validation as compiler output before anything runs. There is no fast path that skips validation.

Compatibility promises are narrow: same-`MUBC_VERSION` artifacts only, strict trailing-byte rejection, and float bits (including NaN payloads) preserved exactly through encode and decode. Opcode numbers are never reused once assigned. Decoding uses no `unsafe`; the only `unsafe` in the workspace is the feature-gated JIT native-trace path.

## JIT boundary

The JIT is experimental and feature-gated with `--features jit`. It only targets small hot loops over local `Int` values. Unsupported traces fall back to the interpreter.

The native path is deliberately narrow: traces are interpreted by the trace engine when no native trace is available, and the Cranelift backend only handles a small integer subset with overflow bailouts back to the interpreter. It is not a general optimizer.

The VM talks to traces only through the `JitBackend` trait in `src/jit.rs` (observe, run-if-ready, stats, clear). Cranelift imports live solely in the feature-gated native-trace region of that file, so upstream API breakage cannot leak into `vm.rs` or the core.

Pinned to Cranelift 0.135 as of this revision (up from 0.115): the jump needed exactly two mechanical renames inside `src/jit.rs` (`MemFlags` to `MemFlagsData`, `finalize()` to `finalize(isa.frontend_config())`) plus two sign-extending immediate helpers (`iadd_imm_s`, `icmp_imm_s`, behavior-identical for the small non-negative immediates used). Full workspace suite passes with and without the feature, including the native-trace equivalence tests.

## Global lookup cache

The VM caches global reads by name for the current globals epoch. Defining or mutating a global evicts only that key; requesting/applying reload, or other full runtime cache invalidation, bumps the epoch and clears the cache. The invalidation counter records every eviction event, even when the evicted key or cache was already empty. This is a focused lookup cache, not a promised inline-cache optimization surface.

## Non-goals for now

- classes, methods, enums, pattern matching, or generics
- broad standard-library expansion
- editor features that do not come from compiler semantics
- performance claims without release-mode measurement
- source-level differentiation syntax, higher-order derivatives, and JIT/VM differentiation

The project is strongest when it stays boring at the boundaries and ambitious only where the behavior is tested.
