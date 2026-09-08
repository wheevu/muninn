# Bytecode artifact policy

`.mubc` files are validated on load, never trusted.
There is no fast path that skips validation.

## Compatibility

- Same-`MUBC_VERSION` artifacts only. Today that is version 1.
- Decoding rejects bad magic, unsupported versions, truncated payloads,
  oversized counts, unknown tags, and invalid UTF-8.
- Strict trailing-byte rejection: one extra byte fails the load.
- Float bits, including NaN payloads, survive encode and decode exactly.
- Opcode numbers are never reused once assigned.
- A future version 2 must ship alongside version 1 with a migration test.
  Breaking version 1 readability is not allowed while this policy stands.

## Validation gate

The decoded module passes the same validation as compiler output before
anything runs: opcodes, operand widths, local slots, jump targets,
function references, and entry function bounds. Accepted modules
additionally guarantee three runtime properties: no operand-stack
underflow on any reachable path, no fall off the end of a function, and
agreement on stack height wherever control-flow paths join. Jump and loop
targets must be instruction starts, not merely in-bounds offsets.
Operand types stay unchecked on purpose: the value stack is dynamically
typed, so type errors remain runtime guards with source spans.

## Cache guidance for hosts

Mirror the CPython `.pyc` model: key any on-disk bytecode cache by
(source hash, `MUBC_VERSION`), never by timestamp alone, so
reproducible builds stay reproducible. Invalidation modes:

- `CHECKED`: recompile when the source hash disagrees, else load.
- `UNCHECKED`: trust the cache; the surrounding build system owns freshness.

## Fuzz corpus

`tests/artifact.rs` holds the checked-in seed corpus: every
`examples/*.mun` program must compile, validate, roundtrip through
encode and decode, and rerun to the same value. The bit-flip sweep
asserts zero accept-then-fault divergences: a mutant the validator
accepts must never hit a validation-class fault
(stack underflow, bad opcode, bad constant or slot, wild jump) when run
under fuel. Grow the corpus with hostile `tests/sandbox.rs` programs
before widening acceptance anywhere.
