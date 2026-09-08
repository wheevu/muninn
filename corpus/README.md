# Expressiveness corpus

Eight small programs that Muninn can express today, kept as data only.
No script here requires a syntax change; the point is to find the
smallest type addition that unblocks the programs Muninn cannot express
cleanly.

## The bake-off question

Records vs fixed arrays vs an explicit fallible value: which single
addition unblocks the most real scripts per unit of semantic cost?
`nested_data_workaround.mun` and `parallel_tensors.mun` are the exhibits:
both work, but row order and lockstep tensors are convention, not checked
shape. Luau answered with gradual tables plus readonly properties,
Rune with anonymous objects first and checked structs second,
Starlark with lists plus tuples plus dicts.

## What each script probes

- `config_eval.mun`: typed constants, derived values, asserts.
- `game_tick.mun`: mutable entity state advanced per tick.
- `tensor_prep.mun`: build, reshape, and reduce a small batch.
- `boundary_checks.mun`: negation, comparison, string concatenation.
- `nested_data_workaround.mun`: nested data without maps, via tensors.
- `parallel_tensors.mun`: lockstep tensors standing in for arrays.
- `shadowing.mun`: shadowing and same-named locals for the LSP index.
- `hot_loop.mun`: the numeric kernel the tracing JIT watches.

## Rules

- Every script must compile and run on the current core.
- Nothing here may assume a proposed feature.
- The decision, when made, must pass the full admission gate in
  `DESIGN.md`: syntax, type rules, diagnostics, runtime representation,
  parser plus semantic plus runtime tests, tooling review.

## Scorecard (records vs fixed arrays vs fallible value)

| script | arrays | records | fallible |
| --- | --- | --- | --- |
| `config_eval.mun` | no | yes, grouped config | no |
| `game_tick.mun` | partial, N homogeneous entities | yes, one entity bundle | no |
| `tensor_prep.mun` | no, tensors cover it | no | no |
| `boundary_checks.mun` | no | no | no |
| `nested_data_workaround.mun` | partial | yes, named pairs | no |
| `parallel_tensors.mun` | partial | partial, named groups | no |
| `shadowing.mun` | no | no | no |
| `hot_loop.mun` | no | no | no |

Verdict: nominal records. Muninn already owns homogeneous numeric
collections through `Tensor`, so fixed numeric arrays would duplicate
the tensor path instead of complementing it. The uncovered shape is
heterogeneous named grouping, which records answer with no overlap.
Nominal (name-keyed, no subtyping, no generics) keeps `ty_compatible`
as plain equality. Fallible values help none of the eight scripts and
stay deferred. Field assignment (`p.x = 1`) is out of scope: whole
record replacement through existing `Assign` covers mutation.
