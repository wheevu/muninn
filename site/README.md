# The playground

A static page that runs Muninn in the reader's tab. The interpreter in
`src/` is compiled to WebAssembly, and everything the page shows comes out of
it: the program's output, its diagnostics, its bytecode, its stack, and its
call frames. The page draws and holds no model of the language.

## Build the interpreter for the page

```sh
./scripts/build_site_wasm.sh
```

This writes `site/muninn.wasm` and records the commit in the page's header
line. The build needs the `wasm32-unknown-unknown` target:

```sh
rustup target add wasm32-unknown-unknown
```

The module is a build product, not source, so it is not committed.

## Serve it

The page fetches the module and the example programs, so it needs HTTP rather
than a `file://` path:

```sh
python3 -m http.server 8000 --directory site
```

Then open <http://localhost:8000>.

## What the page is allowed to do

The host policy in `wasm/src/session.rs` is the one Muninn already has for
embedding untrusted scripts, filled in for a browser:

- instruction fuel, so a loop traps instead of freezing the tab
- a tensor element cap, because the built-in ceiling is 80 MB of `f64`
- an explicit builtin allowlist: printing, assertions, and the tensor
  functions. The IO builtins that reach the host process (`env_get`,
  `fs_read`, `clock_ms`, `stdin_read`, `args_get`) are denied, and the page's
  own reference table says which ones, from the interpreter's table rather
  than from a list written here
- no JIT. The tracing JIT compiles to native code, which a WebAssembly host
  has no use for, so `VmOptions.jit_enabled` is false and the page always
  shows what the interpreter does

Nothing runs on a server. The page is a folder of static files and the module.

## Checking it against the real thing

`wasm/probe.cjs` drives the built module the way the page's worker does, so a
protocol change is caught without a browser:

```sh
node wasm/probe.cjs
```

It runs a program, steps one, walks one to the end, and checks the three
answers that are easy to get wrong: a type error, a denied builtin, and a
runtime trap.
