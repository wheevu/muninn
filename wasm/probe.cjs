// Drives the built module the way site/worker.js does, so a protocol change
// is caught here rather than in a browser console. Every check prints one
// PASS/FAIL line; any failure exits non-zero.
const fs = require("fs");
const path = process.argv[2] || "target/wasm32-unknown-unknown/release/muninn_wasm.wasm";

const bytes = fs.readFileSync(path);
const encoder = new TextEncoder();
const decoder = new TextDecoder();

let failures = 0;

function check(name, cond, detail) {
  if (cond) {
    console.log(`PASS ${name}`);
  } else {
    failures += 1;
    console.log(`FAIL ${name} :: ${detail}`);
  }
}

function show(value) {
  const text = JSON.stringify(value);
  return text.length > 200 ? `${text.slice(0, 200)}...` : text;
}

async function main() {
  const { instance } = await WebAssembly.instantiate(bytes, {});
  const { alloc, dealloc, set_source, call, result_len, memory } = instance.exports;

  function source(text) {
    const buffer = encoder.encode(text);
    const pointer = alloc(buffer.length);
    new Uint8Array(memory.buffer, pointer, buffer.length).set(buffer);
    set_source(pointer, buffer.length);
    dealloc(pointer, buffer.length);
  }

  function action(code) {
    const pointer = call(code);
    return JSON.parse(decoder.decode(new Uint8Array(memory.buffer, pointer, result_len())));
  }

  const ALLOWED = [
    "print",
    "eprint",
    "assert",
    "exit",
    "tensor_zeros",
    "tensor_fill",
    "tensor_reshape",
    "tensor_matmul",
    "tensor_sum",
  ];

  const info = action(0);
  const allowed = info.natives.filter((n) => n.inBrowser).map((n) => n.name).sort();
  check(
    "info reports 18 natives with 9 in browser",
    info.natives.length === 18 && allowed.length === 9,
    show({ natives: info.natives.length, allowed }),
  );
  check(
    "info allows exactly the browser-safe builtins",
    JSON.stringify(allowed) === JSON.stringify([...ALLOWED].sort()),
    show(allowed),
  );

  source('let x: Int = "no";\nx;\n');
  const broken = action(1);
  const diagnostic = broken.diagnostics && broken.diagnostics[0];
  check(
    "a type error produces a diagnostic with a phase and a line",
    broken.ok === false
      && typeof diagnostic.phase === "string"
      && diagnostic.phase.length > 0
      && typeof diagnostic.line === "number",
    show(broken),
  );

  source('print(env_get("HOME"));\n');
  const denied = action(1);
  check(
    "a denied builtin traps with the policy message",
    denied.ok === false
      && typeof denied.error.message === "string"
      && denied.error.message.includes("disabled by host policy"),
    show(denied),
  );

  source('let mut run: Bool = true;\nwhile (run) {\n}\n');
  const fuel = action(1);
  check(
    "fuel exhaustion reports instead of hanging",
    fuel.ok === false
      && typeof fuel.error.message === "string"
      && fuel.error.message.includes("fuel exhausted"),
    show(fuel),
  );

  source('let mut i: Int = 0;\nwhile (i < 3) {\nprint(i);\ni = i + 1;\n}\n');
  const listed = action(2);
  const rows = listed.functions.flatMap((f) => f.instructions);
  const jumpIfFalse = rows.find((r) => r.name === "jump_if_false");
  const loop = rows.find((r) => r.name === "loop");
  check(
    "jump_if_false operand ends in forward",
    !!jumpIfFalse
      && typeof jumpIfFalse.operand === "string"
      && jumpIfFalse.operand.endsWith("forward"),
    show(jumpIfFalse),
  );
  check(
    "loop operand ends in back",
    !!loop && typeof loop.operand === "string" && loop.operand.endsWith("back"),
    show(loop),
  );

  source('let x: Int = 1 + 2;\nprint(x);\n');
  const started = action(2);
  check(
    "operands is empty at instruction 0 and stack is gone",
    Array.isArray(started.operands)
      && started.operands.length === 0
      && !("stack" in started),
    show(started),
  );
  let grown = null;
  for (let i = 0; i < 50; i += 1) {
    const stepped = action(3);
    if (stepped.done) {
      break;
    }
    if (Array.isArray(stepped.operands) && stepped.operands.length > 0) {
      grown = stepped;
      break;
    }
  }
  check(
    "operands is non-empty after pushing a value",
    grown !== null && grown.operands.length > 0,
    show(grown),
  );
  check(
    "frames locals stay in the frames array",
    grown !== null
      && Array.isArray(grown.frames)
      && grown.frames.length > 0
      && Array.isArray(grown.frames[grown.frames.length - 1].locals),
    show(grown && grown.frames),
  );

  source('let t: Tensor = tensor_fill(2, 2, 1.5);\nprint(t);\n');
  const tensor = action(1);
  check(
    "a printed tensor yields a tensor line",
    Array.isArray(tensor.outputTensorLines) && tensor.outputTensorLines.includes(1),
    show(tensor),
  );
  source('print("tensor(shape=[2, 2], data=[1, 2, 3, 4])");\n');
  const impostor = action(1);
  check(
    "a string that reads like a tensor yields no tensor line",
    Array.isArray(impostor.outputTensorLines) && impostor.outputTensorLines.length === 0,
    show(impostor),
  );

  let many = 'let mut i: Int = 0;\nwhile (i < 40000) {\nprint(i);\ni = i + 1;\n}\n';
  source(many);
  const flooded = action(1);
  check(
    "printing past the limit truncates and counts the dropped lines",
    flooded.ok === true
      && flooded.outputTruncated === true
      && typeof flooded.outputDroppedLines === "number"
      && flooded.outputDroppedLines > 0
      && flooded.output.endsWith("\n"),
    show({
      ok: flooded.ok,
      outputTruncated: flooded.outputTruncated,
      outputDroppedLines: flooded.outputDroppedLines,
      tail: flooded.output.slice(-20),
    }),
  );

  source('let x: Int = 1 + 2;\nprint(x);\n');
  action(2);
  const resumed = action(5);
  check(
    "resume on a compiled session reaches done with a step count",
    resumed.done === true && typeof resumed.steps === "number" && resumed.steps > 0,
    show(resumed),
  );

  source('let x: Int = "no";\n');
  action(2);
  const lonely = action(5);
  check(
    "resume with no session returns the not-compiled message",
    lonely.ok === false
      && lonely.message === "Compile the program before running it to the end.",
    show(lonely),
  );

  source('print(21);\n');
  const first = action(1);
  const after = action(3);
  check(
    "step after run returns the finished state with the same output",
    after.done === true && after.output === first.output && after.output === "21\n",
    show({ first: first.output, after: after.output, done: after.done }),
  );

  if (failures > 0) {
    console.log(`${failures} check(s) failed`);
    process.exit(1);
  }
  console.log("all checks passed");
}

main().catch((error) => {
  console.error("FAILED:", error);
  process.exit(1);
});
