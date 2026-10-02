// The playground's only door to the interpreter.
//
// The module is fetched, instantiated, and kept here so a long run cannot
// freeze the page the reader is reading. Everything the page knows about a
// program arrives as JSON from this worker, including the numbers, which
// come from the interpreter and not from anything in this file.

const ACTION_INFO = 0;
const ACTION_RUN = 1;
const ACTION_INIT = 2;
const ACTION_STEP = 3;
const ACTION_HIGHLIGHT = 4;
const ACTION_RESUME = 5;

let module = null;
const encoder = new TextEncoder();
const decoder = new TextDecoder();

/** Copies the program text into linear memory and hands it to the module. */
function setSource(source) {
  const bytes = encoder.encode(source);
  const pointer = module.alloc(bytes.length);
  new Uint8Array(module.memory.buffer, pointer, bytes.length).set(bytes);
  module.set_source(pointer, bytes.length);
  module.dealloc(pointer, bytes.length);
}

/** Runs one action and parses what the module wrote back. */
function act(action, source) {
  if (source !== undefined) setSource(source);
  const pointer = module.call(action);
  const length = module.result_len();
  const bytes = new Uint8Array(module.memory.buffer, pointer, length).slice();
  return JSON.parse(decoder.decode(bytes));
}

async function ensureLoaded() {
  if (module) return;
  const response = await fetch("./muninn.wasm");
  if (!response.ok) {
    throw new Error(
      `muninn.wasm is missing (HTTP ${response.status}). Build it first; see site/README.md.`,
    );
  }
  const bytes = await response.arrayBuffer();
  // No imports: the module is Muninn plus the standard library, and neither
  // asks the host for anything.
  const instantiated = await WebAssembly.instantiate(bytes, {});
  module = instantiated.instance.exports;
}

onmessage = async (event) => {
  const { id, action, source } = event.data;
  try {
    await ensureLoaded();
    postMessage({ id, result: act(action, source) });
  } catch (error) {
    postMessage({ id, error: error.message });
  }
};
