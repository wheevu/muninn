// The page.
//
// It draws what the interpreter reports and holds no model of the language:
// no keyword list, no type rules, no arithmetic. The colouring comes from
// Muninn's own lexer, the diagnostics from its checker, the operands and
// locals from the running VM. Anything the page cannot get from the module,
// it does not show.
//
// One session at a time: state.session is the only run the panels describe,
// so the Result panel and the Execution panel can never disagree. Running
// finishes the session in one response; inspecting compiles without running
// and leaves the stepping to the reader.

const ACTION = { INFO: 0, RUN: 1, INIT: 2, STEP: 3, HIGHLIGHT: 4, RESUME: 5 };

const el = {
  provenance: document.querySelector("#provenance"),
  exampleList: document.querySelector("#example-list"),
  exampleNote: document.querySelector("#example-note"),
  exampleTitle: document.querySelector("#example-title"),
  source: document.querySelector("#source"),
  highlight: document.querySelector("#highlight"),
  gutter: document.querySelector("#gutter"),
  run: document.querySelector("#run"),
  reset: document.querySelector("#reset"),
  live: document.querySelector("#live"),
  diagnostics: document.querySelector("#diagnostics"),
  output: document.querySelector("#output"),
  outputNote: document.querySelector("#output-note"),
  value: document.querySelector("#value"),
  stale: document.querySelector("#stale"),
  inspect: document.querySelector("#inspect"),
  step: document.querySelector("#step"),
  finish: document.querySelector("#finish"),
  next: document.querySelector("#next"),
  operands: document.querySelector("#operands"),
  frames: document.querySelector("#frames"),
  listing: document.querySelector("#listing"),
  listingTitle: document.querySelector("#listing-title"),
  nativeTable: document.querySelector("#native-table"),
  budget: document.querySelector("#budget"),
  workbench: document.querySelector("#workbench"),
};

const state = {
  worker: null,
  generation: 0,
  nextId: 0,
  pending: new Map(),
  inflight: 0,
  revision: 0,
  dirty: false,
  booted: false,
  fatal: false,
  session: null,
  examples: [],
  current: null,
  original: "",
  listing: null,
  tokens: [],
  caretLine: 1,
};

function createWorker() {
  const worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
  const generation = state.generation;
  worker.onmessage = (event) => {
    if (generation !== state.generation) return;
    const { id, result, error } = event.data;
    const entry = state.pending.get(id);
    if (!entry) return;
    state.pending.delete(id);
    clearTimeout(entry.timer);
    // A run-as-you-type run can finish after the reader kept typing. Its
    // revision is old, so it describes a program that is no longer on
    // screen, and landing it would overwrite the newer run.
    if (entry.revision !== state.revision) {
      entry.resolve(undefined);
      return;
    }
    if (error) entry.reject(new Error(error));
    else entry.resolve(result);
  };
  worker.onerror = (event) => {
    if (generation !== state.generation) return;
    retireWorker(event.message || "the worker stopped");
  };
  state.worker = worker;
}

createWorker();

/** Retires the current interpreter and hands the page a fresh one.
 *
 * A worker that errored, rejected, or stayed silent past the wait cannot be
 * trusted for the next request either, so it is terminated rather than
 * reused. Every pending request settles, so no button stays disabled behind
 * a promise that will never land. The source is untouched: nothing the
 * reader typed is ever the thing being recovered from.
 */
function retireWorker() {
  const old = state.worker;
  state.generation += 1;
  if (old) {
    try {
      old.terminate();
    } catch {
      // Termination is best-effort; the generation bump already orphaned it.
    }
  }
  for (const [, entry] of state.pending) {
    clearTimeout(entry.timer);
    // No answer rather than a failure. `undefined` already means "this reply
    // does not describe the program on screen" in this file, and a rejection
    // would send every waiting caller into its own recovery path, which
    // would retire the healthy worker this call is about to build.
    entry.resolve(undefined);
  }
  state.pending.clear();
  state.inflight = 0;
  state.session = null;
  createWorker();
  // A module that never loaded has not "stopped responding"; the header
  // already carries the real reason for that failure, and saying both would
  // describe a restart that did not happen.
  if (state.booted) {
    setOutputNoteError(
      "The interpreter stopped responding. It has been restarted. Nothing you typed was lost.",
    );
  }
  syncTransport();
}

function ask(action, source, options = {}) {
  const transport = options.transport === true;
  const revision = state.revision;
  const generation = state.generation;
  const worker = state.worker;
  return new Promise((resolve, reject) => {
    const id = state.nextId;
    state.nextId += 1;
    // One wait guards every request, so a stuck interpreter is detected the
    // same way whether it wedged mid-run or mid-step. The timer retires the
    // worker, and the retirement settles this request as failed.
    const timer = setTimeout(() => {
      if (generation !== state.generation) return;
      retireWorker();
    }, 15000);
    state.pending.set(id, { resolve, reject, revision, timer });
    if (transport) {
      state.inflight += 1;
      syncTransport();
    }
    const settle = (fn, value) => {
      clearTimeout(timer);
      if (transport) {
        state.inflight = Math.max(0, state.inflight - 1);
        syncTransport();
      }
      fn(value);
    };
    const entry = state.pending.get(id);
    if (entry) {
      const originalResolve = entry.resolve;
      const originalReject = entry.reject;
      entry.resolve = (value) => settle(originalResolve, value);
      entry.reject = (error) => settle(originalReject, error);
    }
    try {
      worker.postMessage({ id, action, source });
    } catch (error) {
      state.pending.delete(id);
      settle(reject, error);
      retireWorker();
    }
  });
}

/** Enables exactly the transport the current session honestly allows.
 *
 * Run and Inspect are always available when no request is in flight. Step
 * and To the end need a live session: one that exists, has not finished,
 * and still matches the source on screen. Editing disables them without
 * clearing the last result, so the reader sees what ran and that it is old.
 */
function syncTransport() {
  // A page that has declared itself unable to run programs must not offer to
  // run one. The fatal note and an enabled Run button contradict each other,
  // and the click would time out into silence.
  if (state.fatal) {
    el.workbench.setAttribute("aria-busy", "false");
    el.run.disabled = true;
    el.inspect.disabled = true;
    el.step.disabled = true;
    el.finish.disabled = true;
    return;
  }
  const busy = state.inflight > 0;
  el.workbench.setAttribute("aria-busy", busy ? "true" : "false");
  el.run.disabled = busy;
  el.inspect.disabled = busy;
  const steppable = !busy && state.session !== null && !state.session.done && !state.dirty;
  el.step.disabled = !steppable;
  el.finish.disabled = !steppable;
}

function setOutputNote(text) {
  el.outputNote.classList.remove("error");
  if (!text) {
    el.outputNote.hidden = true;
    el.outputNote.textContent = "";
    return;
  }
  el.outputNote.hidden = false;
  el.outputNote.textContent = text;
}

function setOutputNoteError(text) {
  el.outputNote.classList.add("error");
  el.outputNote.hidden = false;
  el.outputNote.textContent = text;
}

/** Marks what is on screen as older than the source.
 *
 * The last result stays visible on purpose: the reader compares what ran
 * against what they just typed, instead of staring at a cleared panel while
 * the next run is still in flight.
 */
function markStale() {
  state.revision += 1;
  state.dirty = true;
  el.stale.hidden = false;
  el.stale.textContent = "Source edited since this ran.";
  syncTransport();
}

function clearStale() {
  state.dirty = false;
  el.stale.hidden = true;
  el.stale.textContent = "";
}

/* Examples */

async function loadExamples() {
  const response = await fetch("./examples/index.json");
  if (!response.ok) {
    showFatal("the example list is missing (site/examples/index.json)");
    return;
  }
  state.examples = await response.json();

  for (const example of state.examples) {
    const item = document.createElement("li");
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = example.title;
    button.addEventListener("click", () => select(example));
    item.append(button);
    el.exampleList.append(item);
    example.button = button;
  }

  await select(state.examples[0]);
}

async function select(example) {
  // Bumped before the fetch, so a slow fetch for one example that resolves
  // after the reader picked another is discarded instead of landing.
  state.revision += 1;
  const mine = state.revision;
  state.current = example;
  for (const other of state.examples) {
    other.button.setAttribute("aria-current", String(other === example));
  }
  el.exampleTitle.textContent = example.file;
  el.exampleNote.textContent = example.note;

  const response = await fetch(`./examples/${example.file}`);
  if (!response.ok) {
    showFatal(`${example.file} is missing`);
    return;
  }
  if (mine !== state.revision) return;
  const source = await response.text();
  if (mine !== state.revision) return;
  state.original = source;
  el.source.value = source;
  resizeGutter();
  markStale();
  await run();
  await highlight();
}

/* The editor */

/** Everything that has to follow a change to the source text.
 *
 * Assigning el.source.value fires no input event, so the Tab handler calls
 * this by hand. Without it the drawn layer lags four characters behind the
 * caret and the result on screen is not marked as older than the source.
 */
function onSourceChanged() {
  markStale();
  resizeGutter();
  clearTimeout(el.source._highlightTimer);
  el.source._highlightTimer = setTimeout(highlight, 90);
  if (el.live.checked) {
    clearTimeout(el.source._runTimer);
    el.source._runTimer = setTimeout(run, 400);
  }
}

el.source.addEventListener("input", onSourceChanged);

el.source.addEventListener("keydown", (event) => {
  if (event.key === "Tab") {
    // Shift+Tab is the only way a keyboard-only reader can leave the editor.
    // Everything before it in the document is reached by Shift+Tab, and the
    // browser's own forward Tab is spent on indentation, so swallowing
    // Shift+Tab as well would make the editor a keyboard trap.
    if (event.shiftKey) return;
    event.preventDefault();
    const { selectionStart, selectionEnd, value } = el.source;
    el.source.value = `${value.slice(0, selectionStart)}    ${value.slice(selectionEnd)}`;
    el.source.selectionStart = selectionStart + 4;
    el.source.selectionEnd = selectionStart + 4;
    onSourceChanged();
  }
  if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
    event.preventDefault();
    run();
  }
});

el.source.addEventListener("scroll", () => {
  el.highlight.scrollTop = el.source.scrollTop;
  el.highlight.scrollLeft = el.source.scrollLeft;
  el.gutter.scrollTop = el.source.scrollTop;
});

el.source.addEventListener("keyup", rememberCaret);
el.source.addEventListener("click", rememberCaret);

function rememberCaret() {
  state.caretLine = el.source.value.slice(0, el.source.selectionStart).split("\n").length;
  paintGutter();
  paintCurrentLine();
}

function resizeGutter() {
  const lines = el.source.value.split("\n").length;
  el.gutter.replaceChildren(
    ...Array.from({ length: lines }, (_, index) =>
      Object.assign(document.createElement("div"), { textContent: index + 1 }),
    ),
  );
  paintGutter();
}

function paintGutter() {
  for (const child of el.gutter.children) {
    child.className = Number(child.textContent) === state.caretLine ? "marked" : "";
  }
}

/** Moves the washed line to wherever the caret is.
 *
 * The caret moves on every arrow key and click, and the wash is the only
 * thing that has to follow it. Rebuilding the layer would re-tokenize the
 * whole program to change one class name, so this touches the class alone.
 * Without it the gutter marker and the washed line disagree, and the code
 * looks shifted by a line.
 */
function paintCurrentLine() {
  const lines = el.highlight.children;
  for (let index = 0; index < lines.length; index += 1) {
    lines[index].classList.toggle("current", index + 1 === state.caretLine);
  }
}

async function highlight() {
  const mine = state.revision;
  try {
    const result = await ask(ACTION.HIGHLIGHT, el.source.value);
    if (!result || mine !== state.revision) return;
    state.tokens = result.ok ? result.tokens : [];
    paintHighlight();
    if (!result.ok && result.diagnostics) {
      showDiagnostics(result.diagnostics);
    }
  } catch {
    retireWorker();
  }
}

/** Repaints the layer behind the textarea.
 *
 * The textarea's own text is transparent, so this layer is what the reader
 * sees and it has to hold exactly the text the lexer saw, or the caret stops
 * lining up with the words. One block per line rather than one text run, so
 * the line the caret is on can be washed across the full width of the editor
 * instead of only behind the words that happen to be on it.
 */
function paintHighlight() {
  const lines = el.source.value.split("\n");
  const byLine = new Map();
  for (const token of state.tokens) {
    if (!byLine.has(token.line)) byLine.set(token.line, []);
    byLine.get(token.line).push(token);
  }

  el.highlight.replaceChildren(
    ...lines.map((text, index) => {
      const line = index + 1;
      const lineNode = document.createElement("div");
      lineNode.className = "line";
      const tokens = (byLine.get(line) || []).slice().sort((a, b) => a.column - b.column);
      let column = 0;
      for (const token of tokens) {
        const start = token.column - 1;
        const end = token.endColumn - 1;
        if (start < column || start > text.length) continue;
        lineNode.append(document.createTextNode(text.slice(column, start)));
        const span = document.createElement("span");
        span.className = cssClass(token.class);
        span.textContent = text.slice(start, end);
        lineNode.append(span);
        column = end;
      }
      lineNode.append(document.createTextNode(text.slice(column)));
      return lineNode;
    }),
  );
  el.highlight.scrollTop = el.source.scrollTop;
  el.highlight.scrollLeft = el.source.scrollLeft;
  paintCurrentLine();
}

function cssClass(name) {
  return (
    {
      keyword: "kw",
      type: "ty",
      number: "num",
      string: "str",
      constant: "lit",
    }[name] || ""
  );
}

/* Running */

el.run.addEventListener("click", run);
el.reset.addEventListener("click", async () => {
  // Reset is an edit that happens to restore the original text: the old
  // result is old either way, so it is marked stale and the run clears it.
  el.source.value = state.original;
  resizeGutter();
  markStale();
  await run();
  await highlight();
});
el.inspect.addEventListener("click", inspect);
el.step.addEventListener("click", step);
el.finish.addEventListener("click", finish);

async function run() {
  // A run is one round trip. The response carries the finished session, so
  // there is nothing to load after it and nothing that may overwrite it.
  const mine = state.revision;
  clearStale();
  syncTransport();
  try {
    const result = await ask(ACTION.RUN, el.source.value, { transport: true });
    if (!result || mine !== state.revision) return;
    renderResult(result);
  } catch {
    retireWorker();
  } finally {
    syncTransport();
  }
}

async function inspect() {
  await loadListing(ACTION.INIT);
}

async function loadListing(action) {
  // The only path that compiles without running. The session it leaves
  // behind is fresh: stepped from the start, never from a finished run.
  const mine = state.revision;
  clearStale();
  syncTransport();
  try {
    const result = await ask(action, el.source.value, { transport: true });
    if (!result || mine !== state.revision) return;
    renderInspect(result);
  } catch {
    retireWorker();
  } finally {
    syncTransport();
  }
}

async function step() {
  if (!state.session || state.session.done || state.dirty) return;
  const mine = state.revision;
  try {
    const result = await ask(ACTION.STEP, undefined, { transport: true });
    if (!result || mine !== state.revision) return;
    renderResult(result);
  } catch {
    retireWorker();
  } finally {
    syncTransport();
  }
}

async function finish() {
  // To the end is one round trip that finishes the session where stepping
  // would, so a long program cannot strand the reader halfway through it.
  if (!state.session || state.session.done || state.dirty) return;
  const mine = state.revision;
  try {
    const result = await ask(ACTION.RESUME, undefined, { transport: true });
    if (!result || mine !== state.revision) return;
    renderResult(result);
  } catch {
    retireWorker();
  } finally {
    syncTransport();
  }
}

/** Renders a compiled-but-not-run session.
 *
 * The reader asked to look, not to execute, so the output stays empty and
 * says so through its existing empty state, and the note names the two ways
 * forward. Stepping starts at instruction 0 because nothing has run yet.
 */
function renderInspect(result) {
  if (!result) return;
  if (result.diagnostics) {
    renderCompileFailure(result);
    return;
  }
  if (result.ok === false && !Array.isArray(result.functions)) {
    state.session = null;
    state.listing = null;
    el.next.textContent = result.message || "Nothing to inspect yet.";
    syncTransport();
    return;
  }
  state.session = { done: false };
  state.listing = result.functions || null;
  el.diagnostics.replaceChildren();
  el.output.textContent = "";
  el.output.classList.add("unreached");
  el.value.hidden = true;
  el.value.textContent = "";
  setOutputNote("Compiled but not run. Press Run to execute it, or Step to walk it.");
  renderState(result);
  syncTransport();
}

function renderCompileFailure(result) {
  showDiagnostics(result.diagnostics);
  el.output.textContent = "";
  el.output.classList.add("unreached");
  setOutputNote("");
  el.value.hidden = true;
  el.value.textContent = "";
  el.next.textContent = "This program does not compile, so there is nothing to step through.";
  el.operands.replaceChildren();
  el.frames.replaceChildren();
  el.listing.replaceChildren();
  el.listingTitle.textContent = "Bytecode";
  state.listing = null;
  state.session = null;
  syncTransport();
}

function renderResult(result) {
  if (!result) return;
  if (result.diagnostics) {
    renderCompileFailure(result);
    return;
  }
  if (result.ok === false && !Array.isArray(result.functions) && result.message) {
    state.session = null;
    el.next.textContent = result.message;
    syncTransport();
    return;
  }
  el.diagnostics.replaceChildren();
  el.output.classList.remove("unreached");
  el.output.replaceChildren(...outputNodes(result.output || "", result.outputTensorLines || []));
  if (result.outputTruncated) {
    // Every printed line ends with a newline, so the split leaves a trailing
    // empty entry that no reader ever saw. Counting it would overstate what
    // was kept by one line.
    const kept = result.output ? result.output.replace(/\n$/, "").split("\n").length : 0;
    const dropped = result.outputDroppedLines || 0;
    setOutputNote(
      `The interpreter kept the first ${kept} lines; ${dropped} more were printed and were not kept.`,
    );
  } else {
    setOutputNote("");
  }
  if (result.error) {
    el.value.hidden = true;
    el.value.textContent = "";
    showDiagnostics([
      {
        phase: result.error.phase,
        message: result.error.message,
        line: result.error.line,
        column: result.error.column,
        endLine: result.error.line,
        endColumn: result.error.column,
      },
    ]);
  } else if (result.done && result.value && result.value !== "nil") {
    // A program whose last expression is a `print` returns nil, and "the
    // program returned nil" tells a reader nothing. The execution line below
    // already says the program finished.
    el.value.hidden = false;
    el.value.replaceChildren("The program returned ");
    const strong = document.createElement("b");
    strong.textContent = result.value;
    el.value.append(strong, ".");
  } else {
    el.value.hidden = true;
    el.value.textContent = "";
  }
  if (Array.isArray(result.functions)) {
    state.listing = result.functions;
  }
  state.session = { done: result.done === true };
  renderState(result);
  syncTransport();
}

/** Reads one printed line as the tensor the interpreter reported.
 *
 * The line list comes from the response, never from the text: a line that
 * merely looks like a tensor is printed as the text it is. The shape still
 * has to agree with the data, because a grid drawn from mismatched numbers
 * would be a picture of nothing, and huge tensors stay text so the tab does.
 */
function parseTensorLine(line) {
  const prefix = "tensor(shape=[";
  const middle = "], data=[";
  const suffix = "])";
  if (!line.startsWith(prefix) || !line.endsWith(suffix)) return null;
  const shapeEnd = line.indexOf(middle);
  if (shapeEnd === -1) return null;
  const shapeText = line.slice(prefix.length, shapeEnd);
  const dataText = line.slice(shapeEnd + middle.length, line.length - suffix.length);
  const shape = shapeText
    .split(",")
    .map((part) => part.trim())
    .filter((part) => part !== "");
  const data = dataText
    .split(",")
    .map((part) => part.trim())
    .filter((part) => part !== "");
  if (shape.length === 0 || shape.length > 2) return null;
  if (!shape.every((part) => /^\d+$/.test(part))) return null;
  if (!data.every((part) => Number.isFinite(Number(part)))) return null;
  const dims = shape.map(Number);
  const cells = dims.reduce((product, dim) => product * dim, 1);
  if (data.length === 0 || cells !== data.length || cells > 1024) return null;
  // `data` keeps the characters the interpreter printed. Round-tripping each
  // cell through Number would show "2" for a value printed as "2.0", which is
  // a number the program never printed.
  return { shape: dims, data };
}

function outputNodes(text, tensorLines) {
  const marked = new Set(Array.isArray(tensorLines) ? tensorLines : []);
  const nodes = [];
  const lines = text === "" ? [] : text.split("\n");
  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index];
    const parsed = marked.has(index + 1) ? parseTensorLine(line) : null;
    if (parsed) {
      // The newline is re-emitted rather than assumed. The grid sits inline
      // so its own line can hold the shape label and the cells together, and
      // without this the next printed line would share that line: a scalar
      // printed after a tensor would read as a caption on the tensor.
      nodes.push(tensorGrid(parsed.shape, parsed.data), document.createTextNode("\n"));
    } else {
      // A tensor the page will not draw has to say so. The copy promises a
      // grid for a printed tensor, and silently printing text instead would
      // make that promise wrong without telling the reader.
      const undrawn = marked.has(index + 1) ? "  (not drawn as a grid)" : "";
      nodes.push(document.createTextNode(`${line}${undrawn}\n`));
    }
  }
  return nodes;
}

/** Draws a printed tensor as a grid.
 *
 * The shape label sits beside the grid rather than in it. Inside the grid it
 * would take a cell of its own, and a 2x2 matrix would come out three cells
 * deep and one ragged, which is a picture of nothing.
 */
function tensorGrid(shape, data) {
  const wrapper = document.createElement("div");
  wrapper.className = "tensor-wrap";

  const label = document.createElement("div");
  label.className = "tensor-shape";
  label.textContent = `shape [${shape.join(", ")}]`;
  wrapper.append(label);

  const grid = document.createElement("div");
  grid.className = "tensor";
  const columns = shape.length === 1 ? data.length : shape[1];
  if (columns > 1) grid.style.gridTemplateColumns = `repeat(${columns}, auto)`;
  for (const value of data) {
    const cell = document.createElement("span");
    cell.textContent = value;
    grid.append(cell);
  }
  wrapper.append(grid);
  return wrapper;
}

function renderState(result) {
  if (!result) return;
  const next = result.next;
  el.next.replaceChildren();
  if (next) {
    el.next.append(
      `${next.function} · offset ${next.ip} · `,
      Object.assign(document.createElement("b"), { textContent: next.name }),
      next.operand ? ` ${next.operand}` : "",
      next.line ? ` · line ${next.line}` : "",
    );
  } else if (result.done) {
    const quiet = document.createElement("span");
    quiet.className = "quiet";
    quiet.textContent = `Finished after ${result.steps} instructions.`;
    el.next.append(quiet);
  }

  el.operands.replaceChildren(...slotNodes(result.operands || [], (value) => value));
  const frames = result.frames || [];
  const frameNodes = frames.flatMap((frame) => [
    Object.assign(document.createElement("li"), {
      className: "fn",
      textContent: `${frame.function}()`,
    }),
    ...slotNodes(frame.locals),
  ]);
  // Both slots panels say "empty" the same way. A blank Frames list beside an
  // Operands list that reads "empty" looks like two different facts about the
  // same moment.
  el.frames.replaceChildren(...(frameNodes.length > 0 ? frameNodes : slotNodes([])));
  renderListing(next);
}

function slotNodes(values, label = (value) => value) {
  if (values.length === 0) {
    const empty = document.createElement("li");
    empty.textContent = "";
    return [empty];
  }
  return values.map((value) =>
    Object.assign(document.createElement("li"), { textContent: label(value) }),
  );
}

function renderListing(next) {
  if (!state.listing) return;
  const functionId = next ? next.functionId : 0;
  const fn = state.listing.find((candidate) => candidate.id === functionId) || state.listing[0];
  if (!fn) {
    el.listing.replaceChildren();
    return;
  }
  el.listingTitle.textContent = `Bytecode · ${fn.name}()`;
  el.listing.replaceChildren(
    ...fn.instructions.map((instruction) => {
      const item = document.createElement("li");
      const isCurrent = next && instruction.ip === next.ip;
      const isPast = next && instruction.ip < next.ip;
      if (isCurrent) item.className = "current";
      else if (isPast) item.className = "past";
      const ip = document.createElement("span");
      ip.className = "ip";
      ip.textContent = instruction.ip;
      const body = document.createElement("span");
      const op = document.createElement("span");
      op.className = "op";
      op.textContent = instruction.name;
      body.append(op);
      if (instruction.operand) {
        const arg = document.createElement("span");
        arg.className = "arg";
        arg.textContent = ` ${instruction.operand}`;
        body.append(arg);
      }
      item.append(ip, body);
      return item;
    }),
  );
  const current = el.listing.querySelector("li.current");
  if (current) current.scrollIntoView({ block: "nearest" });
}

function showDiagnostics(diagnostics) {
  el.diagnostics.replaceChildren(
    ...diagnostics.map((diagnostic) => {
      const item = document.createElement("li");
      const where = document.createElement("button");
      where.type = "button";
      where.className = "where";
      where.textContent = `${diagnostic.line}:${diagnostic.column}`;
      where.addEventListener("click", () => reveal(diagnostic));
      const text = document.createElement("span");
      const phase = document.createElement("span");
      phase.className = "phase";
      phase.textContent = `${diagnostic.phase}: `;
      text.append(phase, diagnostic.message);
      item.append(where, text);
      return item;
    }),
  );
}

/** Puts the caret where the interpreter pointed. */
function reveal(diagnostic) {
  const lines = el.source.value.split("\n");
  let start = 0;
  for (let line = 1; line < diagnostic.line; line += 1) start += lines[line - 1].length + 1;
  // The column is where the complaint starts, not where the line does: the
  // selection opens on the offending word instead of the line's first word.
  start += diagnostic.column - 1;
  const end = start + Math.max(1, (diagnostic.endColumn || diagnostic.column + 1) - diagnostic.column);
  el.source.focus();
  el.source.setSelectionRange(start, end);
  state.caretLine = diagnostic.line;
  paintGutter();
  paintHighlight();
}

/* Builtins */

function renderBuiltins(info) {
  el.provenance.textContent =
    `Muninn ${info.commit}, compiled to WebAssembly. ` +
    `${info.natives.filter((native) => native.inBrowser).length} of ${info.natives.length} builtins are enabled in this page.`;
  el.budget.textContent =
    `A program here gets ${info.fuel.toLocaleString()} instructions and at most ` +
    `${info.maxTensorElements.toLocaleString()} tensor elements. A loop that runs past the budget ` +
    `traps with the same error it would on the command line.`;

  el.nativeTable.replaceChildren(
    ...info.natives.map((native) => {
      const row = document.createElement("tr");
      const name = document.createElement("td");
      name.textContent = native.name;
      const signature = document.createElement("td");
      signature.textContent = native.detail;
      const available = document.createElement("td");
      const mark = document.createElement("span");
      mark.className = native.inBrowser ? "yes" : "no";
      mark.textContent = native.inBrowser ? "yes" : "traps here";
      available.append(mark);
      row.append(name, signature, available);
      return row;
    }),
  );
}

function showFatal(message) {
  state.fatal = true;
  el.provenance.classList.add("failed");
  el.provenance.textContent = message;
  syncTransport();
}

/* Start */

(async function start() {
  el.run.disabled = true;
  el.inspect.disabled = true;
  el.step.disabled = true;
  el.finish.disabled = true;
  try {
    const info = await ask(ACTION.INFO, undefined);
    if (!info) {
      showFatal(
        "The interpreter did not answer in time, so this page cannot run programs. Reload to try again.",
      );
      return;
    }
    state.booted = true;
    renderBuiltins(info);
    await loadExamples();
  } catch (error) {
    retireWorker();
    showFatal(error.message);
  } finally {
    syncTransport();
  }
})();
