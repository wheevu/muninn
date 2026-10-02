//! The playground protocol: compile a program, run it, or step it.
//!
//! Every number the page shows comes from the interpreter this crate embeds.
//! The site holds no model of the language: it draws the listing from
//! [`muninn::disassemble`], the stack from [`Vm::value_stack`], the locals
//! from [`Vm::frame_locals`], and the printed text from [`Vm::take_output`].
//!
//! The state reported after each step is the state *before* the next
//! instruction, which is what a reader stepping through a program expects:
//! the highlighted instruction is the one that is about to run.

use muninn::bytecode::{Instruction, OpCode};
use muninn::error::MuninnError;
use muninn::native::{Captured, NativeFunctionKind};
use muninn::token::TokenKind;
use muninn::vm::VmOptions;
use muninn::{BytecodeModule, HostPolicy, Value, Vm, compile_to_bytecode, disassemble};

use crate::json::{Document, Object};

/// The commit this module was built from, when the build script passed one
/// in. A build without it says so rather than pretending to be pinned.
const COMMIT: &str = match option_env!("MUNINN_COMMIT") {
    Some(commit) => commit,
    None => "an unrecorded commit",
};

/// The instruction budget a program gets in this page.
///
/// A script on a public page is written by whoever is reading it, including a
/// reader who wants to see what an infinite loop does. Fuel is the
/// deterministic bound: the loop traps with a vm-phase error instead of
/// spinning the tab. The CLI sets no default.
const FUEL: u64 = 2_000_000;

/// Tensor element cap for this page.
///
/// The built-in ceiling is ten million elements, which is 80 MB of `f64` in a
/// browser tab. This page runs only the reader's own program, and 8 MB is
/// already far past anything the examples use.
const TENSOR_ELEMENTS: usize = 1_000_000;

/// The most printed bytes one session keeps.
///
/// A program that prints in a loop could otherwise grow the JSON document
/// without bound. Lines past the limit are dropped whole and counted, so the
/// page can say so instead of showing a cut-off line.
const OUTPUT_LIMIT: usize = 65_536;

/// The natives a browser-hosted program may call.
///
/// Everything that reaches the host process is denied, so `args_get`,
/// `env_get`, `fs_read`, `fs_write`, `clock_ms`, and `stdin_read` trap with a
/// policy diagnostic instead of running. This is the same policy object the
/// CLI fills in from the terminal, and the reference table on the page reports
/// this list, so the page never claims a builtin works here when it would trap.
const ALLOWED: [NativeFunctionKind; 9] = [
    NativeFunctionKind::Print,
    NativeFunctionKind::Eprint,
    NativeFunctionKind::Assert,
    NativeFunctionKind::Exit,
    NativeFunctionKind::TensorZeros,
    NativeFunctionKind::TensorFill,
    NativeFunctionKind::TensorReshape,
    NativeFunctionKind::TensorMatmul,
    NativeFunctionKind::TensorSum,
];

/// The token classes the page paints.
///
/// These come from Muninn's own lexer, so the page cannot colour a program
/// the way the compiler reads it. A token the lexer rejects is reported as an
/// error, not painted.
fn token_class(kind: &TokenKind) -> &'static str {
    match kind {
        TokenKind::Fn
        | TokenKind::Let
        | TokenKind::Mut
        | TokenKind::If
        | TokenKind::Else
        | TokenKind::Return
        | TokenKind::While
        | TokenKind::Record
        | TokenKind::Import => "keyword",
        TokenKind::TypeInt
        | TokenKind::TypeFloat
        | TokenKind::TypeBool
        | TokenKind::TypeString
        | TokenKind::TypeTensor
        | TokenKind::TypeVoid => "type",
        TokenKind::IntLiteral(_) | TokenKind::FloatLiteral(_) => "number",
        TokenKind::StringLiteral(_) => "string",
        TokenKind::True | TokenKind::False => "constant",
        TokenKind::Identifier(_) => "plain",
        _ => "plain",
    }
}

/// The token ranges for one program, for the page's editor colouring.
///
/// The page asks for these on every keystroke it settles on, so this is
/// deliberately the cheap half of the frontend: a span list and nothing
/// else. Rendering is the page's job; tokenizing is Muninn's.
pub fn highlight(source: &str) -> String {
    let mut document = Document::new();
    let tokens = muninn::lex_document(source);
    document.object(|root| match tokens {
        Ok(tokens) => {
            root.bool("ok", true);
            root.array("tokens", |list| {
                for token in tokens {
                    if matches!(token.kind, TokenKind::Eof) {
                        continue;
                    }
                    let class = token_class(&token.kind);
                    let span = token.span;
                    if span.line == 0
                        || span.end_line != span.line
                        || span.end_column <= span.column
                    {
                        continue;
                    }
                    list.object(|entry| {
                        entry.number("line", span.line as i64);
                        entry.number("column", span.column as i64);
                        entry.number("endColumn", span.end_column as i64);
                        entry.string("class", class);
                    });
                }
            });
        }
        Err(errors) => {
            root.bool("ok", false);
            root.array("diagnostics", |list| {
                for error in errors {
                    list.object(|entry| {
                        entry.string("phase", error.phase);
                        entry.string("message", &error.message);
                        entry.number("line", error.span.line as i64);
                        entry.number("column", error.span.column as i64);
                        entry.number("endLine", error.span.end_line as i64);
                        entry.number("endColumn", error.span.end_column as i64);
                    });
                }
            });
        }
    })
}

/// What the page needs before it can show anything: what this build is, how
/// far it will run a program, and the whole native table with this host's
/// answer for each entry.
pub fn info() -> String {
    let mut document = Document::new();
    document.object(|root| {
        root.string("schema", "muninn/playground/v1");
        root.string("commit", COMMIT);
        root.number("fuel", FUEL as i64);
        root.number("maxTensorElements", TENSOR_ELEMENTS as i64);
        root.array("natives", |natives| {
            for spec in muninn::native::registered_natives() {
                natives.object(|entry| {
                    entry.string("name", spec.name);
                    entry.string("detail", spec.detail);
                    entry.bool("inBrowser", ALLOWED.contains(&spec.kind));
                });
            }
        });
    })
}

/// A compiled program and the interpreter running it.
struct Session {
    module: BytecodeModule,
    vm: Vm,
    finished: bool,
    value: Option<String>,
    fault: Option<MuninnError>,
    output: String,
    /// 1-based output line numbers whose printed value was a tensor, as the
    /// interpreter marked them. The page never guesses from the text.
    output_tensor_lines: Vec<u32>,
    /// Printed lines the bounded capture dropped, whole lines only.
    output_dropped_lines: usize,
}

impl Session {
    fn start(module: BytecodeModule) -> Self {
        let mut policy = HostPolicy::sandboxed();
        for kind in ALLOWED {
            policy.allow(kind);
        }
        policy.max_steps = Some(FUEL);
        policy.max_tensor_elements = TENSOR_ELEMENTS;
        let mut vm = Vm::new_with_policy_and_options(
            module.clone(),
            policy,
            VmOptions {
                jit_enabled: false,
                ..VmOptions::default()
            },
        );
        vm.capture_output_bounded(OUTPUT_LIMIT);
        Self {
            module,
            vm,
            finished: false,
            value: None,
            fault: None,
            output: String::new(),
            output_tensor_lines: Vec::new(),
            output_dropped_lines: 0,
        }
    }

    /// Folds one instruction's outcome into the session, and drains whatever
    /// the program printed while doing it.
    fn record(&mut self, result: Result<Option<Value>, MuninnError>) {
        let (out, err) = self.vm.take_output();
        self.absorb(out);
        self.absorb(err);
        match result {
            Ok(Some(value)) => {
                self.finished = true;
                self.value = Some(value.stringify());
            }
            Ok(None) => {}
            Err(error) => {
                self.finished = true;
                self.fault = Some(error);
            }
        }
    }

    /// Appends one drained capture to the session output. Tensor line numbers
    /// arrive 1-based within the drained text, so each is offset by the lines
    /// already accumulated.
    fn absorb(&mut self, captured: Captured) {
        let base = self.output.bytes().filter(|byte| *byte == b'\n').count() as u32;
        self.output.push_str(&captured.text);
        self.output_tensor_lines
            .extend(captured.tensor_lines.iter().map(|line| base + line));
        self.output_dropped_lines += captured.dropped_lines;
    }

    fn write_state(&self, root: &mut Object<'_>) {
        root.number("steps", self.vm.steps_used() as i64);
        root.bool("done", self.finished);
        match &self.value {
            Some(value) => {
                root.short_string("value", value);
            }
            None => {
                root.null("value");
            }
        }
        match &self.fault {
            Some(fault) => {
                root.object("error", |error| {
                    error.string("phase", fault.phase);
                    error.string("message", &fault.message);
                    error.number("line", fault.span.line as i64);
                    error.number("column", fault.span.column as i64);
                });
            }
            None => {
                root.null("error");
            }
        }

        let frames = self.vm.frames();
        root.array("frames", |list| {
            for (index, frame) in frames.iter().enumerate() {
                let function = &self.module.functions[frame.function_id];
                let locals = self.vm.frame_locals(index);
                list.object(|entry| {
                    entry.string("function", &function.name);
                    entry.number("id", frame.function_id as i64);
                    entry.number("ip", frame.ip as i64);
                    entry.array("locals", |slots| {
                        for (slot, value) in locals.iter().enumerate() {
                            slots.short_string(&format!("{}: {}", slot, value.stringify()));
                        }
                    });
                });
            }
        });
        root.array("operands", |operands| {
            for value in self.operand_region() {
                operands.short_string(&value.stringify());
            }
        });
        root.bool("outputTruncated", self.output_dropped_lines > 0);
        root.number("outputDroppedLines", self.output_dropped_lines as i64);
        root.array("outputTensorLines", |lines| {
            for line in &self.output_tensor_lines {
                lines.number(*line as i64);
            }
        });
        self.write_next(root, frames);
    }

    /// The innermost frame's operand region, bottom first: the value stack
    /// past that frame's locals. Empty when there are no frames. Locals stay
    /// in the frames array; this is only the values an expression has pushed
    /// since.
    fn operand_region(&self) -> &[Value] {
        let Some(frame) = self.vm.frames().last() else {
            return &[];
        };
        let local_count = self
            .module
            .functions
            .get(frame.function_id)
            .map(|function| function.local_count)
            .unwrap_or(0);
        self.vm
            .value_stack()
            .get(frame.stack_base + local_count..)
            .unwrap_or(&[])
    }

    /// The instruction about to run. Before the first step there are no frames
    /// yet, but the entry function's first instruction is still the one the
    /// page should point at.
    fn write_next(&self, root: &mut Object<'_>, frames: &[muninn::CallFrame]) {
        if self.finished {
            root.null("next");
            return;
        }
        let (function_id, ip) = match frames.last() {
            Some(frame) => (frame.function_id, frame.ip),
            None => (self.module.entry_function, 0),
        };
        let function = &self.module.functions[function_id];
        let found = disassemble(&function.chunk)
            .into_iter()
            .find(|instruction| instruction.ip == ip);
        root.object("next", |next| {
            next.string("function", &function.name);
            next.number("functionId", function_id as i64);
            next.number("ip", ip as i64);
            match &found {
                Some(instruction) => {
                    next.string("name", instruction.name);
                    next.number("line", instruction.span.line as i64);
                    next.number("column", instruction.span.column as i64);
                    match operand_text(instruction) {
                        Some(operand) => {
                            next.short_string("operand", &operand);
                        }
                        None => {
                            next.null("operand");
                        }
                    }
                }
                None => {
                    next.string("name", "past the end of the function");
                    next.number("line", 0);
                    next.number("column", 0);
                    next.null("operand");
                }
            }
        });
    }
}

fn write_listing(root: &mut Object<'_>, module: &BytecodeModule) {
    root.array("functions", |functions| {
        for (id, function) in module.functions.iter().enumerate() {
            let listing = disassemble(&function.chunk);
            functions.object(|entry| {
                entry.string("name", &function.name);
                entry.number("id", id as i64);
                entry.number("arity", function.arity as i64);
                entry.number("locals", function.local_count as i64);
                entry.array("instructions", |instructions| {
                    for instruction in &listing {
                        instructions.object(|row| {
                            row.number("ip", instruction.ip as i64);
                            row.string("name", instruction.name);
                            row.number("line", instruction.span.line as i64);
                            match operand_text(instruction) {
                                Some(operand) => {
                                    row.short_string("operand", &operand);
                                }
                                None => {
                                    row.null("operand");
                                }
                            }
                        });
                    }
                });
            });
        }
    });
}

fn write_diagnostics(root: &mut Object<'_>, errors: &[MuninnError]) {
    root.bool("ok", false);
    root.array("diagnostics", |list| {
        for error in errors {
            list.object(|entry| {
                entry.string("phase", error.phase);
                entry.string("message", &error.message);
                entry.number("line", error.span.line as i64);
                entry.number("column", error.span.column as i64);
                entry.number("endLine", error.span.end_line as i64);
                entry.number("endColumn", error.span.end_column as i64);
            });
        }
    });
}

/// The operand as a reader would write it.
///
/// Only the opcodes whose operand is a constant index resolve one. `Call`'s
/// operand is an argument count and `GetLocal`'s is a slot: reading either as
/// an index would print a constant that has nothing to do with the
/// instruction, which is the kind of wrong-but-plausible a listing must not
/// ship.
fn operand_text(instruction: &Instruction) -> Option<String> {
    let operand = instruction.operand?;
    Some(match instruction.op {
        OpCode::Constant => match &instruction.constant {
            Some(constant) => format!("{constant:?}"),
            None => operand.to_string(),
        },
        OpCode::DefineGlobal | OpCode::GetGlobal | OpCode::SetGlobal | OpCode::GetField => {
            match &instruction.constant {
                Some(constant) => format!("{constant:?}"),
                None => operand.to_string(),
            }
        }
        OpCode::Call => format!("{operand} argument(s)"),
        OpCode::Jump | OpCode::JumpIfFalse => format!("{operand} forward"),
        OpCode::Loop => format!("{operand} back"),
        OpCode::GetLocal | OpCode::SetLocal => format!("slot {operand}"),
        OpCode::BuildRecord => format!("{operand} field(s)"),
        _ => operand.to_string(),
    })
}

/// One loaded program. The page keeps a single host: the interpreter is a
/// state machine, and a second one would only hold a second copy of the
/// program.
pub struct Host {
    source: String,
    session: Option<Session>,
}

impl Host {
    /// An empty host. `const` because the module keeps one in a
    /// `thread_local!` initializer, where the value has to be built at compile
    /// time.
    pub const fn new() -> Self {
        Self {
            source: String::new(),
            session: None,
        }
    }

    pub fn set_source(&mut self, source: &str) {
        self.source = source.to_string();
    }

    pub fn source(&self) -> &str {
        &self.source
    }

    /// Runs to completion. This is the whole language in one call, and what the
    /// Run button does. The finished session is kept, so a step afterwards
    /// reports the same finished state instead of claiming nothing is loaded.
    pub fn run(&mut self) -> String {
        let mut document = Document::new();
        document.object(|root| match compile(&self.source) {
            Ok(mut session) => {
                let outcome = session.vm.run().map(Some);
                session.record(outcome);
                root.bool("ok", session.fault.is_none());
                root.string("output", &session.output);
                write_listing(root, &session.module);
                session.write_state(root);
                self.session = Some(session);
            }
            Err(errors) => {
                write_diagnostics(root, &errors);
                self.session = None;
            }
        })
    }

    /// Compiles without running, and reports the listing the stepper draws.
    pub fn init(&mut self) -> String {
        let mut document = Document::new();
        document.object(|root| match compile(&self.source) {
            Ok(session) => {
                root.bool("ok", true);
                write_listing(root, &session.module);
                session.write_state(root);
                self.session = Some(session);
            }
            Err(errors) => {
                write_diagnostics(root, &errors);
                self.session = None;
            }
        })
    }

    /// Executes exactly one instruction and reports the state after it.
    pub fn step(&mut self) -> String {
        let mut document = Document::new();
        let Some(session) = self.session.as_mut() else {
            return document.object(|root| {
                root.bool("ok", false);
                root.string("message", "Load the program before stepping through it.");
            });
        };
        if !session.finished {
            let outcome = session.vm.step_instruction();
            session.record(outcome);
        }
        document.object(|root| {
            root.bool("ok", session.fault.is_none());
            root.string("output", &session.output);
            session.write_state(root);
        })
    }

    /// Runs the loaded program to the end in one call and reports the same
    /// document a step reports. This replaces the page's old loop of single
    /// steps, which silently stopped after its own step budget.
    pub fn resume(&mut self) -> String {
        let mut document = Document::new();
        let Some(session) = self.session.as_mut() else {
            return document.object(|root| {
                root.bool("ok", false);
                root.string(
                    "message",
                    "Compile the program before running it to the end.",
                );
            });
        };
        if !session.finished {
            let outcome = session.vm.run().map(Some);
            session.record(outcome);
        }
        document.object(|root| {
            root.bool("ok", session.fault.is_none());
            root.string("output", &session.output);
            session.write_state(root);
        })
    }
}

fn compile(source: &str) -> Result<Session, Vec<MuninnError>> {
    let module = compile_to_bytecode(source)?;
    Ok(Session::start(module))
}
