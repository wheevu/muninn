use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};

use muninn::bytecode::{BytecodeModule, Chunk, Constant, FunctionBytecode, OpCode};
use muninn::span::Span;
use muninn::vm::Vm;
use muninn::Value;

struct CountingAllocator;

static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
thread_local! {
    static COUNTING_ENABLED: Cell<bool> = const { Cell::new(false) };
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        COUNTING_ENABLED.with(|enabled| {
            if enabled.get() {
                ALLOCATIONS.fetch_add(1, Ordering::SeqCst);
            }
        });
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        COUNTING_ENABLED.with(|enabled| {
            if enabled.get() {
                ALLOCATIONS.fetch_add(1, Ordering::SeqCst);
            }
        });
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[test]
fn scalar_loop_execution_does_not_allocate_after_capacity_reservation() {
    let module = scalar_loop_module();
    let mut vm = Vm::new(module);
    vm.reserve_runtime_capacity(64, 8);

    for _ in 0..8 {
        assert!(vm.step_instruction().expect("warm step").is_none());
    }

    ALLOCATIONS.store(0, Ordering::SeqCst);
    COUNTING_ENABLED.with(|enabled| enabled.set(true));
    for _ in 0..64 {
        assert!(vm.step_instruction().expect("counted step").is_none());
    }
    COUNTING_ENABLED.with(|enabled| enabled.set(false));
    let allocations = ALLOCATIONS.load(Ordering::SeqCst);
    let value = vm.run().expect("vm run");

    assert!(matches!(value, Value::Int(1000)));
    assert_eq!(allocations, 0);
}

#[test]
fn ordinary_function_calls_do_not_allocate_after_capacity_reservation() {
    let module = function_call_module();
    let mut vm = Vm::new(module);
    vm.reserve_runtime_capacity(128, 32);

    for _ in 0..10 {
        assert!(vm.step_instruction().expect("warm step").is_none());
    }

    ALLOCATIONS.store(0, Ordering::SeqCst);
    COUNTING_ENABLED.with(|enabled| enabled.set(true));
    for _ in 0..64 {
        assert!(vm.step_instruction().expect("counted step").is_none());
    }
    COUNTING_ENABLED.with(|enabled| enabled.set(false));
    let allocations = ALLOCATIONS.load(Ordering::SeqCst);
    let value = vm.run().expect("vm run");

    assert!(matches!(value, Value::Int(32)));
    assert_eq!(allocations, 0);
}

fn scalar_loop_module() -> BytecodeModule {
    let span = Span::default();
    let mut chunk = Chunk::new();
    let zero = chunk.add_constant(Constant::Int(0)).expect("zero");
    let one = chunk.add_constant(Constant::Int(1)).expect("one");
    let limit = chunk.add_constant(Constant::Int(1000)).expect("limit");

    chunk.write_op(OpCode::Constant, span);
    chunk.write_u16(zero, span);
    chunk.write_op(OpCode::SetLocal, span);
    chunk.write_u16(0, span);

    let loop_start = chunk.code.len();
    chunk.write_op(OpCode::GetLocal, span);
    chunk.write_u16(0, span);
    chunk.write_op(OpCode::Constant, span);
    chunk.write_u16(limit, span);
    chunk.write_op(OpCode::Less, span);

    chunk.write_op(OpCode::JumpIfFalse, span);
    let exit_patch = chunk.code.len();
    chunk.write_u16(u16::MAX, span);
    chunk.write_op(OpCode::Pop, span);

    chunk.write_op(OpCode::GetLocal, span);
    chunk.write_u16(0, span);
    chunk.write_op(OpCode::Constant, span);
    chunk.write_u16(one, span);
    chunk.write_op(OpCode::Add, span);
    chunk.write_op(OpCode::SetLocal, span);
    chunk.write_u16(0, span);

    chunk.write_op(OpCode::Loop, span);
    let loop_jump = chunk.code.len().saturating_sub(loop_start) + 2;
    chunk.write_u16(loop_jump as u16, span);

    let exit_jump = chunk.code.len().saturating_sub(exit_patch + 2);
    let exit_bytes = (exit_jump as u16).to_le_bytes();
    chunk.code[exit_patch] = exit_bytes[0];
    chunk.code[exit_patch + 1] = exit_bytes[1];
    chunk.write_op(OpCode::Pop, span);

    chunk.write_op(OpCode::GetLocal, span);
    chunk.write_u16(0, span);
    chunk.write_op(OpCode::Return, span);

    BytecodeModule {
        functions: vec![FunctionBytecode {
            name: "entry".to_string(),
            arity: 0,
            local_count: 1,
            expects_return_value: true,
            chunk,
        }],
        entry_function: 0,
        globals: Vec::new(),
    }
}

fn function_call_module() -> BytecodeModule {
    let span = Span::default();
    let mut add_chunk = Chunk::new();
    let add_one = add_chunk.add_constant(Constant::Int(1)).expect("one");
    add_chunk.write_op(OpCode::GetLocal, span);
    add_chunk.write_u16(0, span);
    add_chunk.write_op(OpCode::Constant, span);
    add_chunk.write_u16(add_one, span);
    add_chunk.write_op(OpCode::Add, span);
    add_chunk.write_op(OpCode::Return, span);

    let mut entry_chunk = Chunk::new();
    let zero = entry_chunk.add_constant(Constant::Int(0)).expect("zero");
    let limit = entry_chunk.add_constant(Constant::Int(32)).expect("limit");
    let function = entry_chunk
        .add_constant(Constant::Function(0))
        .expect("function");

    entry_chunk.write_op(OpCode::Constant, span);
    entry_chunk.write_u16(zero, span);
    entry_chunk.write_op(OpCode::SetLocal, span);
    entry_chunk.write_u16(0, span);

    let loop_start = entry_chunk.code.len();
    entry_chunk.write_op(OpCode::GetLocal, span);
    entry_chunk.write_u16(0, span);
    entry_chunk.write_op(OpCode::Constant, span);
    entry_chunk.write_u16(limit, span);
    entry_chunk.write_op(OpCode::Less, span);
    entry_chunk.write_op(OpCode::JumpIfFalse, span);
    let exit_patch = entry_chunk.code.len();
    entry_chunk.write_u16(u16::MAX, span);
    entry_chunk.write_op(OpCode::Pop, span);

    entry_chunk.write_op(OpCode::Constant, span);
    entry_chunk.write_u16(function, span);
    entry_chunk.write_op(OpCode::GetLocal, span);
    entry_chunk.write_u16(0, span);
    entry_chunk.write_op(OpCode::Call, span);
    entry_chunk.write_u8(1, span);
    entry_chunk.write_op(OpCode::SetLocal, span);
    entry_chunk.write_u16(0, span);

    entry_chunk.write_op(OpCode::Loop, span);
    let loop_jump = entry_chunk.code.len().saturating_sub(loop_start) + 2;
    entry_chunk.write_u16(loop_jump as u16, span);

    let exit_jump = entry_chunk.code.len().saturating_sub(exit_patch + 2);
    let exit_bytes = (exit_jump as u16).to_le_bytes();
    entry_chunk.code[exit_patch] = exit_bytes[0];
    entry_chunk.code[exit_patch + 1] = exit_bytes[1];
    entry_chunk.write_op(OpCode::Pop, span);

    entry_chunk.write_op(OpCode::GetLocal, span);
    entry_chunk.write_u16(0, span);
    entry_chunk.write_op(OpCode::Return, span);

    BytecodeModule {
        functions: vec![
            FunctionBytecode {
                name: "increment".to_string(),
                arity: 1,
                local_count: 1,
                expects_return_value: true,
                chunk: add_chunk,
            },
            FunctionBytecode {
                name: "entry".to_string(),
                arity: 0,
                local_count: 1,
                expects_return_value: true,
                chunk: entry_chunk,
            },
        ],
        entry_function: 1,
        globals: Vec::new(),
    }
}
