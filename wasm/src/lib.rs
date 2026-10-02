//! Muninn in a browser tab.
//!
//! The playground in `site/` runs programs by calling into this module: it
//! hands over a UTF-8 source buffer, calls an action, and reads back a UTF-8
//! JSON document describing what the interpreter did. The page is static and
//! offline-capable; everything it knows about the language it learned from
//! the interpreter, not from a copy of the language.
//!
//! There is no `main` and no JavaScript binding layer. The page fetches the
//! `.wasm` bytes itself and calls four exports.
//!
//! # Actions
//!
//! | code | meaning |
//! |---|---|
//! | 0 | what this build is, and the native table |
//! | 1 | compile and run to completion |
//! | 2 | compile, report the listing, run nothing |
//! | 3 | run one instruction of the loaded program |
//! | 4 | highlight the source for the editor |
//! | 5 | run the loaded program to the end |
//!
//! # Why an action code instead of JSON
//!
//! The request carries no text except the source, so a JSON parser would be
//! a second thing to keep correct for two integers. The response is generated
//! text, so a writer is all that is needed.

use std::cell::RefCell;

mod json;
mod session;

thread_local! {
    static HOST: RefCell<session::Host> = const { RefCell::new(session::Host::new()) };
    static RESULT: RefCell<String> = const { RefCell::new(String::new()) };
}

const ACTION_INFO: u32 = 0;
const ACTION_RUN: u32 = 1;
const ACTION_INIT: u32 = 2;
const ACTION_STEP: u32 = 3;
const ACTION_HIGHLIGHT: u32 = 4;
const ACTION_RESUME: u32 = 5;

/// Hands `len` bytes of linear memory to the module and returns where they
/// landed. The page writes the program text there before calling an action.
#[unsafe(no_mangle)]
pub extern "C" fn alloc(len: usize) -> *mut u8 {
    let mut buffer = Vec::<u8>::with_capacity(len);
    let pointer = buffer.as_mut_ptr();
    std::mem::forget(buffer);
    pointer
}

/// Releases a buffer from [`alloc`]. The page calls it after it has copied
/// the bytes it wants out.
///
/// # Safety
///
/// `pointer` must be null, or a pointer this module returned from [`alloc`]
/// with the same `len` and not yet released.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dealloc(pointer: *mut u8, len: usize) {
    if pointer.is_null() {
        return;
    }
    drop(unsafe { Vec::from_raw_parts(pointer, 0, len) });
}

/// The program text for the next action. Must be called before the action
/// that should compile it; the stepper keeps the program it already loaded,
/// so calling it does not disturb a session in progress.
///
/// # Safety
///
/// `pointer` must be valid for `len` readable bytes, which for this module
/// means a buffer the page wrote with [`alloc`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn set_source(pointer: *const u8, len: usize) {
    let bytes = unsafe { std::slice::from_raw_parts(pointer, len) };
    let source = match std::str::from_utf8(bytes) {
        Ok(source) => source.to_string(),
        Err(_) => String::from_utf8_lossy(bytes).into_owned(),
    };
    HOST.with(|host| host.borrow_mut().set_source(&source));
}

/// Runs one action and returns a pointer to the JSON result. Read it with
/// [`result_len`] bytes from linear memory. The pointer stays valid until
/// the next call, so the page copies the bytes before doing anything else.
#[unsafe(no_mangle)]
pub extern "C" fn call(action: u32) -> *const u8 {
    let result = HOST.with(|host| {
        // The source is copied out before the mutable borrow: highlighting
        // needs to read it, and holding both borrows at once would panic.
        let source = host.borrow().source().to_string();
        let mut host = host.borrow_mut();
        match action {
            ACTION_INFO => session::info(),
            ACTION_RUN => host.run(),
            ACTION_INIT => host.init(),
            ACTION_STEP => host.step(),
            ACTION_RESUME => host.resume(),
            ACTION_HIGHLIGHT => session::highlight(&source),
            _ => {
                let mut document = json::Document::new();
                document.object(|unknown| {
                    unknown.bool("ok", false);
                    unknown.string("message", "unknown action");
                })
            }
        }
    });
    RESULT.with(|slot| *slot.borrow_mut() = result);
    RESULT.with(|slot| slot.borrow().as_ptr())
}

/// The length in bytes of the last result.
#[unsafe(no_mangle)]
pub extern "C" fn result_len() -> usize {
    RESULT.with(|slot| slot.borrow().len())
}
