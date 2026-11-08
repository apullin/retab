//! Lets a pass hand the whole graph to stock `_preprocess`, for inputs where stock
//! raises its own error or RETAB can't lower the graph. The pipeline entry points turn
//! a pending request into a `RETAB_STOCK_FALLBACK:` error, which the Python seam
//! catches and answers by calling stock.
//!
//! A panic goes the same way ([`catch_panic`]). It is always a RETAB bug, and stock can
//! still lower the graph, but PyO3 would raise it as `PanicException`, a
//! `BaseException` that the seam's `except Exception` lets through to the user.

use std::any::Any;
use std::cell::{Cell, RefCell};
use std::panic::{self, AssertUnwindSafe};

thread_local! {
    static REQUEST: RefCell<Option<String>> = const { RefCell::new(None) };
    /// How many [`catch_panic`] calls are running on this thread. They nest when a
    /// Python callback that an entry point calls, such as the U55 probe, lowers another
    /// program.
    static CATCHING: Cell<usize> = const { Cell::new(0) };
    /// Where the last panic under [`catch_panic`] happened, from the panic hook.
    static PANIC_LOCATION: RefCell<Option<String>> = const { RefCell::new(None) };
}

/// Request a stock fallback; the first request in a pipeline run wins.
pub fn request(reason: impl Into<String>) {
    REQUEST.with(|slot| {
        slot.borrow_mut().get_or_insert_with(|| reason.into());
    });
}

pub fn take() -> Option<String> {
    REQUEST.with(|slot| slot.borrow_mut().take())
}

/// Run `body`, the work of a Python entry point, and return the fallback reason of a
/// panic in it: `Rust panic: <message> (at <file>:<line>:<column>)`. The panic hook
/// doesn't print such a panic, since the reason carries its message and location.
///
/// A stock fallback request still pending when the panic unwinds is taken and named
/// in the reason, but the panic gives the reason's key: the request alone would have
/// sent the graph to stock, while the panic is a bug to fix.
///
/// `AssertUnwindSafe` holds because the entry points only read the Python objects they
/// get. What a panic can leave half done is RETAB's thread-local per-run state: the
/// pending request, taken here; the stashed IR, which the caller drops; the U55 probe,
/// which `with_transpose_probe` restores on unwind; the serializer's shape and dtype
/// caches, which each serialization clears first; and the unused inputs, which only a
/// finished serialization writes. New per-run state needs the same care: a reset at the
/// start of its run, or a guard that restores it on unwind.
pub fn catch_panic<R>(body: impl FnOnce() -> R) -> Result<R, String> {
    install_panic_hook();
    PANIC_LOCATION.with(|slot| slot.borrow_mut().take());
    CATCHING.with(|depth| depth.set(depth.get() + 1));
    let caught = panic::catch_unwind(AssertUnwindSafe(body));
    CATCHING.with(|depth| depth.set(depth.get() - 1));
    caught.map_err(|payload| {
        let mut reason = format!("Rust panic: {}", panic_message(payload.as_ref()));
        // No location when the panic skipped the hook: `resume_unwind`, which PyO3 uses
        // when a Python callback raises a PanicException.
        if let Some(location) = PANIC_LOCATION.with(|slot| slot.borrow_mut().take()) {
            reason.push_str(&format!(" (at {location})"));
        }
        if let Some(request) = take() {
            reason.push_str(&format!("; pending stock fallback request: {request}"));
        }
        reason
    })
}

fn panic_message(payload: &(dyn Any + Send)) -> &str {
    if let Some(message) = payload.downcast_ref::<&str>() {
        message
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message
    } else {
        "a panic payload that is not a string"
    }
}

/// Install, once, a panic hook that records where a panic under [`catch_panic`]
/// happened instead of printing it. Every other panic goes to the previous hook.
fn install_panic_hook() {
    static INSTALL: std::sync::Once = std::sync::Once::new();
    INSTALL.call_once(|| {
        let previous = panic::take_hook();
        panic::set_hook(Box::new(move |info| {
            // `try_with`: a panic in a thread-local destructor finds the slots gone.
            if !CATCHING.try_with(|depth| depth.get() > 0).unwrap_or(false) {
                return previous(info);
            }
            let location = info
                .location()
                .map(|at| format!("{}:{}:{}", at.file(), at.line(), at.column()));
            let _ = PANIC_LOCATION.try_with(|slot| {
                if let Ok(mut slot) = slot.try_borrow_mut() {
                    *slot = location;
                }
            });
        }));
    });
}

#[cfg(test)]
mod tests {
    use super::catch_panic;

    #[test]
    fn keeps_the_first_request_until_taken() {
        super::take();
        super::request("first: a");
        super::request("second: b");
        assert_eq!(super::take().as_deref(), Some("first: a"));
        assert_eq!(super::take(), None);
    }

    /// The fallback reason of the panic in `body`.
    fn reason_of(body: impl FnOnce() -> u8) -> String {
        catch_panic(body).unwrap_err()
    }

    /// The location `catch_panic` reports for a panic on the line after `line!()`.
    fn next_line(line: u32) -> String {
        format!("(at {}:{}:", file!(), line + 1)
    }

    #[test]
    fn a_panic_becomes_a_reason_with_its_message_and_location() {
        assert_eq!(catch_panic(|| 7), Ok(7));

        let line = line!();
        let reason = reason_of(|| panic!("boom {}", 7));
        assert!(reason.starts_with("Rust panic: boom 7 "), "{reason}");
        assert!(reason.contains(&next_line(line)), "{reason}");

        let items: &[u8] = &[1, 2];
        let line = line!();
        let reason = reason_of(|| items[std::hint::black_box(2)]);
        assert!(
            reason.starts_with("Rust panic: index out of bounds"),
            "{reason}"
        );
        assert!(reason.contains(&next_line(line)), "{reason}");

        let reason = reason_of(|| std::panic::panic_any(7u8));
        assert!(
            reason.starts_with("Rust panic: a panic payload that is not a string (at "),
            "{reason}"
        );
    }

    /// PyO3 resumes a PanicException raised by a Python callback with `resume_unwind`,
    /// which skips the hook: the reason has no location, not the previous panic's.
    #[test]
    fn a_resumed_panic_has_no_location() {
        reason_of(|| panic!("first"));
        let reason = reason_of(|| std::panic::resume_unwind(Box::new(String::from("resumed"))));
        assert_eq!(reason, "Rust panic: resumed");
    }

    /// The panic, not the request, gives the reason's key, and the request is gone
    /// afterwards, so it can't reach the next run.
    #[test]
    fn a_pending_request_is_named_after_the_panic() {
        super::take();
        let reason = reason_of(|| {
            super::request("IrRewriteConv: conv_1: a reason");
            panic!("boom")
        });
        assert!(reason.starts_with("Rust panic: boom (at "), "{reason}");
        assert!(
            reason.ends_with("; pending stock fallback request: IrRewriteConv: conv_1: a reason"),
            "{reason}"
        );
        assert_eq!(super::take(), None);
    }

    /// An entry point that a Python callback runs inside another reports its own panic,
    /// and the outer one goes on to report its own.
    #[test]
    fn nested_catches_report_their_own_panics() {
        let outer = reason_of(|| {
            let line = line!();
            let inner = reason_of(|| panic!("inner"));
            assert!(inner.starts_with("Rust panic: inner "), "{inner}");
            assert!(inner.contains(&next_line(line)), "{inner}");
            panic!("outer")
        });
        assert!(outer.starts_with("Rust panic: outer (at "), "{outer}");
        assert_eq!(super::CATCHING.with(|depth| depth.get()), 0);
    }
}
