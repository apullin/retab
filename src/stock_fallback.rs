//! Lets a pass hand the whole graph to stock `_preprocess`, for inputs where stock
//! raises its own error or RETAB can't lower the graph. The pipeline entry points turn
//! a pending request into a `RETAB_STOCK_FALLBACK:` error, which the Python seam
//! catches and answers by calling stock.

use std::cell::RefCell;

thread_local! {
    static REQUEST: RefCell<Option<String>> = const { RefCell::new(None) };
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

#[cfg(test)]
mod tests {
    #[test]
    fn keeps_the_first_request_until_taken() {
        super::take();
        super::request("first: a");
        super::request("second: b");
        assert_eq!(super::take().as_deref(), Some("first: a"));
        assert_eq!(super::take(), None);
    }
}
