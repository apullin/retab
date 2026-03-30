use std::fmt;
use std::io::{self, Write};

pub(crate) fn log_line(args: fmt::Arguments<'_>) {
    let mut stderr = io::stderr().lock();
    let _ = stderr.write_fmt(args);
    let _ = stderr.write_all(b"\n");
    let _ = stderr.flush();
}

pub(crate) fn log_pass_start(runner: &str, index: usize, total: usize, name: &str) {
    log_line(format_args!(
        "[{runner}] starting pass {}/{}: {}",
        index + 1,
        total,
        name
    ));
}

pub(crate) fn log_pass_result(
    runner: &str,
    name: &str,
    modified: bool,
    before: usize,
    after: usize,
) {
    log_line(format_args!(
        "[{runner}] {} — modified={}, nodes: {} -> {}",
        name, modified, before, after
    ));
}
