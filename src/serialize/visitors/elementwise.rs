use super::emitter::EmitContext;
use super::*;

/// Register direct unary ops from a flat table of `(OpTarget, Op, Attribute)`.
/// Each entry becomes an inline non-capturing closure coerced to `EmitFn`.
macro_rules! register_unary {
    ($table:expr; $( $target:expr => $op:expr, $attr:expr; )+) => {
        $table.extend([
            $( ($target, (|ctx: &EmitContext| ctx.emit_unary($op, $attr)) as EmitFn), )+
        ]);
    };
}

/// Register direct binary ops from a flat table of `(OpTarget, Op, Attribute)`.
macro_rules! register_binary {
    ($table:expr; $( $target:expr => $op:expr, $attr:expr; )+) => {
        $table.extend([
            $( ($target, (|ctx: &EmitContext| ctx.emit_binary($op, $attr)) as EmitFn), )+
        ]);
    };
}

/// Register bool-output binary ops from a flat table of
/// `(OpTarget, Op, Attribute, swap_inputs)`.
macro_rules! register_bool_binary {
    ($table:expr; $( $target:expr => $op:expr, $attr:expr, $swap:expr; )+) => {
        $table.extend([
            $( ($target, (|ctx: &EmitContext| ctx.emit_bool_binary($op, $attr, $swap)) as EmitFn), )+
        ]);
    };
}

mod comparison;
mod logical;
mod unary;

pub fn register(table: &mut EmitterTable) {
    for register in [
        unary::register as fn(&mut EmitterTable),
        logical::register,
        comparison::register,
    ] {
        register(table);
    }
}
