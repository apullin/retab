use super::emitter::EmitContext;
use super::*;

macro_rules! direct_unary_emitters {
    ($($vis:vis fn $name:ident => ($op:expr, $attr:expr);)+) => {
        $(
            $vis fn $name(ctx: &EmitContext) -> EmitResult {
                ctx.emit_unary($op, $attr)
            }
        )+
    };
}

macro_rules! direct_binary_emitters {
    ($($vis:vis fn $name:ident => ($op:expr, $attr:expr);)+) => {
        $(
            $vis fn $name(ctx: &EmitContext) -> EmitResult {
                ctx.emit_binary($op, $attr)
            }
        )+
    };
}

macro_rules! bool_binary_emitters {
    ($($vis:vis fn $name:ident => ($op:expr, $attr:expr, $swap_inputs:expr);)+) => {
        $(
            $vis fn $name(ctx: &EmitContext) -> EmitResult {
                ctx.emit_bool_binary($op, $attr, $swap_inputs)
            }
        )+
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
