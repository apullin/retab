use super::*;

pub(super) fn emit_bankers_round_with_fill(
    result: &mut EmitResult,
    base_name: &str,
    shift_base: &str,
    input_name: String,
    shape: Vec<i32>,
    base_dtype: DType,
    ir_dtype: crate::graph::ir::DType,
    output_tensor: TosaTensor,
    fill_value: impl Fn(f64) -> f64,
    context: &str,
) -> String {
    let half_name = format!("{base_name}_round_half");
    let one_name = format!("{base_name}_round_one");
    let two_name = format!("{base_name}_round_two");
    let floor_x_name = format!("{base_name}_round_floor_x");
    let frac_name = format!("{base_name}_round_frac");
    let is_tie_name = format!("{base_name}_round_is_tie");
    let plus_half_name = format!("{base_name}_round_plus_half");
    let round_up_name = format!("{base_name}_round_up");
    let half_floor_mul_name = format!("{base_name}_round_half_floor_mul");
    let half_floor_name = format!("{base_name}_round_half_floor");
    let even_floor_name = format!("{base_name}_round_even_floor");
    let floor_is_even_name = format!("{base_name}_round_floor_is_even");
    let floor_plus_one_name = format!("{base_name}_round_floor_plus_one");
    let tie_result_name = format!("{base_name}_round_tie_result");
    let output_name = output_tensor.name.clone();

    for (name, value, label) in [
        (half_name.clone(), 0.5, "half"),
        (one_name.clone(), 1.0, "one"),
        (two_name.clone(), 2.0, "two"),
    ] {
        assert!(
            push_filled_const_if_supported(
                result,
                name,
                shape.clone(),
                ir_dtype,
                fill_value(value),
            ),
            "{context}: unsupported {label}-fill dtype {:?}",
            ir_dtype
        );
    }

    let shift_name = push_shift_const(result, shift_base);

    result
        .unary(
            floor_x_name.clone(),
            shape.clone(),
            base_dtype,
            Op::Floor,
            Attribute::Floor(FloorAttribute {}),
            input_name.clone(),
        )
        .binary(
            frac_name.clone(),
            shape.clone(),
            base_dtype,
            Op::Sub,
            Attribute::Sub(SubAttribute {}),
            input_name.clone(),
            floor_x_name.clone(),
        )
        .binary(
            is_tie_name.clone(),
            shape.clone(),
            DType::Bool,
            Op::Equal,
            Attribute::Equal(EqualAttribute {}),
            frac_name,
            half_name.clone(),
        )
        .binary(
            plus_half_name.clone(),
            shape.clone(),
            base_dtype,
            Op::Add,
            Attribute::Add(AddAttribute {}),
            input_name,
            half_name.clone(),
        )
        .unary(
            round_up_name.clone(),
            shape.clone(),
            base_dtype,
            Op::Floor,
            Attribute::Floor(FloorAttribute {}),
            plus_half_name,
        )
        .ternary(
            half_floor_mul_name.clone(),
            shape.clone(),
            base_dtype,
            Op::Mul,
            Attribute::Mul(MulAttribute { shift: 0 }),
            floor_x_name.clone(),
            half_name,
            shift_name.clone(),
        )
        .unary(
            half_floor_name.clone(),
            shape.clone(),
            base_dtype,
            Op::Floor,
            Attribute::Floor(FloorAttribute {}),
            half_floor_mul_name,
        )
        .ternary(
            even_floor_name.clone(),
            shape.clone(),
            base_dtype,
            Op::Mul,
            Attribute::Mul(MulAttribute { shift: 0 }),
            half_floor_name,
            two_name,
            shift_name,
        )
        .binary(
            floor_is_even_name.clone(),
            shape.clone(),
            DType::Bool,
            Op::Equal,
            Attribute::Equal(EqualAttribute {}),
            floor_x_name.clone(),
            even_floor_name,
        )
        .binary(
            floor_plus_one_name.clone(),
            shape.clone(),
            base_dtype,
            Op::Add,
            Attribute::Add(AddAttribute {}),
            floor_x_name.clone(),
            one_name,
        )
        .ternary(
            tie_result_name.clone(),
            shape.clone(),
            base_dtype,
            Op::Select,
            Attribute::Select(SelectAttribute {}),
            floor_is_even_name,
            floor_x_name,
            floor_plus_one_name,
        );

    result.push_output_op(
        output_tensor,
        Op::Select,
        Attribute::Select(SelectAttribute {}),
        vec![is_tie_name, tie_result_name, round_up_name],
    );

    output_name
}
