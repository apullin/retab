"""Shared data model for the fast_preprocess analyze/plan/apply rewrite."""

from dataclasses import dataclass


@dataclass(frozen=True)
class _ControlFlowBoundarySite:
    submodule: object
    index: int
    boundary_kind: str
    cf_node: object = None
    placeholder: object = None
    output_node: object = None
    output_arg: object = None

    @classmethod
    def for_input(cls, *, submodule, index, placeholder, cf_node=None):
        return cls(
            submodule=submodule,
            index=index,
            boundary_kind="input",
            cf_node=cf_node,
            placeholder=placeholder,
        )

    @classmethod
    def for_output(
        cls,
        *,
        submodule,
        index,
        output_node,
        output_arg=None,
        cf_node=None,
    ):
        return cls(
            submodule=submodule,
            index=index,
            boundary_kind="output",
            cf_node=cf_node,
            output_node=output_node,
            output_arg=output_arg,
        )


@dataclass(frozen=True)
class _ControlFlowInputBoundaryContract:
    submodule: object
    index: int
    placeholder: object
    input_qparam: object
    output_qparam: object
    site: object = None

    @classmethod
    def from_site(cls, site, *, input_qparam, output_qparam):
        return cls(
            submodule=site.submodule,
            index=site.index,
            placeholder=site.placeholder,
            input_qparam=input_qparam,
            output_qparam=output_qparam,
            site=site,
        )


@dataclass(frozen=True)
class _ControlFlowOutputBoundaryContract:
    submodule: object
    index: int
    target_qparam: object
    site: object = None
    output_arg: object = None

    @classmethod
    def from_site(cls, site, *, target_qparam):
        return cls(
            submodule=site.submodule,
            index=site.index,
            target_qparam=target_qparam,
            site=site,
            output_arg=site.output_arg,
        )


@dataclass(frozen=True)
class _ControlFlowBinaryQparamContract:
    input_qparams: dict[int, object]
    output_qparam: object


@dataclass(frozen=True)
class _ControlFlowBinaryAlignmentEnv:
    parent_args: tuple[object, ...]
    placeholders: tuple[object, ...]
    placeholder_qparams: dict[int, object]
    placeholder_index_by_name: dict[str, int]
    output_args: tuple[object, ...]
    output_qparams_map: dict[int, object]


@dataclass(frozen=True)
class _ControlFlowBinaryOperandSpec:
    arg_idx: int
    arg: object
    source_node: object
    qparam: object
    is_lift: bool = False
    is_scalar_like: bool = False
    preferred_output_rescale: object = None
    preferred_output_qparam: object = None


@dataclass(frozen=True)
class _ControlFlowBinarySiblingAlignment:
    input_qparam: object
    target_qparam: object
    preferred_output_qparam: object = None


@dataclass(frozen=True)
class _ControlFlowBinaryAlignmentPlan:
    submodule: object
    binary_node: object
    operand_specs: dict[int, object]
    env: object = None
    qparam_contract: object = None
    preferred_output_rescale: object = None
    accumulator_qparam: object = None
    mode: str = "repair"


@dataclass(frozen=True)
class _ControlFlowOutputContract:
    submodule_output: object
    submodule_user: object
    desired_fake_tensor: object
    desired_dim_order: object
