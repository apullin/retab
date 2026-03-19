"""Drop-in replacement for TOSABackend._preprocess() using the Rust engine.

Hooks into the real ARM backend pipeline at preprocess() time, where the FX
graph already has ARM annotations from to_edge_transform_and_lower. Replaces
the slow Python pass infrastructure + Python TOSA serializer with:

  1. FX-direct Rust passes (fold Q/DQ, fuse activations, rewrite conv, etc.)
  2. Targeted Python passes only where needed (rescale insertion)
  3. Rust TOSA flatbuffer serialization

The call chain is:
  EthosUBackend.preprocess(ep, compile_specs)
    -> TOSABackend._preprocess(ep, tosa_compile_spec)  <-- we replace this
    -> EthosUBackend._compile_tosa_flatbuffer(tosa_bytes, ...)  (Vela)

By replacing _preprocess, we only swap the TOSA generation step. Vela
compilation still happens normally downstream.
"""

import logging
import time
from typing import List

from executorch.backends.arm.tosa.compile_spec import TosaCompileSpec
from executorch.exir.backend.backend_details import PreprocessResult
from executorch.exir.backend.compile_spec_schema import CompileSpec
from torch.export.exported_program import ExportedProgram

logger = logging.getLogger(__name__)

_ORIGINAL_GEN_EDGE_MANAGER_FOR_PARTITIONERS = None


def _program_contains_targets(program: ExportedProgram, targets: set) -> bool:
    """Return True if the program or control-flow submodules contain any target."""
    from executorch.exir.program import _program as program_mod

    for node in program.graph.nodes:
        if node.op == "call_function" and node.target in targets:
            return True

    for _, submod, _ in program_mod.get_control_flow_submodules(program.graph_module):
        for node in submod.graph.nodes:
            if node.op == "call_function" and node.target in targets:
                return True

    return False


def _install_edge_decomp_fast_path() -> None:
    """Skip redundant EDGE_DO_NOT_DECOMP cleanup decompositions.

    ExecuTorch's EDGE_DO_NOT_DECOMP path always runs a second
    `run_decompositions(default_table)` pass after the preserve-op rewrite
    phase. That second pass is only needed if the first pass actually left any
    preserve-candidate ops in the graph. When none remain, the second pass is a
    no-op that still costs multiple seconds on large models.
    """
    global _ORIGINAL_GEN_EDGE_MANAGER_FOR_PARTITIONERS

    from collections import defaultdict
    import copy

    from executorch.exir.program import _program as program_mod

    current = program_mod._gen_edge_manager_for_partitioners
    if getattr(current, "__arm_tosa_engine_patched__", False):
        return

    _ORIGINAL_GEN_EDGE_MANAGER_FOR_PARTITIONERS = current

    def optimized_gen_edge_manager_for_partitioners(
        partitioner,
        aten_programs,
        config,
        constant_methods,
        generate_etrecord=False,
    ):
        ops_set_to_not_decompose_by_program = defaultdict(list)
        edge_programs = {}
        skipped_second_passes = 0

        for name, program in aten_programs.items():
            # Functionalize program before asking partitioners to preserve ops.
            program = program.run_decompositions({})

            if partitioner is not None:
                partitioners_for_program = partitioner.get(name, [])
                final_ops_to_preserve = set()

                if not partitioners_for_program:
                    table = program_mod._default_decomposition_table()
                    for op in config.preserve_ops:
                        table.pop(op, None)
                    program = program.run_decompositions(table)

                for curr_partitioner in partitioners_for_program:
                    curr_ops_no_decomp, _ = curr_partitioner.ops_to_not_decompose(
                        program
                    )

                    can_skip_using_edge_do_not_decomp = (
                        program_mod._can_skip_using_EDGE_DO_NOT_DECOMP(
                            curr_partitioner, program
                        )
                    )

                    if can_skip_using_edge_do_not_decomp:
                        table = program_mod._default_decomposition_table()
                        ops_needing_preservation = []

                        for op in curr_ops_no_decomp:
                            if table.pop(op, None) is not None:
                                ops_needing_preservation.append(op)

                        program = program.run_decompositions(table)
                        final_ops_to_preserve.update(ops_needing_preservation)
                    else:
                        curr_ops_no_decomp = (
                            program_mod._remove_invalid_ops_for_not_decompose(
                                curr_ops_no_decomp
                            )
                        )

                        table = program_mod._default_decomposition_table()
                        for op in curr_ops_no_decomp:
                            table.pop(op, None)

                        # First pass: preserve this partitioner's candidate ops
                        # from nested decompositions.
                        program = program.run_decompositions(table)

                        if not _program_contains_targets(
                            program, set(curr_ops_no_decomp)
                        ):
                            skipped_second_passes += 1
                            continue

                        temp_partitioner_dict = {name: [curr_partitioner]}
                        preserved_ops = (
                            program_mod._replace_aten_ops_with_transformed_ops(
                                name, program, temp_partitioner_dict
                            )
                            or []
                        )
                        final_ops_to_preserve.update(preserved_ops)

                        # Second pass is only needed when preserve-candidate ops
                        # actually survive the first pass.
                        program = program.run_decompositions(
                            program_mod._default_decomposition_table()
                        )
                        program_mod._restore_transformed_ops_to_aten_ops(program)

                ops_set_to_not_decompose_by_program[name].extend(final_ops_to_preserve)

            edge_programs[name] = program_mod._generate_edge_program(
                config,
                program,
                preserve_ops=ops_set_to_not_decompose_by_program.get(name, []),
            )

        all_ops_not_to_decompose = list(
            set().union(*ops_set_to_not_decompose_by_program.values())
        )
        config._core_aten_ops_exception_list = list(
            set(config._core_aten_ops_exception_list) | set(all_ops_not_to_decompose)
        )

        edge_manager = program_mod.EdgeProgramManager(
            edge_programs,
            constant_methods,
            config,
            all_ops_not_to_decompose,
        )

        if generate_etrecord:
            etrecord = program_mod._create_empty_etrecord()
            etrecord.add_exported_program(aten_programs)
            etrecord.add_edge_dialect_program(copy.deepcopy(edge_manager))
            edge_manager._etrecord = etrecord

        if skipped_second_passes:
            logger.info(
                "[fast_preprocess] skipped %d redundant EDGE_DO_NOT_DECOMP cleanup passes",
                skipped_second_passes,
            )

        return edge_manager

    optimized_gen_edge_manager_for_partitioners.__arm_tosa_engine_patched__ = True
    optimized_gen_edge_manager_for_partitioners.__wrapped__ = current
    program_mod._gen_edge_manager_for_partitioners = (
        optimized_gen_edge_manager_for_partitioners
    )


def _uninstall_edge_decomp_fast_path() -> None:
    global _ORIGINAL_GEN_EDGE_MANAGER_FOR_PARTITIONERS

    if _ORIGINAL_GEN_EDGE_MANAGER_FOR_PARTITIONERS is None:
        return

    from executorch.exir.program import _program as program_mod

    program_mod._gen_edge_manager_for_partitioners = (
        _ORIGINAL_GEN_EDGE_MANAGER_FOR_PARTITIONERS
    )
    _ORIGINAL_GEN_EDGE_MANAGER_FOR_PARTITIONERS = None


# -- Internal implementations ------------------------------------------------


def _serialize_with_python_tosa(
    edge_program: ExportedProgram,
    compile_spec: TosaCompileSpec,
    merged_state_dict: dict,
) -> bytes:
    """Serialize an already-lowered FX graph using ExecuTorch's Python TOSA serializer.

    This bypasses ArmPassManager by temporarily turning its transform hook into
    an identity function. The expensive graph-lowering work has already been
    done by the Rust FX-direct pipeline; this only reuses the known-good Python
    TOSA serializer on the post-pass graph.
    """
    import tosa_serializer as ts

    import torch
    from torch.fx import GraphModule
    from torch.export.graph_signature import InputKind

    from executorch.backends.arm._passes import ArmPassManager
    from executorch.backends.arm.tosa.backend import TOSABackend
    from executorch.backends.transforms.utils import create_constant_placeholder

    def first_meta_val(node):
        val = node.meta.get("val")
        if isinstance(val, (list, tuple)):
            return val[0] if val else None
        return val

    def qparam_dtype(qp):
        if hasattr(qp, "dtype"):
            return qp.dtype
        if isinstance(qp, dict):
            return qp.get("dtype")
        return None

    def normalize_tensor_getattrs() -> int:
        """Convert tensor get_attr nodes into constant placeholders.

        The ExecuTorch Python TOSA serializer only accepts get_attr nodes that
        reference nested GraphModules. Our Rust FX-direct passes still create
        tensor get_attr nodes for some synthetic constants (late-folded FULL,
        TABLE buffers, zero biases). Canonicalize those back into proper
        constant placeholders before invoking the Python serializer.
        """
        graph_module = edge_program.graph_module
        graph = graph_module.graph

        insert_pos = None
        for node in graph.nodes:
            if (
                node.op == "placeholder"
                and node.name in edge_program.graph_signature.user_inputs
            ):
                insert_pos = node
                break
        if insert_pos is None:
            insert_pos = next(iter(graph.nodes), None)
        if insert_pos is None:
            return 0

        converted = 0
        for node in list(graph.nodes):
            if node.op != "get_attr":
                continue

            attr_name = str(node.target)
            attr = getattr(graph_module, attr_name, None)
            if isinstance(attr, GraphModule):
                continue
            if not isinstance(attr, torch.Tensor):
                continue

            placeholder_name = f"{attr_name}_placeholder"
            counter = 0
            existing_names = {n.name for n in graph.nodes}
            while placeholder_name in existing_names:
                counter += 1
                placeholder_name = f"{attr_name}_placeholder_{counter}"

            with graph.inserting_before(insert_pos):
                const_node = create_constant_placeholder(
                    exp_program=edge_program,
                    graph=graph,
                    kind=InputKind.BUFFER,
                    name=placeholder_name,
                    data=attr.detach().clone(),
                    persistent_buffer=True,
                )

            for key, value in node.meta.items():
                if key == "val":
                    continue
                const_node.meta[key] = value
            const_node.meta["arm_constant_placeholder"] = True

            node.replace_all_uses_with(const_node)
            graph.erase_node(node)
            converted += 1

        if converted:
            graph.eliminate_dead_code()
            graph_module.recompile()

        return converted

    def normalize_quantized_tensor_meta() -> int:
        """Fix stale FakeTensor dtypes on quantized-region nodes.

        The Rust FX-direct passes rely on qparam metadata and explicit op args
        more than on node.meta["val"]. The Python serializer is stricter and
        uses meta["val"] as the source of truth for output dtype. Canonicalize
        obviously stale floating-point FakeTensor metadata before serialization.
        """
        fixed = 0
        graph = edge_program.graph_module.graph
        float_dtypes = {
            torch.float16,
            torch.float32,
            torch.float64,
            torch.bfloat16,
        }
        same_dtype_ops = (
            "aten.permute_copy",
            "aten.avg_pool2d",
            "aten.max_pool2d_with_indices",
            "aten.clamp.default",
            "aten.view_copy",
            "aten.add.Tensor",
            "aten.sub.Tensor",
            "aten.mul.Tensor",
            "sum.dim_IntList",
            "tosa.TRANSPOSE",
        )

        for node in graph.nodes:
            if node.op != "call_function":
                continue

            val = first_meta_val(node)
            if val is None or not hasattr(val, "new_empty"):
                continue
            if getattr(val, "dtype", None) not in float_dtypes:
                continue

            target = str(node.target)
            new_dtype = None

            if any(op in target for op in same_dtype_ops):
                if node.args and hasattr(node.args[0], "meta"):
                    input_val = first_meta_val(node.args[0])
                    if input_val is not None:
                        new_dtype = getattr(input_val, "dtype", None)
            elif "tosa.RESCALE" in target:
                if len(node.args) > 1 and isinstance(node.args[1], torch.dtype):
                    new_dtype = node.args[1]
            elif any(
                op in target
                for op in ("tosa.CONV2D", "tosa.DEPTHWISE_CONV2D", "tosa.CONV3D")
            ):
                if node.args and hasattr(node.args[0], "meta"):
                    input_val = first_meta_val(node.args[0])
                    if input_val is not None and getattr(input_val, "dtype", None) in (
                        torch.int8,
                        torch.int16,
                    ):
                        new_dtype = torch.int32
            else:
                output_qparams = node.meta.get("output_qparams")
                if output_qparams:
                    qp = output_qparams.get(0) if hasattr(output_qparams, "get") else None
                    if qp is None:
                        try:
                            qp = output_qparams[0]
                        except Exception:
                            qp = None
                    new_dtype = qparam_dtype(qp)

            if new_dtype is not None and new_dtype != val.dtype:
                node.meta["val"] = val.new_empty(val.shape, dtype=new_dtype)
                fixed += 1

        return fixed

    artifact_path = compile_spec.get_intermediate_path() or ""
    version = compile_spec.tosa_spec.version
    tosa_graph = ts.TosaSerializer(
        artifact_path,
        targetMajor=version.major,
        targetMinor=version.minor,
        targetPatch=version.micro,
        targetDraft=True if version.minor > 0 else False,
    )

    live_state_dict = edge_program.state_dict
    live_snapshot = dict(live_state_dict)
    orig_transform = ArmPassManager.transform_to_backend_pipeline
    ArmPassManager.transform_to_backend_pipeline = (
        lambda self, exported_program, graph_module: graph_module
    )

    try:
        live_state_dict.clear()
        live_state_dict.update(merged_state_dict)
        normalized_getattrs = normalize_tensor_getattrs()
        if normalized_getattrs:
            logger.info(
                "[fast_preprocess] normalized %d tensor get_attr nodes for Python TOSA serializer",
                normalized_getattrs,
            )
        normalized_meta = normalize_quantized_tensor_meta()
        if normalized_meta:
            logger.info(
                "[fast_preprocess] normalized %d quantized node metadata dtypes for Python TOSA serializer",
                normalized_meta,
            )
        TOSABackend._preprocess_module(
            edge_program.graph_module,
            edge_program,
            compile_spec,
            tosa_graph,
            None,
        )
        return bytes(tosa_graph.serialize())
    finally:
        ArmPassManager.transform_to_backend_pipeline = orig_transform
        live_state_dict.clear()
        live_state_dict.update(live_snapshot)


def _fast_preprocess(
    edge_program: ExportedProgram,
    compile_spec: TosaCompileSpec,
) -> PreprocessResult:
    """Lower an exported program to TOSA using the Rust engine.

    Replaces TOSABackend._preprocess. Same signature: (ExportedProgram,
    TosaCompileSpec) -> PreprocessResult.

    The edge_program graph has ARM annotations from partitioning. We run our
    hybrid pipeline: FX-direct Rust passes on the live Python graph, then
    ingest the result into Rust IR for TOSA serialization.
    """
    import arm_tosa_engine

    t0 = time.perf_counter()

    graph_module = edge_program.graph_module
    state_dict = dict(edge_program.state_dict)
    graph_signature = edge_program.graph_signature

    # lower_to_tosa_v2 runs the full hybrid pipeline:
    #   1. FX-direct passes on live Python graph (Rust fold Q/DQ, Python
    #      rescales, table ops, fuse duplicates, clamp, conv, memory format)
    #   2. Ingest post-pass graph into Rust IR
    #   3. Serialize to TOSA flatbuffer
    tosa_bytes = arm_tosa_engine.lower_to_tosa_v2(
        graph_module,
        state_dict,
        graph_signature,
        edge_program,
    )

    t_total = time.perf_counter() - t0
    logger.info(f"[fast_preprocess] TOSA generation: {t_total:.3f}s")

    return PreprocessResult(processed_bytes=bytes(tosa_bytes))


def _full_python_preprocess(
    edge_program: ExportedProgram,
    compile_spec: TosaCompileSpec,
) -> PreprocessResult:
    """Run full Python ArmPassManager passes + Rust serialization.

    Correctness baseline: Python passes are known-correct, and we only
    replace the TOSA serialization with Rust. If this produces correct
    output, any remaining issues are in our Rust passes.
    """
    import arm_tosa_engine

    t0 = time.perf_counter()

    graph_module = edge_program.graph_module

    from executorch.backends.arm._passes import ArmPassManager

    pass_manager = ArmPassManager(compile_spec)
    graph_module = pass_manager.transform_to_backend_pipeline(
        exported_program=edge_program, graph_module=graph_module
    )

    t_passes = time.perf_counter() - t0

    state_dict = dict(edge_program.state_dict)
    graph_signature = edge_program.graph_signature
    tosa_bytes = arm_tosa_engine.lower_to_tosa(
        graph_module, state_dict, graph_signature
    )

    t_total = time.perf_counter() - t0
    logger.info(
        f"[python_passes_preprocess] TOSA generation: {t_total:.3f}s "
        f"(passes={t_passes:.3f}s, serialize={t_total - t_passes:.3f}s)"
    )

    return PreprocessResult(processed_bytes=bytes(tosa_bytes))


def _fast_python_preprocess(
    edge_program: ExportedProgram,
    compile_spec: TosaCompileSpec,
) -> PreprocessResult:
    """Run Python ArmPassManager passes with smart no-op skipping + Rust serialization.

    Same as python_passes mode but installs the smart pass skipping optimization
    which scans graph ops before each pass and skips passes whose target ops are
    absent. Saves ~35-40s by avoiding ExportPass retrace on ~70 no-op passes.
    """
    import arm_tosa_engine
    from fast_pass_manager import install_smart_skipping, uninstall_smart_skipping

    t0 = time.perf_counter()

    install_smart_skipping()
    try:
        graph_module = edge_program.graph_module

        from executorch.backends.arm._passes import ArmPassManager

        pass_manager = ArmPassManager(compile_spec)
        graph_module = pass_manager.transform_to_backend_pipeline(
            exported_program=edge_program, graph_module=graph_module
        )
    finally:
        uninstall_smart_skipping()

    t_passes = time.perf_counter() - t0

    state_dict = dict(edge_program.state_dict)
    graph_signature = edge_program.graph_signature
    tosa_bytes = arm_tosa_engine.lower_to_tosa(
        graph_module, state_dict, graph_signature
    )

    t_total = time.perf_counter() - t0
    logger.info(
        f"[fast_python_preprocess] TOSA generation: {t_total:.3f}s "
        f"(passes={t_passes:.3f}s, serialize={t_total - t_passes:.3f}s)"
    )

    return PreprocessResult(processed_bytes=bytes(tosa_bytes))


def _all_rust_preprocess(
    edge_program: ExportedProgram,
    compile_spec: "TosaCompileSpec",
) -> PreprocessResult:
    """All-Rust: FX-direct passes + Rust TOSA serialization.

    IMPORTANT: After run_fx_passes() transforms the FX graph, we must use
    ingest_and_serialize() (ingest + serialize only) rather than lower_to_tosa()
    (which would re-run the full IR pass pipeline on top of the already-
    transformed graph, causing double rescales, double transposes, etc.).
    """
    import copy
    import os
    import time
    from pathlib import Path

    t0 = time.perf_counter()
    gm = edge_program.graph_module

    # Deep-copy the state_dict BEFORE passes run, because some FX-direct
    # passes (e.g. RewriteConv) may mutate ep.state_dict in-place.
    # We want the original (un-mutated) weight data for pre-existing tensors,
    # but we still need to ingest any new constant placeholders created by the
    # FX passes later in the pipeline.
    original_state_dict = copy.deepcopy(dict(edge_program.state_dict))

    import arm_tosa_engine

    # Run the full FX-direct pipeline (all passes in Rust via PyO3)
    arm_tosa_engine.run_fx_passes(gm, edge_program)
    t_passes = time.perf_counter() - t0

    # Merge in any pass-created constant placeholders added to the live
    # ExportedProgram after the pre-pass snapshot. Keep original entries for
    # existing keys so we do not pick up in-place mutations to legacy weights.
    merged_state_dict = original_state_dict
    for key, value in dict(edge_program.state_dict).items():
        if key not in merged_state_dict:
            merged_state_dict[key] = copy.deepcopy(value)

    serializer_backend = os.environ.get("ARM_TOSA_ENGINE_SERIALIZER", "python")
    if serializer_backend == "rust":
        # Ingest + serialize only -- do NOT re-run IR passes (they already ran
        # as FX-direct passes above).  lower_to_tosa() would run
        # build_phase2_pipeline() again, corrupting the graph with duplicate
        # transformations.
        tosa_bytes = arm_tosa_engine.ingest_and_serialize(
            gm, merged_state_dict, edge_program.graph_signature
        )
    elif serializer_backend == "python":
        tosa_bytes = _serialize_with_python_tosa(
            edge_program, compile_spec, merged_state_dict
        )
    else:
        raise ValueError(
            f"Unknown ARM_TOSA_ENGINE_SERIALIZER={serializer_backend!r}. "
            "Use 'python' or 'rust'."
        )

    dump_path = os.environ.get("ARM_TOSA_ENGINE_DUMP_TOSA")
    if dump_path:
        dump_file = Path(dump_path)
        dump_file.parent.mkdir(parents=True, exist_ok=True)
        dump_file.write_bytes(bytes(tosa_bytes))
        logger.info(f"[all_rust] dumped TOSA flatbuffer to {dump_file}")

    t_total = time.perf_counter() - t0
    logger.info(
        f"[all_rust] passes={t_passes:.3f}s, serialize={t_total-t_passes:.3f}s, total={t_total:.3f}s"
    )

    return PreprocessResult(processed_bytes=bytes(tosa_bytes))


# -- Public API: patching and subclass alternatives --------------------------


def fast_preprocess(
    edge_program: ExportedProgram, compile_specs: List[CompileSpec]
) -> PreprocessResult:
    """Drop-in for TOSABackend.preprocess (public method).

    Parses compile_specs then delegates to _fast_preprocess.
    """
    compile_spec = TosaCompileSpec.from_list(compile_specs)
    return _fast_preprocess(edge_program, compile_spec)


def patch_tosa_backend(mode="fast"):
    """Monkey-patch TOSABackend._preprocess with our replacement.

    This is the effective hook: EthosUBackend.preprocess calls
    TOSABackend._preprocess to generate the TOSA flatbuffer, then feeds
    it to Vela. By patching _preprocess, we replace TOSA generation
    while Vela compilation continues normally.

    Args:
        mode: "fast" uses Rust passes + Rust serialization (default).
              "python_passes" uses Python passes + Rust serialization
              (correctness baseline).
              "python_passes_fast" uses Python passes with smart no-op
              skipping + Rust serialization (~40s savings).
    """
    from executorch.backends.arm.tosa.backend import TOSABackend

    _install_edge_decomp_fast_path()

    if mode == "fast":
        TOSABackend._preprocess = staticmethod(_fast_preprocess)
        TOSABackend.preprocess = staticmethod(fast_preprocess)
        logger.info("[fast_preprocess] Patched TOSABackend (fast mode)")
    elif mode == "python_passes":
        TOSABackend._preprocess = staticmethod(_full_python_preprocess)
        logger.info("[fast_preprocess] Patched TOSABackend (python_passes mode)")
    elif mode == "python_passes_fast":
        TOSABackend._preprocess = staticmethod(_fast_python_preprocess)
        logger.info(
            "[fast_preprocess] Patched TOSABackend (python_passes_fast mode)"
        )
    elif mode == "all_rust":
        TOSABackend._preprocess = staticmethod(_all_rust_preprocess)
        logger.info("[fast_preprocess] Patched TOSABackend (all_rust mode)")
    else:
        raise ValueError(
            f"Unknown mode: {mode!r}. "
            "Use 'fast', 'all_rust', 'python_passes', or 'python_passes_fast'."
        )


def unpatch_tosa_backend():
    """Restore the original TOSABackend._preprocess."""
    import importlib

    import executorch.backends.arm.tosa.backend as mod

    _uninstall_edge_decomp_fast_path()
    importlib.reload(mod)
    logger.info("[fast_preprocess] Restored original TOSABackend")


class FastTOSABackend:
    """Subclass-style alternative to monkey-patching.

    Can be used if a backend swapping mechanism is available.
    """

    @staticmethod
    def preprocess(
        edge_program: ExportedProgram, compile_specs: List[CompileSpec]
    ) -> PreprocessResult:
        return fast_preprocess(edge_program, compile_specs)

    @staticmethod
    def _preprocess(
        edge_program: ExportedProgram,
        compile_spec: TosaCompileSpec,
    ) -> PreprocessResult:
        return _fast_preprocess(edge_program, compile_spec)
