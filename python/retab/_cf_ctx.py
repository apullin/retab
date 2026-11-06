"""The delegate metadata RETAB's preprocess returns (`_delegate_info_meta`)."""


def _flatten_output_nodes(output_arg):
    if isinstance(output_arg, (list, tuple)):
        for item in output_arg:
            yield from _flatten_output_nodes(item)
    elif hasattr(output_arg, "op"):
        yield output_arg


def _collect_delegate_info_meta(graph_module):
    """Each output's fake value and node name, from the edge graph.

    RETAB lowers from the edge graph without changing it, so these are the
    values the partitioner handed over.
    """
    output_node = graph_module.graph.output_node()
    return {
        "output_meta": [
            {"fake_val": node.meta.get("val"), "node_name": node.name}
            for node in _flatten_output_nodes(output_node.args[0])
        ]
    }
