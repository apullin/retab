"""Python helpers for decomposing ops in the FX graph.

Called from Rust (src/fx_decompose.rs) via PyO3. Each function receives
a single FX Node, derives the Graph from it, and replaces the node with
a decomposed subgraph of simpler ops.
"""
import torch


def decompose_layernorm(node):
    """Replace a layer_norm node with mean/sub/var/rsqrt/mul/add pattern.

    node.args = [input, normalized_shape, weight, bias, eps]
    """
    graph = node.graph
    input_node = node.args[0]
    normalized_shape = node.args[1]  # list of ints
    weight = node.args[2] if len(node.args) > 2 else None
    bias = node.args[3] if len(node.args) > 3 else None
    eps = node.args[4] if len(node.args) > 4 else 1e-5

    # Compute which dims to reduce over
    ndim = len(input_node.meta["val"].shape)
    n_norm = len(normalized_shape) if isinstance(normalized_shape, (list, tuple)) else 1
    dims = list(range(ndim - n_norm, ndim))

    with graph.inserting_before(node):
        # mean = input.mean(dims, keepdim=True)
        mean = graph.call_function(torch.ops.aten.mean.dim, args=(input_node, dims, True))
        mean.meta["val"] = torch.ops.aten.mean.dim(input_node.meta["val"], dims, True)

        # diff = input - mean
        diff = graph.call_function(torch.ops.aten.sub.Tensor, args=(input_node, mean))
        diff.meta["val"] = input_node.meta["val"]  # same shape

        # var = (diff * diff).mean(dims, keepdim=True)
        sq = graph.call_function(torch.ops.aten.mul.Tensor, args=(diff, diff))
        sq.meta["val"] = diff.meta["val"]
        var = graph.call_function(torch.ops.aten.mean.dim, args=(sq, dims, True))
        var.meta["val"] = mean.meta["val"]

        # inv_std = rsqrt(var + eps)
        eps_node = graph.call_function(torch.ops.aten.full.default, args=([1], eps))
        eps_node.meta["val"] = torch.ops.aten.full.default([1], eps)
        var_eps = graph.call_function(torch.ops.aten.add.Tensor, args=(var, eps_node))
        var_eps.meta["val"] = var.meta["val"]
        inv_std = graph.call_function(torch.ops.aten.rsqrt.default, args=(var_eps,))
        inv_std.meta["val"] = var_eps.meta["val"]

        # normalized = diff * inv_std
        normalized = graph.call_function(torch.ops.aten.mul.Tensor, args=(diff, inv_std))
        normalized.meta["val"] = diff.meta["val"]

        # result = normalized * weight + bias (if present)
        result = normalized
        if weight is not None and not isinstance(weight, type(None)):
            result = graph.call_function(torch.ops.aten.mul.Tensor, args=(result, weight))
            result.meta["val"] = normalized.meta["val"]
        if bias is not None and not isinstance(bias, type(None)):
            result = graph.call_function(torch.ops.aten.add.Tensor, args=(result, bias))
            result.meta["val"] = normalized.meta["val"]

    node.replace_all_uses_with(result)
    graph.erase_node(node)
    return True


def decompose_softmax(node):
    """Replace softmax(x, dim) with exp(x - max(x)) / sum(exp(x - max(x)))."""
    graph = node.graph
    input_node = node.args[0]
    dim = node.args[1] if len(node.args) > 1 else -1

    with graph.inserting_before(node):
        max_val = graph.call_function(
            torch.ops.aten.amax.default, args=(input_node, [dim], True)
        )
        max_val.meta["val"] = input_node.meta["val"]  # shape approximation
        shifted = graph.call_function(
            torch.ops.aten.sub.Tensor, args=(input_node, max_val)
        )
        shifted.meta["val"] = input_node.meta["val"]
        exp_val = graph.call_function(torch.ops.aten.exp.default, args=(shifted,))
        exp_val.meta["val"] = input_node.meta["val"]
        sum_val = graph.call_function(
            torch.ops.aten.sum.dim_IntList, args=(exp_val, [dim], True)
        )
        sum_val.meta["val"] = max_val.meta["val"]
        recip = graph.call_function(torch.ops.aten.reciprocal.default, args=(sum_val,))
        recip.meta["val"] = sum_val.meta["val"]
        result = graph.call_function(
            torch.ops.aten.mul.Tensor, args=(exp_val, recip)
        )
        result.meta["val"] = input_node.meta["val"]

    node.replace_all_uses_with(result)
    graph.erase_node(node)
    return True


def decompose_batchnorm(node):
    """Replace batch_norm with (x - mean) * rsqrt(var + eps) * weight + bias.

    node.args = [input, weight, bias, running_mean, running_var,
                 training, momentum, eps, ...]
    """
    graph = node.graph
    input_node = node.args[0]
    weight = node.args[1]
    bias = node.args[2]
    running_mean = node.args[3]
    running_var = node.args[4]
    eps = node.args[7] if len(node.args) > 7 else 1e-5

    with graph.inserting_before(node):
        centered = graph.call_function(
            torch.ops.aten.sub.Tensor, args=(input_node, running_mean)
        )
        centered.meta["val"] = input_node.meta["val"]
        eps_node = graph.call_function(torch.ops.aten.full.default, args=([1], eps))
        eps_node.meta["val"] = torch.ops.aten.full.default([1], eps)
        var_eps = graph.call_function(
            torch.ops.aten.add.Tensor, args=(running_var, eps_node)
        )
        var_eps.meta["val"] = (
            running_var.meta["val"]
            if hasattr(running_var, "meta")
            else input_node.meta["val"]
        )
        inv_std = graph.call_function(
            torch.ops.aten.rsqrt.default, args=(var_eps,)
        )
        inv_std.meta["val"] = var_eps.meta["val"]
        normed = graph.call_function(
            torch.ops.aten.mul.Tensor, args=(centered, inv_std)
        )
        normed.meta["val"] = input_node.meta["val"]
        scaled = graph.call_function(
            torch.ops.aten.mul.Tensor, args=(normed, weight)
        )
        scaled.meta["val"] = input_node.meta["val"]
        result = graph.call_function(
            torch.ops.aten.add.Tensor, args=(scaled, bias)
        )
        result.meta["val"] = input_node.meta["val"]

    node.replace_all_uses_with(result)
    graph.erase_node(node)
    return True
