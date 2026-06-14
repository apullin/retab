use super::*;

pub(super) fn resolve_input_name_from_node_id(graph: &Graph, id: NodeId) -> Option<String> {
    let node = graph.node(id)?;
    if is_getitem_node(node) {
        let Some(source_id) = node.args.first().and_then(Arg::as_node) else {
            return Some(node.name.clone());
        };
        let source = graph.node(source_id)?;
        let output_idx = match node.args.get(1) {
            Some(Arg::Int(idx)) => *idx,
            _ => 0,
        };
        if output_idx == 0 {
            return Some(primary_output_name(graph, source));
        }
    }
    Some(node.name.clone())
}

pub(super) fn collect_input_names(graph: &Graph, node: &Node) -> Vec<String> {
    let mut names = Vec::new();
    for arg in &node.args {
        match arg {
            Arg::Node(id) => {
                if let Some(name) = resolve_input_name_from_node_id(graph, *id) {
                    names.push(name);
                }
            }
            Arg::NodeList(ids) => {
                for id in ids {
                    if let Some(name) = resolve_input_name_from_node_id(graph, *id) {
                        names.push(name);
                    }
                }
            }
            _ => {}
        }
    }
    names
}

pub(super) fn input_name(graph: &Graph, node: &Node, index: usize) -> Option<String> {
    collect_input_names(graph, node).into_iter().nth(index)
}

pub(super) fn is_getitem_node(node: &Node) -> bool {
    matches!(&node.target, OpTarget::Other(target) if target.contains("getitem"))
}

/// Match the Python serializer's multi-output convention: when an op's users
/// are all `getitem` nodes, emit the first real tensor result under the
/// `getitem(..., 0)` node name instead of the tuple-producing node name.
pub(super) fn primary_output_name(graph: &Graph, node: &Node) -> String {
    if node.users.is_empty() {
        return node.name.clone();
    }

    let mut all_getitems = true;
    let mut first_output_alias: Option<String> = None;
    let mut fallback_alias: Option<String> = None;

    for user_id in &node.users {
        let user = match graph.node(*user_id) {
            Some(user) => user,
            None => {
                all_getitems = false;
                break;
            }
        };
        if !is_getitem_node(user) {
            all_getitems = false;
            break;
        }
        if fallback_alias.is_none() {
            fallback_alias = Some(user.name.clone());
        }
        if matches!(user.args.get(1), Some(Arg::Int(0))) && first_output_alias.is_none() {
            first_output_alias = Some(user.name.clone());
        }
    }

    if all_getitems {
        first_output_alias
            .or(fallback_alias)
            .unwrap_or_else(|| node.name.clone())
    } else {
        node.name.clone()
    }
}
