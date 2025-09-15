from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple, Iterator
import yaml
import numpy as np

@dataclass
class HierNode:
    name: str
    description: Optional[str] = None
    router_instruction: Optional[str] = None
    children: List['HierNode'] = field(default_factory=list)
    parent: Optional['HierNode'] = field(default=None, repr=False)
    path: str = ""  # slash-joined from root (excluding root's own name for brevity)

    # Prototypes built from data
    proto_desc_vec: Optional[np.ndarray] = None   # embedding of description/label
    proto_centroid_vec: Optional[np.ndarray] = None  # centroid of training examples mapped to this node
    example_indices: List[int] = field(default_factory=list)  # indices into training dataset belonging to this node (leaf)

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def iter_nodes(self) -> Iterator['HierNode']:
        yield self
        for c in self.children:
            yield from c.iter_nodes()

    def by_path(self, path: str) -> Optional['HierNode']:
        if self.path == path:
            return self
        for c in self.children:
            r = c.by_path(path)
            if r is not None:
                return r
        return None

    def depth(self) -> int:
        d, p = 0, self.parent
        while p is not None:
            d += 1
            p = p.parent
        return d

def build_tree_from_dict(d: Dict[str, Any], parent: Optional[HierNode]=None, trail: List[str]=None) -> HierNode:
    if trail is None:
        trail = []
    node = HierNode(
        name=d.get('name', 'node'),
        description=d.get('description'),
        router_instruction=d.get('router_instruction')
    )
    node.parent = parent
    node.path = "/".join([*trail, node.name]) if trail else node.name
    for child in d.get('children', []) or []:
        cnode = build_tree_from_dict(child, parent=node, trail=[*trail, node.name])
        node.children.append(cnode)
    return node

def load_hierarchy(yaml_path: str) -> HierNode:
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    root_dict = data.get('root') or data
    root = build_tree_from_dict(root_dict, parent=None, trail=[])
    return root

def iter_leaves(node: HierNode) -> Iterator[HierNode]:
    for n in node.iter_nodes():
        if n.is_leaf():
            yield n

def path_to_list(path: str) -> List[str]:
    return [p for p in path.split('/') if p]

def common_depth(path_a: str, path_b: str) -> int:
    a, b = path_to_list(path_a), path_to_list(path_b)
    k = 0
    for x, y in zip(a, b):
        if x == y:
            k += 1
        else:
            break
    return k
