from typing import List, Tuple
from .graph import GridGraph, Node
from .cost import compute_edge_cost, HeuristicName


def compute_path_cost(
    graph: GridGraph,
    path: List[Node],
    heuristic: HeuristicName = "distance",
    alpha1: float = 1.0,
    alpha2: float = 1.0,
    alpha3: float = 1.0,
) -> float:
    """
    Compute the total cost of a path as the sum of the edge costs
    between consecutive nodes.
    """
    if len(path) < 2:
        return 0.0

    total_cost = 0.0
    for u, v in zip(path[:-1], path[1:]):
        attrs = graph.get_edge(u, v)
        total_cost += compute_edge_cost(
            attrs,
            heuristic=heuristic,
            alpha1=alpha1,
            alpha2=alpha2,
            alpha3=alpha3,
        )
    return total_cost
