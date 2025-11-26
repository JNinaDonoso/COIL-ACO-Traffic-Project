from typing import Literal
from .graph import EdgeAttributes

HeuristicName = Literal["distance", "time", "traffic", "mixed"]


def compute_edge_cost(
    attrs: EdgeAttributes,
    heuristic: HeuristicName = "distance",
    alpha1: float = 1.0,
    alpha2: float = 1.0,
    alpha3: float = 1.0,
) -> float:
    """
    Compute the cost of traversing an edge given its attributes and the chosen heuristic.

    - distance: cost = distance
    - time:     cost = time
    - traffic:  cost = distance * traffic
    - mixed:    cost = alpha1 * distance + alpha2 * time + alpha3 * traffic
    """
    if heuristic == "distance":
        return attrs.distance
    elif heuristic == "time":
        return attrs.time
    elif heuristic == "traffic":
        return attrs.distance * attrs.traffic
    elif heuristic == "mixed":
        return (
            alpha1 * attrs.distance
            + alpha2 * attrs.time
            + alpha3 * attrs.traffic
        )
    else:
        return attrs.distance
