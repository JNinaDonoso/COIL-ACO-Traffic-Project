from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

Node = Tuple[int, int]

@dataclass
class EdgeAttributes:
    distance: float      # physical length of the street
    traffic: float       # traffic level (probably a coefficient from 0 to 1? we´ll see later)
    time: float          # time used to travel the street

class GridGraph:
    """
    Undirected grid graph.
    Nodes are (row, col) pairs.
    Edges connect neighbours inside the grid.
    Each edge has attributes: distance, traffic, base_time.
    """

    def __init__(self, rows: int, cols: int, default_distance: float = 1.0):
        self.rows = rows
        self.cols = cols
        self.default_distance = default_distance              # default distance to begin testing

        self.adj: Dict[Node, Dict[Node, EdgeAttributes]] = {}      # node -> dict(neighbor -> attributes)

        self._build_grid()

    def _build_grid(self) -> None:
        """Create nodes and edges for a rows x cols grid."""
        for i in range(self.rows):
            for j in range(self.cols):
                node = (i, j)
                if node not in self.adj:
                    self.adj[node] = {}

                neighbours = [
                    (i - 1, j),  # up neighbour
                    (i + 1, j),  # down neighbour
                    (i, j - 1),  # left neighbour
                    (i, j + 1),  # right neighbour
                ]

                for ni, nj in neighbours:
                    if 0 <= ni < self.rows and 0 <= nj < self.cols:
                        neighbour = (ni, nj)

                        if neighbour not in self.adj:
                            self.adj[neighbour] = {}

                        if neighbour not in self.adj[node]:
                            attrs = EdgeAttributes(
                                distance = self.default_distance,
                                traffic = 0.0,
                                time= self.default_distance,  # assuming velocity = 1 initially
                            )

                        self.adj[node][neighbour] = attrs
                        self.adj[neighbour][node] = attrs

    def nodes(self) -> List[Node]:
        """Returns a list of every node on the graph"""
        return list(self.adj.keys())

    def neighbours(self, node: Node) -> Dict[Node, EdgeAttributes]:
        """Returns a dictionary with every neighbour and their attributes from a node"""
        return self.adj.get(node, {})

    def get_edge(self, u: Node, v: Node) -> EdgeAttributes:
        """Return attributes of edge (u, v)"""
        return self.adj[u][v]

    def update_edge(
            self,
            u: Node,
            v: Node,
            distance: Optional[float] = None,
            traffic: Optional[float] = None,
            time: Optional[float] = None,
    ) -> None:
        """
        Update attributes for edge (u, v)
        """
        attrs = self.get_edge(u, v)
        if distance is not None:
            attrs.distance = distance
        if traffic is not None:
            attrs.traffic = traffic
        if time is not None:
            attrs.time = time

    def __repr__(self) -> str:
        return f"Grid (rows={self.rows}, cols={self.cols})"
