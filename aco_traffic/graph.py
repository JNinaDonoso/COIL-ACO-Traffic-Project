from typing import Dict, Tuple, List


Node = Tuple[int, int]

class GridGraph:
    """
    Undirected grid graph.
    Nodes are (row, col) pairs.
    Edges connect neighbours inside the grid.
    """

    def __init__(self, rows: int, cols: int, default_distance: float = 1.0):
        self.rows = rows
        self.cols = cols
        self.default_distance = default_distance              # default distance to begin testing

        self.adj: Dict[Node, Dict[Node, float]] = {}          # node -> dict(neighbor -> distance)

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

                        self.adj[node][neighbour] = self.default_distance # add default distance in both directions
                        self.adj[neighbour][node] = self.default_distance # assume "doble vía" in every edge

    def nodes(self) -> List[Node]:
        """Returns a list of every node on the graph"""
        return list(self.adj.keys())

    def neighbours(self, node: Node) -> Dict[Node, float]:
        """Returns a dictionary with every neighbour and their distances from a node"""
        return self.adj.get(node, {})

    def get_distance(self, u: Node, v: Node) -> float:
        """Returns the distance of edge (u, v)"""
        return self.adj[u][v]

    def __repr__(self) -> str:
        return f"Grid (rows={self.rows}, cols={self.cols})"
