from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
from .graph import GridGraph, Node

EdgeKey = Tuple[Node, Node]


def plot_grid_and_path(
    graph: GridGraph,
    best_path: Optional[List[Node]] = None,
    pheromone: Optional[Dict[EdgeKey, float]] = None,
    waypoints: Optional[List[Node]] = None,
    start: Optional[Node] = None,
    end: Optional[Node] = None,
    best_cost: Optional[float] = None,
    heuristic: Optional[str] = None,
    title: str = "ACO Route",
) -> None:
    """
    Draws the grid, the best path and identifies waypoints
    """

    fig, ax = plt.subplots()

    if best_cost is not None and heuristic is not None:
        title = f"{title} | heuristic: {heuristic}, best cost: {best_cost:.3f}"

    drawn_edges = set()

    for u in graph.nodes():
        x1, y1 = u[0], u[1]
        for v, attrs in graph.neighbours(u).items():
            if (v, u) in drawn_edges:
                continue
            x2, y2 = v[0], v[1]

            traffic = attrs.traffic
            if traffic >= 0.7:
                edge_color = "red"  # high traffic
            elif traffic >= 0.3:
                edge_color = "orange"  # medium traffic
            else:
                edge_color = "grey"  # low traffic

            ax.plot([x1, x2], [y1, y2], linewidth=2.0, color=edge_color, alpha=0.7)
            drawn_edges.add((u, v))

    xs = [node[0] for node in graph.nodes()]
    ys = [node[1] for node in graph.nodes()]
    ax.scatter(xs, ys, s=40, color="grey", zorder=3)

    if start:
        ax.scatter(start[0], start[1], s=100, color="green", zorder=5, label="Start")

    if end:
        ax.scatter(end[0], end[1], s=100, color="purple", zorder=5, label="End")

    if waypoints:
        for wp in waypoints:
            ax.scatter(wp[0], wp[1], s=100, color="yellow", edgecolor="black", zorder=5)
        ax.scatter([], [], s=100, color="yellow", edgecolor="black", label="Waypoints")


    if best_path and len(best_path) >= 2:
        px = [node[0] for node in best_path]
        py = [node[1] for node in best_path]
        ax.plot(px, py, linewidth=5.0, color="blue", zorder=4, label="Best path")

    ax.plot([], [], color="grey", linewidth=2, label="Low traffic")
    ax.plot([], [], color="orange", linewidth=2, label="Medium traffic")
    ax.plot([], [], color="red", linewidth=2, label="High traffic")

    ax.set_title(title)
    ax.set_aspect("equal", "box")
    ax.grid(True, alpha=0.3)

    try:
        rows = graph.rows
        cols = graph.cols
        ax.set_xticks(list(range(cols)))
        ax.set_yticks(list(range(rows)))
    except AttributeError:
        pass

    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.05, 0.5),
        borderaxespad=0.
    )
    plt.tight_layout()
    plt.show()


def plot_solver_result(solver, title: str = "ACO Best Route") -> None:
    """Draws directly from an AcoSolver."""
    plot_grid_and_path(
        graph=solver.graph,
        best_path=solver.best_path,
        pheromone=solver.pheromone,
        waypoints=solver.waypoints,
        start=solver.start,
        end=solver.end,
        best_cost=solver.best_cost,
        heuristic=solver.heuristic,
        title=title,
    )
