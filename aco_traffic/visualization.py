from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from .graph import GridGraph, Node
from .aco_solver import AcoSolver

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
        _draw_path_with_centered_arrows(
            ax,
            best_path,
            color="blue",
            linewidth=5.0,
        )
        ax.plot([], [], color="blue", linewidth=5.0, label="Best path")

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

def _draw_state_on_axis(
    ax,
    graph: GridGraph,
    pheromone: Optional[Dict[EdgeKey, float]],
    best_path: Optional[List[Node]],
    waypoints: Optional[List[Node]],
    start: Optional[Node],
    end: Optional[Node],
    title: str,
) -> None:
    """Draws a current state on an axis for animation."""

    ax.clear()

    max_tau = None
    if pheromone:
        max_tau = max(pheromone.values())
        if max_tau <= 0:
            max_tau = None

    drawn_edges = set()

    for u in graph.nodes():
        x1, y1 = u[0], u[1]
        for v, attrs in graph.neighbours(u).items():
            if (v, u) in drawn_edges:
                continue
            x2, y2 = v[0], v[1]

            traffic = attrs.traffic
            if traffic >= 0.7:
                edge_color = "red"
            elif traffic >= 0.3:
                edge_color = "orange"
            else:
                edge_color = "grey"

            lw = 1.0
            if pheromone and max_tau:
                tau = pheromone.get((u, v), 0.0)
                lw = 0.5 + 3.0 * (tau / max_tau) # Linear scaling for pheromone tracking

            ax.plot([x1, x2], [y1, y2], linewidth=lw, color=edge_color, alpha=0.7)
            drawn_edges.add((u, v))

    xs = [node[0] for node in graph.nodes()]
    ys = [node[1] for node in graph.nodes()]
    ax.scatter(xs, ys, s=40, color="grey", zorder=3)

    if start:
        ax.scatter(start[0], start[1], s=100, color="green", zorder=5)
    if end:
        ax.scatter(end[0], end[1], s=100, color="purple", zorder=5)
    if waypoints:
        for wp in waypoints:
            ax.scatter(wp[0], wp[1], s=100, color="yellow", edgecolor="black", zorder=5)

    if best_path and len(best_path) >= 2:
        _draw_path_with_centered_arrows(
            ax,
            best_path,
            color="blue",
            linewidth=5.0,
        )
        ax.plot([], [], color="blue", linewidth=5.0, label="Best path")

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

def animate_solver(solver, interval: int = 400, repeat: bool = True) -> FuncAnimation:
    """
    Animate ACO evolution
    """

    if not solver.history_pheromones:
        raise ValueError(
            "No history on solver"
        )

    fig, ax = plt.subplots()

    num_frames = len(solver.history_pheromones)

    def update(frame_idx: int):
        pher = solver.history_pheromones[frame_idx]
        best_path = solver.history_best_paths[frame_idx]
        best_cost = solver.history_best_costs[frame_idx]

        title = (
            f"ACO evolution | iter {frame_idx + 1}/{num_frames} | "
            f"heuristic: {solver.heuristic}, best cost: {best_cost:.3f}"
        )

        _draw_state_on_axis(
            ax=ax,
            graph=solver.graph,
            pheromone=pher,
            best_path=best_path,
            waypoints=solver.waypoints,
            start=solver.start,
            end=solver.end,
            title=title,
        )

    anim = FuncAnimation(
        fig,
        update,
        frames=num_frames,
        interval=interval,
        repeat=repeat,
    )

    plt.tight_layout()
    plt.show()
    return anim


def _draw_interactive_state(
    ax,
    graph: GridGraph,
    start: Optional[Node],
    end: Optional[Node],
    waypoints: List[Node],
    solver: Optional[AcoSolver],
    title_prefix: str = "Interactive ACO",
) -> None:
    """
    Draws the current state for the interactive map
    """

    ax.clear()

    pheromone = solver.pheromone if solver is not None else None
    best_path = solver.best_path if solver is not None else None
    best_cost = solver.best_cost if solver is not None else None
    heuristic = solver.heuristic if solver is not None else None

    max_tau = None
    if pheromone:
        max_tau = max(pheromone.values())
        if max_tau <= 0:
            max_tau = None

    drawn_edges = set()

    for u in graph.nodes():
        x1, y1 = u[0], u[1]
        for v, attrs in graph.neighbours(u).items():
            if (v, u) in drawn_edges:
                continue
            x2, y2 = v[0], v[1]

            traffic = attrs.traffic
            if traffic >= 0.7:
                edge_color = "red"
            elif traffic >= 0.3:
                edge_color = "orange"
            else:
                edge_color = "grey"

            lw = 1.0
            if pheromone and max_tau:
                tau = pheromone.get((u, v), 0.0)
                lw = 0.5 + 3.0 * (tau / max_tau)

            ax.plot([x1, x2], [y1, y2], linewidth=lw, color=edge_color, alpha=0.7)
            drawn_edges.add((u, v))

    xs = [node[0] for node in graph.nodes()]
    ys = [node[1] for node in graph.nodes()]
    ax.scatter(xs, ys, s=40, color="grey", zorder=3)

    if start is not None:
        ax.scatter(start[0], start[1], s=100, color="green", zorder=5)
    if end is not None:
        ax.scatter(end[0], end[1], s=100, color="purple", zorder=5)
    for wp in waypoints:
        ax.scatter(wp[0], wp[1], s=100, color="yellow", edgecolor="black", zorder=5)

    if best_path and len(best_path) >= 2:
        _draw_path_with_centered_arrows(
            ax,
            best_path,
            color="blue",
            linewidth=5.0,
        )
        ax.plot([], [], color="blue", linewidth=5.0, label="Best path")

    if best_cost is not None and heuristic is not None:
        title = f"{title_prefix} | heuristic: {heuristic}, best cost: {best_cost:.3f}"
    else:
        title = title_prefix

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


def interactive_aco_demo(
    graph: GridGraph,
    heuristic: str = "traffic",
    num_ants: int = 20,
    num_iterations: int = 30,
    rho: float = 0.3,
    alpha: float = 1.0,
    beta: float = 2.0,
) -> None:
    """
    Simple GUI that allows selection of endpoints and waypoints
    """

    fig, ax = plt.subplots()
    plt.subplots_adjust(right=0.8)

    state = {
        "start": None,          # type: Optional[Node]
        "end": None,            # type: Optional[Node]
        "waypoints": set(),     # type: set[Node]
        "solver": None,         # type: Optional[AcoSolver]
        "last_animation": None
    }

    def redraw():
        _draw_interactive_state(
            ax=ax,
            graph=graph,
            start=state["start"],
            end=state["end"],
            waypoints=sorted(state["waypoints"]),
            solver=state["solver"],
            title_prefix="Interactive ACO (click to set start/end/waypoints, 'r' to run, 'c' to clear)",
        )
        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes is not ax:
            return
        if event.xdata is None or event.ydata is None:
            return

        x = int(round(event.xdata))
        y = int(round(event.ydata))

        if x < 0 or x >= graph.cols or y < 0 or y >= graph.rows:
            return

        node: Node = (x, y)

        if state["start"] is None:
            state["start"] = node
            print(f"Start set at {node}")
        elif state["end"] is None:
            state["end"] = node
            print(f"End set at {node}")
        else:
            if node == state["start"] or node == state["end"]:
                print("This node is already start or end; cannot be a waypoint.")
                return
            if node in state["waypoints"]:
                state["waypoints"].remove(node)
                print(f"Waypoint removed: {node}")
            else:
                state["waypoints"].add(node)
                print(f"Waypoint added: {node}")

        state["solver"] = None
        redraw()

    def on_key(event):
        if event.key == "c":
            state["start"] = None
            state["end"] = None
            state["waypoints"].clear()
            state["solver"] = None
            print("State cleared.")
            redraw()

        elif event.key == "r":
            if state["start"] is None or state["end"] is None:
                print("Please set start and end before running ACO.")
                return

            print(
                f"Running ACO"
            )

            solver = AcoSolver(
                graph=graph,
                start=state["start"],
                waypoints=sorted(state["waypoints"]),
                end=state["end"],
                heuristic=heuristic,
                num_ants=num_ants,
                num_iterations=num_iterations,
                rho=rho,
                alpha=alpha,
                beta=beta,
            )
            solver.run(verbose=False)
            state["solver"] = solver

            print(f"Best path: {solver.best_path}")
            print(f"Best cost: {solver.best_cost:.3f}")

            redraw()

            anim = animate_solver(solver, interval=400, repeat=True)
            state["last_animation"] = anim

    cid_click = fig.canvas.mpl_connect("button_press_event", on_click)
    cid_key = fig.canvas.mpl_connect("key_press_event", on_key)

    redraw()
    plt.show()

    fig.canvas.mpl_disconnect(cid_click)
    fig.canvas.mpl_disconnect(cid_key)


def _draw_path_with_centered_arrows(
    ax,
    path: List[Node],
    color: str = "blue",
    linewidth: float = 4.0,
    head_width: float = 0.15,
    head_length: float = 0.15,
) -> None:
    """
    Draws a path with arrows
    """
    if not path or len(path) < 2:
        return

    xs = [node[0] for node in path]
    ys = [node[1] for node in path]
    ax.plot(xs, ys, linewidth=linewidth, color=color, zorder=6)

    for (x1, y1), (x2, y2) in zip(path[:-1], path[1:]):
        dx = x2 - x1
        dy = y2 - y1

        xm = x1 + 0.5 * dx
        ym = y1 + 0.5 * dy

        adx = 0.4 * dx
        ady = 0.4 * dy

        ax.arrow(
            xm - 0.5 * adx,
            ym - 0.5 * ady,
            adx,
            ady,
            length_includes_head=True,
            head_width=head_width,
            head_length=head_length,
            linewidth=linewidth * 0.8,
            color=color,
            alpha=0.9,
            zorder=7,
        )