from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button, RadioButtons
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

            ax.plot([x1, x2], [y1, y2], linewidth=2, color=edge_color, alpha=0.7)
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

            ax.plot([x1, x2], [y1, y2], linewidth=2, color=edge_color, alpha=0.7)
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
    initial_heuristic: str = "distance",
    initial_num_ants: int = 80,
    initial_num_iterations: int = 150,
    initial_max_steps: int = 100,
    initial_rho: float = 0.3,
    initial_alpha: float = 1.0,
    initial_beta: float = 2.0,
) -> None:
    """
    More complex interactive program
    """

    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.07, right=0.68, bottom=0.25)

    state = {
        "start": None,          # type: Optional[Node]
        "end": None,            # type: Optional[Node]
        "waypoints": set(),     # type: set[Node]
        "solver": None,         # type: Optional[AcoSolver]
        "last_animation": None
    }

    params = {
        "num_ants": initial_num_ants,
        "num_iterations": initial_num_iterations,
        "max_steps": initial_max_steps,
        "alpha": initial_alpha,
        "beta": initial_beta,
        "rho": initial_rho,
        "heuristic": initial_heuristic,
    }

    # ====== SLIDERS ====== #
    ax_ants = fig.add_axes([0.72, 0.55, 0.22, 0.03])
    slider_ants = Slider(ax_ants, "Ants", 5, 500, valinit=params["num_ants"], valstep=1)

    ax_iters = fig.add_axes([0.72, 0.50, 0.22, 0.03])
    slider_iters = Slider(ax_iters, "Iters", 5, 2000, valinit=params["num_iterations"], valstep=1)

    ax_steps = fig.add_axes([0.72, 0.45, 0.22, 0.03])
    slider_steps = Slider(ax_steps, "Steps", 50, 1000, valinit=params["max_steps"], valstep=1)

    ax_alpha = fig.add_axes([0.72, 0.40, 0.22, 0.03])
    slider_alpha = Slider(ax_alpha, "alpha", 0.0, 5.0, valinit=params["alpha"])

    ax_beta = fig.add_axes([0.72, 0.35, 0.22, 0.03])
    slider_beta = Slider(ax_beta, "beta", 0.0, 5.0, valinit=params["beta"])

    ax_rho = fig.add_axes([0.72, 0.30, 0.22, 0.03])
    slider_rho = Slider(ax_rho, "rho", 0.0, 1.0, valinit=params["rho"])

    # ====== RADIO BUTTONS ====== #
    ax_radio = fig.add_axes([0.72, 0.65, 0.22, 0.12])
    heuristics = ["distance", "traffic", "time"]

    if params["heuristic"] not in heuristics:
        params["heuristic"] = heuristics[0]
    radio_heur = RadioButtons(ax_radio, heuristics, active=heuristics.index(params["heuristic"]))

    # ====== RUN ====== #
    ax_run = fig.add_axes([0.75, 0.20, 0.15, 0.06])
    btn_run = Button(ax_run, "RUN")


    def redraw():
        _draw_interactive_state(
            ax=ax,
            graph=graph,
            start=state["start"],
            end=state["end"],
            waypoints=sorted(state["waypoints"]),
            solver=state["solver"],
            title_prefix="Interactive ACO (click: start/end/waypoints | 'r' or button: run | 'c': clear)",
        )
        fig.canvas.draw_idle()


    def update_ants(val):
        params["num_ants"] = int(slider_ants.val)

    def update_iters(val):
        params["num_iterations"] = int(slider_iters.val)

    def update_steps(val):
        params["max_steps"] = int(slider_steps.val)

    def update_alpha(val):
        params["alpha"] = float(slider_alpha.val)

    def update_beta(val):
        params["beta"] = float(slider_beta.val)

    def update_rho(val):
        params["rho"] = float(slider_rho.val)

    def update_heur(label):
        params["heuristic"] = str(label)

    slider_ants.on_changed(update_ants)
    slider_iters.on_changed(update_iters)
    slider_steps.on_changed(update_steps)
    slider_alpha.on_changed(update_alpha)
    slider_beta.on_changed(update_beta)
    slider_rho.on_changed(update_rho)
    radio_heur.on_clicked(update_heur)


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

    def run_aco():
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
            heuristic=params["heuristic"],
            num_ants=params["num_ants"],
            num_iterations=params["num_iterations"],
            rho=params["rho"],
            alpha=params["alpha"],
            beta=params["beta"],
        )
        solver.run(verbose=False)
        state["solver"] = solver

        print(f"Best path: {solver.best_path}")
        print(f"Best cost: {solver.best_cost:.3f}")

        redraw()

        anim = animate_solver(solver, interval=100, repeat=False)
        state["last_animation"] = anim

    def on_key(event):
        if event.key == "c":
            state["start"] = None
            state["end"] = None
            state["waypoints"].clear()
            state["solver"] = None
            print("State cleared.")
            redraw()
        elif event.key == "r":
            run_aco()

    def on_run_button_clicked(event):
        run_aco()

    cid_click = fig.canvas.mpl_connect("button_press_event", on_click)
    cid_key = fig.canvas.mpl_connect("key_press_event", on_key)
    btn_run.on_clicked(on_run_button_clicked)

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