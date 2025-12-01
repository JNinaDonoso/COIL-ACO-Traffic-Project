from aco_traffic.graph import GridGraph
from aco_traffic.aco_solver import AcoSolver
from aco_traffic.visualization import plot_solver_result, animate_solver


def main():
    grid = GridGraph(rows=3, cols=3, default_distance=1.0)

    grid.update_edge((0, 0), (0, 1), distance=7.0, traffic=1)
    grid.update_edge((1, 1), (0, 1), distance=10.0, traffic=0.5)
    grid.update_edge((0, 1), (0, 2), distance=7.0)

    start = (0, 0)
    waypoints = [(0, 2), (2, 2)]
    end = (2, 0)

    solver = AcoSolver(
        graph=grid,
        start=start,
        waypoints=waypoints,
        end=end,
        heuristic="traffic",
        num_ants=30,
        num_iterations=60,
        alpha=1.0,
        beta=2.0,
        rho=0.3,
        Q=1.0,
        max_steps_per_ant=40,
        alpha1=1.0,
        alpha2=1.0,
        alpha3=1.0,
        num_elites=3,
        elitist_weight=1.0,
    )

    solver.run(verbose=True)

    print("\n=== ACO RESULT ===")
    print("Best path found:", solver.best_path)
    print("Best cost:", solver.best_cost)

    plot_solver_result(solver, title="ACO Best Route")
    animate_solver(solver, interval=200)


if __name__ == "__main__":
    main()
