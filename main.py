from aco_traffic.graph import GridGraph
from aco_traffic.aco_solver import AcoSolver


def main():
    grid = GridGraph(rows=4, cols=4, default_distance=1.0)

    grid.update_edge((0, 0), (0, 1), distance=7.0)
    grid.update_edge((0, 1), (0, 2), distance=7.0)

    start = (0, 0)
    waypoints = [(0, 3), (2, 2)]
    end = (0, 0)

    solver = AcoSolver(
        graph=grid,
        start=start,
        waypoints=waypoints,
        end=end,
        heuristic="distance",
        num_ants=40,
        num_iterations=50,
        num_elites=5,
        alpha=1.0,
        beta=4.0,
        rho=0.2,
        Q=1.0,
        max_steps_per_ant=50,
        alpha1=1.0,
        alpha2=1.0,
        alpha3=1.0,
    )

    solver.run(verbose=True)

    print("\n=== ACO RESULT ===")
    print("Best path found:", solver.best_path)
    print("Best cost:", solver.best_cost)


if __name__ == "__main__":
    main()
