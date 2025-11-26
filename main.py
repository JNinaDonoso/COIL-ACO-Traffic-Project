from aco_traffic.graph import GridGraph
from aco_traffic.ant import SimpleAnt
from aco_traffic.path_utils import compute_path_cost


def main():
    grid = GridGraph(rows=3, cols=3, default_distance=1.0)
    print("Graph created:", grid)

    grid.update_edge((0, 0), (0, 1), distance=3.0) # longer street
    grid.update_edge((0, 1), (0, 2), traffic=0.95) # very high traffic street

    start = (0, 0)
    waypoints = [(0, 2), (2, 2)]
    end = (2, 0)

    ant = SimpleAnt(
        graph=grid,
        start=start,
        waypoints=waypoints,
        end=end,
        heuristic="traffic",
        max_steps=50,
    )

    ant.walk()

    print("\nAnt finished walk:")
    print("  Path:", ant.path)
    print("  Visited waypoints:", ant.visited_waypoints)
    print("  Finished condition:", ant.finished)

    total_cost = compute_path_cost(grid, ant.path, heuristic="traffic")
    print(f"  Total path cost (traffic heuristic): {total_cost:.2f}")


if __name__ == "__main__":
    main()
