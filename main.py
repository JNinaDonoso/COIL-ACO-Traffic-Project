from aco_traffic.graph import GridGraph
from aco_traffic.visualization import interactive_aco_demo

def main():
    grid = GridGraph(rows=5, cols=5, default_distance=1.0)

    grid.update_edge((0, 0), (0, 1), traffic=0.8)
    grid.update_edge((3, 3), (3, 4), traffic=0.5)

    interactive_aco_demo(
        graph=grid,
        heuristic="traffic",
        num_ants=150,
        num_iterations=200,
        rho=0.3,
        alpha=1.0,
        beta=2.0,
    )

if __name__ == "__main__":
    main()
