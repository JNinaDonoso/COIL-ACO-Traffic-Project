from aco_traffic.graph import GridGraph
from aco_traffic.cost import compute_edge_cost


def main():

    grid = GridGraph(rows=3, cols=3, default_distance=1.0)
    print("Graph created:", grid)

    grid.update_edge((0, 0), (0, 1), distance=2.0, time=2.0)       # longer street
    grid.update_edge((1, 1), (1, 2), traffic=0.8, time=1.5)        # high traffic street

    edges_to_test = [((0, 0), (0, 1)), ((1, 1), (1, 2))]

    heuristics = ["distance", "time", "traffic", "mixed"]

    for (u, v) in edges_to_test:
        attrs = grid.get_edge(u, v)
        print(f"\nEdge {u} -> {v} attrs = {attrs}")
        for h in heuristics:
            cost = compute_edge_cost(attrs, heuristic=h, alpha1=1.0, alpha2=1.0, alpha3=1.0)
            print(f"  heuristic={h:8s} -> cost={cost:.3f}")


if __name__ == "__main__":
    main()
