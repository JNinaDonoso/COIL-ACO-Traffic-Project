from aco_traffic.graph import GridGraph


def main():

    grid = GridGraph(rows=3, cols=3, default_distance=1.0)

    print("Graph created:", grid)
    print("Nodes in graph:")
    for node in sorted(grid.nodes()):
        print(" ", node)

    test_nodes = [(0, 0), (1, 1), (2, 2)]
    print("\nNeighbours:")
    for node in test_nodes:
        neighbours = grid.neighbours(node)
        print(f"Node {node}:")
        for nb, dist in neighbours.items():
            print(f" -> {nb}, distance={dist}")

if __name__ == "__main__":
    main()
