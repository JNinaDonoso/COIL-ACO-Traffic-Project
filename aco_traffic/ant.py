from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Set, Optional
from .graph import GridGraph, Node
from .cost import compute_edge_cost, HeuristicName


def manhattan_distance(a: Node, b: Node) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


@dataclass
class SimpleAnt:
    graph: GridGraph
    start: Node
    waypoints: List[Node]
    end: Node
    heuristic: HeuristicName = "distance"
    max_steps: int = 100
    omega: float = 0.5  # coefficient that controls how inclined is the ant to move to target

    path: List[Node] = field(default_factory=list)
    visited_waypoints: Set[Node] = field(default_factory=set)
    finished: bool = False

    def walk(self) -> None:
        """
        Construct a path in a greedy but goal-directed way:
            At each step, move to the neighbor that minimizes:
                edge_cost + omega * distance_to_target
            Target is the nearest unvisited waypoint, or the end if all waypoints are visited.
        Stop if all waypoints are visited AND we reach the end, or we reach max_steps.
        """
        self.path = [self.start]
        self.visited_waypoints = set()
        self.finished = False

        current = self.start

        if current in self.waypoints:
            self.visited_waypoints.add(current)

        for _ in range(self.max_steps):
            if current == self.end and self._all_waypoints_visited():
                self.finished = True
                break

            neighbours = self.graph.neighbours(current)
            if not neighbours:
                break

            target = self._choose_target()

            best_neighbor: Optional[Node] = None
            best_score: float = float("inf")

            for nb, attrs in neighbours.items():
            #    if len(self.path) >= 2 and nb == self.path[-2] and len(neighbours) > 1:
            #        continue

                edge_cost = compute_edge_cost(attrs, heuristic=self.heuristic)
                dir_cost = manhattan_distance(nb, target)
                total_cost = edge_cost + self.omega * dir_cost

                if total_cost < best_score:
                    best_score = total_cost
                    best_neighbor = nb

            if best_neighbor is None:
                break

            current = best_neighbor
            self.path.append(current)

            if current in self.waypoints:
                self.visited_waypoints.add(current)

    def _all_waypoints_visited(self) -> bool:
        return all(wp in self.visited_waypoints for wp in self.waypoints)

    def _choose_target(self) -> Node:
        """Nearest unvisited waypoint by Manhattan distance, or end if all waypoints are visited."""
        unvisited = [wp for wp in self.waypoints if wp not in self.visited_waypoints]
        if not unvisited:
            return self.end

        return min(unvisited, key=lambda wp: manhattan_distance(self.path[-1], wp))
