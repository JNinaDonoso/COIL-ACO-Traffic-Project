from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional
from .graph import GridGraph, Node
from .path_utils import compute_path_cost
from .cost import HeuristicName


EdgeKey = Tuple[Node, Node]

@dataclass
class AcoSolver:
    """
    Core structure for the Ant Colony Optimization solver.
    """

    graph: GridGraph
    start: Node
    waypoints: List[Node]
    end: Node

    # ====== ACO Params ====== #
    num_ants: int = 20
    num_iterations: int = 50
    num_elites: int = 3

    alpha: float = 1.0               # pheromone weight
    beta: float = 2.0                # heuristic weight
    rho: float = 0.5                 # evaporation rate
    Q: float = 1.0                   # pheromone deposit
    initial_pheromone: float = 1.0   # we can later take into account closed streets without initial pheromone
    max_steps_per_ant: int = 100

    # ====== Cost Params ====== #
    heuristic: HeuristicName = "distance"
    alpha1: float = 1.0
    alpha2: float = 1.0
    alpha3: float = 1.0


    pheromone: Dict[EdgeKey, float] = field(init=False, default_factory=dict)
    best_path: Optional[List[Node]] = field(init=False, default=None)
    best_cost: float = field(init=False, default=float("inf"))

    def __post_init__(self) -> None:
        self._initialize_pheromone()

    # ====== PHEROMONES ====== #

    def _initialize_pheromone(self) -> None:
        """
        Initializes pheromone on every edge of the graph with the initial_pheromone value.
        Since the streets are "doble vía", (u,v) and (v,u) have the same pheromone value.
        """
        self.pheromone = {}
        for u in self.graph.nodes():
            for v in self.graph.neighbours(u).keys():
                if (v, u) in self.pheromone:
                    continue
                self.pheromone[(u, v)] = self.initial_pheromone
                self.pheromone[(v, u)] = self.initial_pheromone

    def get_pheromone(self, u: Node, v: Node) -> float:
        """
        Returns pheromone value of an edge
        """
        return self.pheromone.get((u, v), self.initial_pheromone)

    def set_pheromone(self, u: Node, v: Node, value: float) -> None:
        """
        Loads the pheromone symmetrically on the edges
        """
        self.pheromone[(u, v)] = value
        self.pheromone[(v, u)] = value

    def evaporate_pheromone(self) -> None:
        """
        Global pheromone evaporation
        """
        factor = 1.0 - self.rho
        for edge in list(self.pheromone.keys()):
            self.pheromone[edge] *= factor

    def deposit_pheromone_on_path(self, path: List[Node]) -> None:
        """
        Pheromone deposition on a path.
        """
        if len(path) < 2:
            return

        cost = compute_path_cost(
            self.graph,
            path,
            heuristic=self.heuristic,
            alpha1=self.alpha1,
            alpha2=self.alpha2,
            alpha3=self.alpha3,
        )
        if cost <= 0:
            return

        delta_tau = self.Q / cost

        for u, v in zip(path[:-1], path[1:]):
            current_tau = self.get_pheromone(u, v)
            self.set_pheromone(u, v, current_tau + delta_tau)

        if cost < self.best_cost:
            self.best_cost = cost
            self.best_path = list(path)

    def reset_best_solution(self) -> None:
        self.best_cost = float("inf")
        self.best_path = None

    def run(self, verbose: bool = False) -> None:
        """
        Run the ACO procedure for num_iterations.
        """
        from .ant import AcoAnt  # imported here to avoid circular imports

        for iteration in range(self.num_iterations):
            ants: List[AcoAnt] = []

            for _ in range(self.num_ants):
                ant = AcoAnt(
                    solver=self,
                    max_steps=self.max_steps_per_ant,
                )
                ant.construct_solution()
                ants.append(ant)

            self.evaporate_pheromone()

            valid_ants = [ant for ant in ants if ant.finished and len(ant.path) >= 2]

            if not valid_ants:
                if verbose:
                    print(f"Iteration {iteration}: no valid solutions.")
                continue

            valid_ants_sorted = sorted(
                valid_ants,
                key=lambda ant: compute_path_cost(
                    self.graph,
                    ant.path,
                    heuristic=self.heuristic,
                    alpha1=self.alpha1,
                    alpha2=self.alpha2,
                    alpha3=self.alpha3,
                )
            )
            k = min(self.num_elites, len(valid_ants_sorted))
            top_k = valid_ants_sorted[:k]
            best_ant_iter=top_k[0]
            best_cost_iter=compute_path_cost(
                self.graph,
                best_ant_iter.path,
                heuristic=self.heuristic,
                alpha1=self.alpha1,
                alpha2=self.alpha2,
                alpha3=self.alpha3,
            )

            for ant in top_k:
                self.deposit_pheromone_on_path(ant.path)

            if verbose:
                print(
                    f"Iteration {iteration}: best iter cost = {best_cost_iter:.3f}, "
                    f"global best = {self.best_cost:.3f}"
                )