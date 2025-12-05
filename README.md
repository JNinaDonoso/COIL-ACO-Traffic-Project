# COIL-ACO Traffic Optimization Project

## Project Overview

This project implements an **Ant Colony Optimization (ACO)** algorithm for traffic-aware route planning on a grid-based road network. The system finds optimal paths between user-defined start and end points while considering traffic conditions, distance, and time constraints. Users can interactively set waypoints and visualize the algorithm's evolution in real-time.

## Video Link

**[Project Demo Video](#)** *https://drive.google.com/file/d/1GZkkYz7V-Z3o1jA7OtFIPWfIDwB9tPTy/view*

---

## Project Structure

```
COIL-ACO-Traffic-Project/
├── code/                    # All source code files
│   ├── aco_traffic/         # Core ACO algorithm package
│   │   ├── __init__.py
│   │   ├── aco_solver.py    # Main ACO solver implementation
│   │   ├── ant.py           # Ant agent implementations
│   │   ├── cost.py          # Cost/heuristic functions
│   │   ├── graph.py         # Grid graph data structure
│   │   ├── path_utils.py    # Path computation utilities
│   │   └── visualization.py # Matplotlib visualization & interactive demo
│   ├── main.py              # Main entry point
│   └── gridGraph.py         # Grid graph preset loader
├── data/                    # Configuration and input data
│   └── graphPresets.json    # Predefined graph configurations
├── tests/                   # Test scripts
├── docs/                    # Documentation and screenshots
├── report/                  # Final report document
└── README.md                # This file
```

---
### Prerequisites

- **Python 3.8+**
- **pip** (Python package manager)

### Installation Steps

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd COIL-ACO-Traffic-Project
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   # or
   venv\Scripts\activate     # On Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install matplotlib
   ```

---

## How to Run

### Running the Interactive Demo

Navigate to the project root directory and run:

```bash
cd code
python main.py
```

### Using the Interactive Interface

1. **Select a preset graph** from the available options (5x5, 10x10, or 15x15 grids with various traffic levels)

2. **Set Start Point:** Click on any node to set the starting location (green marker)

3. **Set End Point:** Click on another node to set the destination (purple marker)

4. **Add Waypoints (optional):** Click additional nodes to add intermediate waypoints (yellow markers)

5. **Adjust Parameters:** Use the sliders on the right panel to tune:
   - **Ants:** Number of ants per iteration (5-300)
   - **Iters:** Number of ACO iterations (5-500)
   - **alpha:** Pheromone influence weight (0-5)
   - **beta:** Heuristic influence weight (0-5)
   - **rho:** Evaporation rate (0-1)

6. **Select Heuristic:** Choose between `distance`, `traffic`, or `time` using the radio buttons

7. **Run ACO:** Press the **RUN** button or the **'r'** key to execute the algorithm

8. **Clear:** Press **'c'** to reset and start over

### Understanding Traffic Visualization

- **Grey edges:** Low traffic (< 30%)
- **Orange edges:** Medium traffic (30-70%)
- **Red edges:** High traffic (> 70%)
- **Blue path:** Best route found by ACO

---

## Reproducing Results

1. Run `python main.py` from the `code/` directory
2. Select preset `5x5_high_traffic` for a quick demonstration
3. Click node (0,0) as start and (4,4) as end
4. Use default parameters and click RUN
5. Observe how ACO avoids high-traffic edges when using the `traffic` heuristic

---

## Technologies and Libraries

| Technology | Purpose |
|------------|---------|
| **Python 3.8+** | Core programming language |
| **Matplotlib** | Visualization, interactive plots, and animations |
| **dataclasses** | Clean data structure definitions |
| **pathlib** | Cross-platform file path handling |

---
### Ant Colony Optimization (ACO)

The implementation uses a **rank-based Ant System** with elitist reinforcement:

- **Pheromone Model:** Symmetric pheromone trails on graph edges
- **Transition Rule:** Probabilistic selection based on pheromone (τ) and heuristic (η)
- **Update Strategy:** Top-k ants deposit pheromone with ranked weights
- **Elitist Reinforcement:** Global best path receives additional pheromone

### Heuristics Supported

1. **Distance:** Minimize total path length
2. **Time:** Minimize travel time
3. **Traffic:** Penalize high-traffic edges: `cost = distance × (1 + traffic)`
