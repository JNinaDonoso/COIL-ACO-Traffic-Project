from aco_traffic.graph import GridGraph
from aco_traffic.visualization import interactive_aco_demo
from pathlib import Path
import json

#path to the presets file
PRESETS_FILE = Path("graphPresets.json")


def load_preset_graph(preset_name: str) -> GridGraph:
    with open(PRESETS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    presets = {preset["name"]: preset for preset in data["presets"]}

    if preset_name not in presets:
        available = ", ".join(presets.keys())
        raise ValueError(f"Unknown preset '{preset_name}'. Available presets: {available}")

    preset = presets[preset_name]
    grid = GridGraph(
        rows=preset["rows"],
        cols=preset["cols"],
        default_distance=preset.get("default_distance", 1.0),
    )

    for edge in preset.get("edges", []):
        start = tuple(edge["from"])
        end = tuple(edge["to"])
        kwargs = {}
        if "traffic" in edge:
            kwargs["traffic"] = edge["traffic"]
        if "distance" in edge:
            kwargs["distance"] = edge["distance"]
        grid.update_edge(start, end, **kwargs)

    return grid


def list_presets():
    with open(PRESETS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [p["name"] for p in data["presets"]]


def main():
    presets = list_presets()
    print("Available graph presets:")
    for i, name in enumerate(presets, start=1):
        print(f"{i}. {name}")

    choice = input("Select a preset by number (default 1): ").strip()
    choice_idx = int(choice) - 1 if choice else 0

    if not (0 <= choice_idx < len(presets)):
        raise ValueError("Invalid preset selection.")

    preset_name = presets[choice_idx]
    print(f"Loading preset: {preset_name}")

    grid = load_preset_graph(preset_name)
    interactive_aco_demo(graph=grid)


if __name__ == "__main__":
    main()
