import json
from pathlib import Path
from aco_traffic.graph import GridGraph

PRESETS_FILE = r"graphPresets.json"


def load_preset_graph(preset_name: str, presets_path: Path = PRESETS_FILE) -> GridGraph:
    with open(presets_path, "r", encoding="utf-8") as f:
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
