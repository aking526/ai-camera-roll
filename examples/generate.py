"""Run with: uv run python examples/generate.py"""

import os
from pathlib import Path

from ai_camera_roll import OpenCodeGoLLM, generate_camera_roll


def main() -> None:
    llm = OpenCodeGoLLM(model=os.environ["OPENCODE_GO_MODEL"])
    result = generate_camera_roll(
        "Maya is a 24-year-old architecture student in Philadelphia. She loves "
        "weekend bike rides, cooking for friends, and sketching neighborhood "
        "buildings. She wears a faded green cap with a mountain patch.",
        llm=llm,
    )
    destination = Path("output/camera_roll.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    print(f"Saved {len(result.camera_roll.photos)} photo ideas to {destination}")


if __name__ == "__main__":
    main()
