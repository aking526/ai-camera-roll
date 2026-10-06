"""Run with: uv run python examples/generate.py"""

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

from ai_camera_roll import OpenCodeGoLLM, generate_camera_roll


def main() -> None:
    load_dotenv(".env", override=False)
    llm = OpenCodeGoLLM(model=os.environ["OPENCODE_GO_MODEL"])
    result = generate_camera_roll(
        "Maya is a 24-year-old architecture student in Philadelphia. She loves "
        "weekend bike rides, cooking for friends, and sketching neighborhood "
        "buildings. She wears a faded green cap with a mountain patch.",
        llm=llm,
    )
    destination = Path("output/camera_roll.yaml")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        yaml.safe_dump(
            result.model_dump(mode="json"),
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
            width=100,
        ),
        encoding="utf-8",
    )
    print(f"Saved {len(result.camera_roll.photos)} photo ideas to {destination}")


if __name__ == "__main__":
    main()
