"""Generate images from a saved result: uv run python examples/generate_images.py"""

import argparse
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

from ai_camera_roll import GenerationResult, OpenCodeGoLLM
from ai_camera_roll.image_pipeline import generate_images
from ai_camera_roll.images import CloudflareImageModel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("output/camera_roll.yaml"))
    parser.add_argument("--output", type=Path, default=Path("output/images"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    load_dotenv(".env", override=False)
    result = GenerationResult.model_validate(
        yaml.safe_load(args.source.read_text(encoding="utf-8"))
    )
    llm = OpenCodeGoLLM(
        model=os.environ["OPENCODE_GO_MODEL"], timeout=300, max_tokens=8192
    )
    generate_images(
        result,
        llm=llm,
        image_model=CloudflareImageModel(),
        output_dir=args.output,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
