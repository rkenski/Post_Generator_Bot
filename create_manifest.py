# asset_descriptor.py
"""Automated shape‑descriptor bot (multimodal LLM)

Given a directory of PNG (or SVG) brand assets, this script calls a vision‑
enabled LLM (GPT‑4o, gpt‑4o‑mini, or GPT‑4 Turbo with Vision) to obtain:
    • `description`: short natural‑language summary of what the shape looks like.
    • `use_cases`: array of concise ideas on how the asset might be used in a
       social‑media carousel (e.g. "highlight behind headline", "divider between
       sections", "background texture" …).

It outputs **assets_manifest_detailed.json**, a list of objects:
    {
      "name": "highlight_brush_01",
      "description": "Irregular lime‑green brush stroke, semi‑transparent",
      "use_cases": [
        "Underline to emphasise a word",
        "Background accent behind call‑to‑action",
        "Animated wipe reveal when used in motion graphics"
      ],
      "path": "verde/highlight_brush_01.png"
    }

--------------------------------------------------------------------------
Usage (CLI):
--------------------------------------------------------------------------
$ python asset_descriptor.py \
    --assets_root ./brand_assets \
    --examples_json examples/few_shot_examples.json \
    --output manifest_detailed.json

Requirements:
    pip install openai pillow tqdm

Environment variables:
    OPENAI_API_KEY        – mandatory
    OPENAI_API_MODEL      – optional (default: "gpt‑4o‑mini")

Notes:
* The script streams JPEG‑encoded bytes inside a Data URL as per OpenAI’s
  Chat‑Completions vision spec ([platform.openai.com](https://platform.openai.com/docs/guides/images-vision?utm_source=chatgpt.com)).
* Rate‑limited to 60 req/min by default (OpenAI policy); adjust with
  --rpm flag or tune asyncio.Semaphore.
* Few‑shot examples dramatically improve output quality.  Provide 3‑5 examples
  that pair an image with an ideal description & use cases.
"""

# Set the default model to a specific version
os.environ["OPENAI_API_MODEL"] = "gpt-4.1-2025-04-14"

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import os
import pathlib
import re
import sys
from dataclasses import dataclass
from typing import List, Dict, Sequence

from PIL import Image
from tqdm import tqdm

try:
    from openai import AsyncOpenAI  # OpenAI ≥ 1.15
except ImportError:  # pragma: no cover
    raise SystemExit("pip install --upgrade openai>=1.15")

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class Example:
    """Few‑shot example with image file & target answer."""

    path: str
    description: str
    use_cases: List[str]

@dataclass
class Descriptor:
    name: str
    description: str
    use_cases: List[str]
    path: str  # relative asset path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_IMAGE_MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
}


def _encode_data_url(img_path: str) -> str:
    ext = pathlib.Path(img_path).suffix.lower()
    mime = _IMAGE_MIME.get(ext, "image/png")
    with open(img_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _build_messages(asset_path: str, examples: Sequence[Example]) -> List[Dict]:
    """Build the chat message array with few‑shot pairs plus new query."""
    msgs: List[Dict] = [
        {
            "role": "system",
            "content": (
                "You are a senior brand designer. When shown a transparent PNG shape, "
                "describe what the shape looks like and list up to three possible "
                "use cases for a social‑media post. Respond STRICTLY as valid JSON with "
                "the keys: description (string) and use_cases (array of strings)."
            ),
        }
    ]

    # Few‑shot examples
    for ex in examples:
        msgs.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this shape and suggest use cases:"},
                    {
                        "type": "image_url",
                        "image_url": {"url": _encode_data_url(ex.path)},
                    },
                ],
            }
        )
        msgs.append(
            {
                "role": "assistant",
                "content": json.dumps({"description": ex.description, "use_cases": ex.use_cases}, ensure_ascii=False),
            }
        )

    # Actual query
    msgs.append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this shape and suggest use cases:"},
                {"type": "image_url", "image_url": {"url": _encode_data_url(asset_path)}},
            ],
        }
    )
    return msgs

# ---------------------------------------------------------------------------
# Async worker
# ---------------------------------------------------------------------------

async def _describe_asset(client: AsyncOpenAI, asset_path: str, examples: Sequence[Example], model: str) -> Descriptor:
    messages = _build_messages(asset_path, examples)
    resp = await client.chat.completions.create(model=model, messages=messages, max_tokens=256, timeout=60)

    try:
        data = json.loads(resp.choices[0].message.content)
    except Exception as err:
        raise RuntimeError(f"Invalid JSON from model for {asset_path}: {err}\nRaw: {resp.choices[0].message.content}")

    name = pathlib.Path(asset_path).stem
    rel_path = os.path.relpath(asset_path, start=args.assets_root)

    return Descriptor(name=name, description=data["description"], use_cases=data["use_cases"], path=rel_path.replace(os.sep, "/"))

# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

async def main_async(args: argparse.Namespace) -> None:
    # Load few‑shot examples
    examples: List[Example] = []
    if args.examples_json:
        raw = json.loads(pathlib.Path(args.examples_json).read_text("utf‑8"))
        for item in raw:
            examples.append(Example(path=item["path"], description=item["description"], use_cases=item["use_cases"]))
    if not examples:
        print("⚠️  Running without few‑shot examples – results may be generic.")

    # Gather PNG/SVG paths
    asset_paths = [str(p) for p in pathlib.Path(args.assets_root).rglob("*.png")]
    if not asset_paths:
        sys.exit("No PNG assets found under " + args.assets_root)

    # Async client & semaphore for rate limiting
    client = AsyncOpenAI()
    semaphore = asyncio.Semaphore(args.concurrent_requests)

    results: List[Descriptor] = []

    async def worker(path_: str):
        async with semaphore:
            desc = await _describe_asset(client, path_, examples, model=args.model)
            results.append(desc)

    # Progress bar
    await asyncio.gather(*[worker(p) for p in tqdm(asset_paths, desc="Describing", unit="img")])

    # Save manifest
    manifest = [d.__dict__ for d in results]
    with open(args.output, "w", encoding="utf‑8") as fp:
        json.dump(manifest, fp, ensure_ascii=False, indent=2)
    print(f"\n✅  Wrote {len(manifest)} items → {args.output}")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate rich descriptions & use‑case tags for PNG brand assets via a vision LLM.")
    p.add_argument("--assets_root", required=True, help="Root directory containing colour sub‑folders with PNGs")
    p.add_argument("--examples_json", help="Optional JSON file with few‑shot example mappings")
    p.add_argument("--output", default="assets_manifest_detailed.json")
    p.add_argument("--model", default=os.getenv("OPENAI_API_MODEL", "gpt‑4o‑mini"))
    p.add_argument("--concurrent_requests", type=int, default=5, help="Max concurrent API calls (respect rate limits!)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        sys.exit("Interrupted")
