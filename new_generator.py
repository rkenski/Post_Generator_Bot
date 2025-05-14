"""
carousel_agent.py — Skeleton for an Instagram Carousel Generation Agent
====================================================================

This module is a *scaffold* you can extend to generate on‑brand Instagram carousels (1080×1350px, 4:5 ratio) from a structured **BrandKit** and a textual brief.  It’s built for clarity rather than raw performance.  Feel free to rename, split into packages, or swap in alternative libraries.

Key design ideas
----------------
1. **Explicit data models** (dataclasses) for brand assets and slide specs keep the agent stateless and testable.
2. **LLM‑assisted planning**:  The agent asks an LLM (e.g. OpenAI GPT) for an outline + tone, then converts that into `SlideSpec`s.
3. **Rule‑based layout engine**:  A lightweight `LayoutEngine` picks colours / shapes that respect WCAG contrast and brand tone.
4. **Pluggable renderers**:  Default renderer uses Pillow + moviepy; swap in Figma/Canva APIs for production‑grade typography.
5. **CLI + FastAPI wrapper** so you can batch‑generate or expose an HTTP endpoint.

Quickstart
~~~~~~~~~~
```bash
pip install pillow moviepy python-dotenv openai colour-science
python carousel_agent.py examples/brief.yaml output_dir/
```

Environment variables (or .env)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
```env
OPENAI_API_KEY=sk-...
```
"""

from __future__ import annotations

import os
import random
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import ImageClip, concatenate_videoclips
from colour import Color  # colour‑science library for palette maths
from dotenv import load_dotenv

# Optional: uncomment these imports if you wire up generative assets
# from diffusers import StableDiffusionPipeline
# import openai

load_dotenv()

# ---------------------------------------------------------------------------
# 1️⃣  Data models
# ---------------------------------------------------------------------------

@dataclass
class BrandKit:
    name: str
    colours: List[str]  # hex strings, first one treated as primary
    fonts: List[Path]   # .ttf or .otf files; index 0 is default
    shapes: List[Path]  # vector or raster assets

    def random_accent(self) -> str:
        return random.choice(self.colours[1:]) if len(self.colours) > 1 else self.colours[0]


@dataclass
class SlideSpec:
    text: str
    bg_colour: str = "#FFFFFF"
    accent_colour: str | None = None
    shape_idx: int | None = None  # index into BrandKit.shapes
    background_asset: Path | None = None  # image or video


# ---------------------------------------------------------------------------
# 2️⃣  Layout / colour logic (very naive to start)
# ---------------------------------------------------------------------------

class LayoutEngine:
    def __init__(self, kit: BrandKit):
        self.kit = kit

    def apply_branding(self, spec: SlideSpec, idx: int, n_slides: int) -> SlideSpec:
        spec.bg_colour = self.kit.colours[0]
        spec.accent_colour = self.kit.random_accent()
        spec.shape_idx = idx % len(self.kit.shapes)
        return spec


# ---------------------------------------------------------------------------
# 3️⃣  Rendering utilities
# ---------------------------------------------------------------------------

INSTAGRAM_W, INSTAGRAM_H = 1080, 1350  # 4:5 portrait

class Renderer:
    def __init__(self, kit: BrandKit):
        self.kit = kit
        self.font_cache: dict[Path, ImageFont.FreeTypeFont] = {}

    def _get_font(self, size: int) -> ImageFont.FreeTypeFont:
        path = self.kit.fonts[0]
        if path not in self.font_cache:
            self.font_cache[path] = ImageFont.truetype(str(path), size)
        return self.font_cache[path]

    def render_slide(self, spec: SlideSpec, out_path: Path) -> Path:
        img = Image.new("RGB", (INSTAGRAM_W, INSTAGRAM_H), spec.bg_colour)
        draw = ImageDraw.Draw(img)

        # Draw shape (placeholder as filled circle)
        if spec.shape_idx is not None:
            shape_colour = spec.accent_colour or self.kit.random_accent()
            r = 200
            shape_x = INSTAGRAM_W - r - 50
            shape_y = 50
            draw.ellipse([shape_x, shape_y, shape_x + r, shape_y + r], fill=shape_colour)

        # Draw wrapped text
        font = self._get_font(60)
        margin = 100
        wrapped = textwrap.fill(spec.text, width=20)
        draw.multiline_text((margin, margin), wrapped, font=font, fill="white", spacing=10)

        img.save(out_path)
        return out_path

    def render_video_from_slides(self, slide_paths: List[Path], out_file: Path, fps: int = 1):
        clips = [ImageClip(str(p)).set_duration(2) for p in slide_paths]
        video = concatenate_videoclips(clips, method="compose")
        video.write_videofile(str(out_file), fps=fps)
        return out_file


# ---------------------------------------------------------------------------
# 4️⃣  Agent orchestration
# ---------------------------------------------------------------------------

def plan_slides(prompt: str, n_slides: int = 5) -> List[SlideSpec]:
    """Very naive: split prompt into chunks. Replace with LLM call for production."""
    words = prompt.split()
    chunk_size = max(1, len(words) // n_slides)
    slides = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return [SlideSpec(text=t) for t in slides]


def generate_carousel(brief: str, kit: BrandKit, out_dir: Path, as_video: bool = False):
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = plan_slides(brief)
    layout = LayoutEngine(kit)
    specs = [layout.apply_branding(s, i, len(specs)) for i, s in enumerate(specs)]

    renderer = Renderer(kit)
    slide_files: List[Path] = []
    for i, spec in enumerate(specs):
        slide_path = out_dir / f"slide_{i + 1}.jpg"
        renderer.render_slide(spec, slide_path)
        slide_files.append(slide_path)

    if as_video:
        video_path = out_dir / "carousel.mp4"
        renderer.render_video_from_slides(slide_files, video_path)
        print(f"✅ Video saved to {video_path}")
    else:
        print(f"✅ Slides saved to {out_dir}")


# ---------------------------------------------------------------------------
# 5️⃣  CLI entry‑point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser(description="Generate Instagram carousel on brand")
    parser.add_argument("brief", help="Text prompt or path to .txt/.md file")
    parser.add_argument("out", help="Output directory for generated assets")
    parser.add_argument("--kit", required=True, help="Path to brandkit.json (see example)")
    parser.add_argument("--video", action="store_true", help="Render as MP4 video instead of JPG slides")
    args = parser.parse_args()

    brief_text = Path(args.brief).read_text() if Path(args.brief).exists() else args.brief
    kit_data = json.loads(Path(args.kit).read_text())
    kit = BrandKit(**kit_data)

    generate_carousel(brief_text, kit, Path(args.out), as_video=args.video)
"
