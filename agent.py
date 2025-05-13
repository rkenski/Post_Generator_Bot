# instagram_carousel.py — v0.2
"""AI Agent for Generating On‑Brand Instagram Carousels (revised)

Changelog v0.2 (2025‑05‑07)
──────────────────────────
• **Asset catalogue** – 209 brand PNG/SVG shape assets can now be loaded from a
  folder or a JSON manifest.  The agent scores each asset against the slide copy
  (Portuguese‑aware) and picks the best‑matching ones.
• **LLM‑guided asset picking** – if OpenAI GPT‑4o is available, we ask for the
  top‑3 assets per slide using zero‑shot reasoning; otherwise a bag‑of‑words
  fallback scoring is used.
• **Pillow layout engine** – assets are composed on the canvas behind/around the
  text while respecting safe zones; simple anchor presets added.
• **Multi‑language style** – content‑planning prompt now passes through the
  original copy verbatim so line‑break decisions match Portuguese punctuation.
• **CLI** – new --assets_dir / --assets_manifest flags.

Prereqs (requirements.txt excerpt):
    pillow>=10.3.0
    requests>=2.32
    openai>=1.15
    python‑multilingual‑tokenizer>=0.1  # tiny wrapper for basic tokenizer

"""
from __future__ import annotations

import argparse
import io
import json
import os
import pathlib
import random
import re
import textwrap
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import requests
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class BrandKit:
    primary_colors: List[str]
    fonts: Dict[str, str]  # style → font path/URL
    logos: Dict[str, str]  # variant → URL
    extra_assets: Dict[str, str]

@dataclass
class Asset:
    """Visual element (PNG/SVG) that can decorate a slide."""

    name: str
    tags: List[str]
    path_or_url: str

@dataclass
class SlidePlan:
    slide: int
    heading: str
    body: str
    suggested_assets: List[str] = field(default_factory=list)

# ---------------------------------------------------------------------------
# Canva API helpers
# ---------------------------------------------------------------------------
CANVA_BASE = "https://api.canva.com/v1"

def _canva_headers(token: str) -> Dict[str, str]:
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

def fetch_brand_kit(token: str, brand_id: str) -> BrandKit:
    resp = requests.get(f"{CANVA_BASE}/brands/{brand_id}/kit", headers=_canva_headers(token), timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return BrandKit(
        primary_colors=[c["hex"] for c in data["colors"]],
        fonts={f["style"]: f["download_url"] for f in data["fonts"]},
        logos={l["variant"]: l["download_url"] for l in data.get("logos", [])},
        extra_assets={a["name"]: a["download_url"] for a in data.get("assets", [])},
    )

# ---------------------------------------------------------------------------
# Asset catalogue & selection
# ---------------------------------------------------------------------------

def load_asset_catalogue(assets_dir: str | None = None, manifest: str | None = None) -> List[Asset]:
    """Load assets from a directory (filename‑based tags) or a JSON manifest."""
    catalogue: List[Asset] = []
    if manifest:
        data = json.loads(pathlib.Path(manifest).read_text("utf‑8"))
        for item in data:
            catalogue.append(Asset(name=item["name"], tags=item["tags"], path_or_url=item["path"]))
    elif assets_dir:
        for fp in pathlib.Path(assets_dir).glob("*.png"):
            tags = re.split(r"[_\-]", fp.stem)  # crude tokenisation on filename
            catalogue.append(Asset(name=fp.stem, tags=tags, path_or_url=str(fp)))
    else:
        raise ValueError("Provide either assets_dir or manifest")
    return catalogue

# ------------- Asset scoring helpers ------------- #

try:
    from openai import OpenAI  # type: ignore
    _OPENAI = OpenAI()
except ModuleNotFoundError:
    _OPENAI = None  # offline mode


def pick_assets_for_slide(copy_text: str, assets: Sequence[Asset], top_k: int = 3) -> List[Asset]:
    """Return the best‑fitting decorative assets for this copy block."""
    if _OPENAI:
        # Ask GPT‑4o to choose asset names (fast, accurate)
        sys_prompt = "Você é um designer de marcas. Escolha os 3 elementos visuais que melhor reforçam a mensagem do slide, considerando as opções disponíveis (nomes dos arquivos). Responda apenas com uma lista em JSON: [\"asset1\", \"asset2\", ...]."
        asset_names = [a.name for a in assets]
        payload = {
            "model": "gpt‑4o‑mini",
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": f"COPIA DO SLIDE:\n{copy_text}\n\nOPÇÕES DE ASSET:\n{', '.join(asset_names)}"},
            ],
            "temperature": 0.2,
        }
        try:
            r = _OPENAI.chat.completions.create(**payload)  # type: ignore[attr-defined]
            chosen = json.loads(r.choices[0].message.content)
            return [a for a in assets if a.name in chosen][:top_k]
        except Exception:
            # fall back if LLM fails
            pass

    # Offline fallback – bag‑of‑words Jaccard similarity
    from collections import Counter
    import math

    def tokenise(s: str) -> set[str]:
        return set(re.findall(r"[\wÀ-ÿ]+", s.lower()))

    tokens = tokenise(copy_text)
    scored = []
    for a in assets:
        score = len(tokens.intersection(set(a.tags))) / (len(tokens) + 1e-3)
        scored.append((score, a))
    scored.sort(reverse=True, key=lambda x: x[0])
    return [a for _, a in scored[:top_k]]

# ---------------------------------------------------------------------------
# LLM content plan (minor localisation tweaks)
# ---------------------------------------------------------------------------
OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"

def create_content_plan(prompt: str, style_md: str, copies: List[str]) -> List[SlidePlan]:
    system_prompt = (
        "Você é um diretor de arte senior. Gere um plano JSON para um carrossel do Instagram, definindo para cada slide os campos heading (título/subtítulo curto) e body (texto corrido). Use exatamente, sem alterar, o copy fornecido. Segue guia de estilo:\n" + style_md
    )
    payload = {
        "model": "gpt‑4o‑mini",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(copies, ensure_ascii=False)},
        ],
        "temperature": 0.3,
    }
    r = requests.post(OPENAI_CHAT_URL, headers={"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}", "Content-Type": "application/json"}, json=payload, timeout=60)
    r.raise_for_status()
    raw_plan = json.loads(r.json()["choices"][0]["message"]["content"])
    plan: List[SlidePlan] = []
    for idx, p in enumerate(raw_plan, start=1):
        plan.append(SlidePlan(slide=idx, heading=p["heading"], body=p["body"]))
    return plan

# ---------------------------------------------------------------------------
# Background generation (unchanged)
# ---------------------------------------------------------------------------

def generate_background_images(num: int, brand: BrandKit) -> List[Image.Image]:
    from openai import OpenAI  # type: ignore

    client = OpenAI()
    imgs: List[Image.Image] = []
    for _ in range(num):
        try:
            prompt = f"Abstract minimal background in brand palette {', '.join(brand.primary_colors[:2])}, soft shapes, flat design, 4K"
            res = client.images.generate(prompt=prompt, n=1, size="1024x1024")
            img_bytes = requests.get(res.data[0].url, timeout=30).content
            imgs.append(Image.open(io.BytesIO(img_bytes)).convert("RGBA"))
        except Exception:
            imgs.append(Image.new("RGBA", (1080, 1080), brand.primary_colors[0]))
    return imgs

# ---------------------------------------------------------------------------
# Layout & composition
# ---------------------------------------------------------------------------

_ANCHORS = [(60, 60), (800, 50), (50, 700), (700, 700)]  # where to drop decorative assets


def compose_slide(plan: SlidePlan, bg: Image.Image, brand: BrandKit, catalogue: Sequence[Asset]) -> Image.Image:
    """Compose slide with copy and chosen decorative assets."""
    slide = bg.resize((1080, 1080)).copy()
    draw = ImageDraw.Draw(slide)

    # Heading
    heading_font_path = _ensure_font(brand, "heading")
    heading_font = ImageFont.truetype(heading_font_path, size=92)
    draw.text((80, 180), plan.heading, font=heading_font, fill=brand.primary_colors[1])

    # Body text (auto‑wrap to 35 chars)
    body_font_path = _ensure_font(brand, "body")
    body_font = ImageFont.truetype(body_font_path, size=48)
    wrapped = textwrap.fill(plan.body, width=35)
    draw.text((80, 380), wrapped, font=body_font, fill="#3d3d3d")

    # Decorative assets
    chosen_assets = pick_assets_for_slide(plan.heading + " " + plan.body, catalogue)
    plan.suggested_assets = [a.name for a in chosen_assets]
    for anchor, asset in zip(_ANCHORS, chosen_assets):
        try:
            asset_img = _open_asset(asset.path_or_url, max_size=350)
            slide.alpha_composite(asset_img, dest=anchor)
        except Exception:
            continue

    # Logo bottom‑right
    logo_path = download_asset(brand.logos.get("primary"))
    if logo_path:
        logo = Image.open(logo_path).convert("RGBA").resize((120, 120))
        slide.alpha_composite(logo, dest=(900, 900))

    return slide


def _open_asset(path_or_url: str, max_size: int = 350) -> Image.Image:
    if re.match(r"https?://", path_or_url):
        data = requests.get(path_or_url, timeout=30).content
        img = Image.open(io.BytesIO(data)).convert("RGBA")
    else:
        img = Image.open(path_or_url).convert("RGBA")
    # scale if needed
    scale = max(img.width, img.height) / max_size
    if scale > 1:
        img = img.resize((int(img.width / scale), int(img.height / scale)))
    return img


def _ensure_font(brand: BrandKit, style: str) -> str:
    url = brand.fonts.get(style)
    if not url:
        raise ValueError(f"Missing font style {style}")
    cache = pathlib.Path(".cache/fonts")
    cache.mkdir(parents=True, exist_ok=True)
    dst = cache / pathlib.Path(url).name
    if not dst.exists():
        dst.write_bytes(requests.get(url, timeout=30).content)
    return str(dst)


def download_asset(url: str | None) -> str | None:
    if not url:
        return None
    cache = pathlib.Path(".cache/assets")
    cache.mkdir(parents=True, exist_ok=True)
    dst = cache / pathlib.Path(url).name
    if not dst.exists():
        dst.write_bytes(requests.get(url, timeout=30).content)
    return str(dst)

# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def export_slides(slides: List[Image.Image], out_dir: str) -> List[str]:
    out_path = pathlib.Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    files = []
    for i, s in enumerate(slides, 1):
        fname = out_path / f"slide_{i:02d}.png"
        s.save(fname, format="PNG", optimize=True)
        files.append(str(fname))
    return files

# ---------------------------------------------------------------------------
# CLI glue
# ---------------------------------------------------------------------------

def build_carousel(prompt_json: str, style_file: str, brand_id: str, assets_dir: str | None, assets_manifest: str | None, out_dir: str) -> None:
    token = os.environ["CANVA_API_KEY"]
    brand = fetch_brand_kit(token, brand_id)

    # Input prompt & copies
    prompt_data = json.loads(pathlib.Path(prompt_json).read_text("utf‑8"))
    copies = prompt_data["copies"]
    main_prompt = prompt_data.get("prompt", "")

    style_md = pathlib.Path(style_file).read_text("utf‑8")

    plan = create_content_plan(main_prompt, style_md, copies)
    bgs = generate_background_images(len(plan), brand)

    catalogue = load_asset_catalogue(assets_dir, assets_manifest)

    slides = [compose_slide(p, bg, brand, catalogue) for p, bg in zip(plan, bgs)]
    exported = export_slides(slides, out_dir)

    print("\nExported slides:")
    for f in exported:
        print(" •", f)

    # Dump plan + chosen assets for auditing
    with open(pathlib.Path(out_dir) / "slide_plan_with_assets.json", "w", encoding="utf‑8") as fp:
        json.dump([p.__dict__ for p in plan], fp, ensure_ascii=False, indent=2)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate on‑brand Instagram carousel")
    p.add_argument("--prompt_json", required=True)
    p.add_argument("--style_guideline", required=True)
    p.add_argument("--brand_id", required=True)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--assets_dir")
    group.add_argument("--assets_manifest")
    p.add_argument("--output_dir", default="./output")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    build_carousel(
        prompt_json=args.prompt_json,
        style_file=args.style_guideline,
        brand_id=args.brand_id,
        assets_dir=args.assets_dir,
        assets_manifest=args.assets_manifest,
        out_dir=args.output_dir,
    )

if __name__ == "__main__":
    main()

# Import os module if needed elsewhere in the code
import os
# API credentials should be loaded from environment variables or a secure configuration file
# rather than hardcoded in the source code


'''
args = argparse.Namespace(
    prompt_json="data/aqi/prompts/prompts.json",
    style_guideline="data/aqi/style_guidelines/style_guidelines.md",
    brand_id="aqi",
    assets_dir="data/aqi/assets",
    assets_manifest="data/aqi/assets_manifest.json",
    output_dir="data/aqi/output"
)
'''
'''
Example:
python agent.py \
    --prompt_json data/aqi/prompts/prompts.json \
    --style_guideline data/aqi/style_guidelines/style_guidelines.md \
    --brand_id aqi \
    --assets_manifest data/aqi/assets_manifest_prod.json \
    --output_dir data/aqi/output
'''
'''