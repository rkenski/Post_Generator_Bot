# instagram_carousel.py — v0.3
"""AI Agent for Generating On‑Brand Instagram Carousels (revised)

Changelog v0.3 (2025‑05‑13)
──────────────────────────
• **Local asset management** – Removed Canva API dependency, now uses local
  folders for fonts, logos, and shape assets
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
from openai import OpenAI
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

import requests
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class BrandKit:
    primary_colors: List[str]
    fonts: Dict[str, str]  # style → font path
    logos: Dict[str, str]  # variant → path
    extra_assets: Dict[str, str]

@dataclass
class Asset:
    """Visual element (PNG/SVG) that can decorate a slide."""

    name: str
    description: str
    use_cases: List[str]
    path_or_url: str

@dataclass
class SlidePlan:
    slide: int
    heading: str
    body: str
    suggested_assets: List[str] = field(default_factory=list)

# ---------------------------------------------------------------------------
# Local brand kit loading
# ---------------------------------------------------------------------------
def load_brand_kit(brand_id: str) -> BrandKit:
    """Load brand kit from local directories instead of Canva API."""
    # Base path for the brand assets
    base_path = pathlib.Path(f"data/{brand_id}")
    
    # Define primary colors - these could be loaded from a config file
    primary_colors = ["#1D3C34", "#F2F2F2", "#D4C19C", "#F24C3D"]
    
    # Load fonts from the fonts directory
    fonts_dir = base_path / "fonts"
    fonts = {}
    
    # Process all font directories
    if fonts_dir.exists():
        # First, find all available font files (both TTF and OTF) across all font directories
        all_fonts = list(fonts_dir.glob("**/*.ttf")) + list(fonts_dir.glob("**/*.otf"))
        heading_fonts = [f for f in all_fonts if "bold" in f.name.lower()]
        body_fonts = [f for f in all_fonts if "bold" not in f.name.lower()]
        
        # Map font styles to specific font families if possible
        font_families = {}
        for font_dir in fonts_dir.iterdir():
            if font_dir.is_dir():
                family_name = font_dir.name.lower()
                # Include both TTF and OTF fonts
                family_fonts = list(font_dir.glob("*.ttf")) + list(font_dir.glob("*.otf"))
                if family_fonts:
                    font_families[family_name] = family_fonts
        
        # Assign heading font - prefer 'abordage' bold or any bold font
        if "abordage" in font_families and any("bold" in f.name.lower() for f in font_families["abordage"]):
            fonts["heading"] = str(next(f for f in font_families["abordage"] if "bold" in f.name.lower()))
        elif heading_fonts:
            fonts["heading"] = str(heading_fonts[0])
        elif all_fonts:
            fonts["heading"] = str(all_fonts[0])
        
        # Assign body font - prefer 'cantata one', then 'compagnon', then any non-bold font
        if "cantata one" in font_families:
            fonts["body"] = str(font_families["cantata one"][0])
        elif "compagnon" in font_families:
            # For compagnon, prefer regular style if available
            compagnon_regular = next((f for f in font_families["compagnon"] if "regular" in f.name.lower()), None)
            if compagnon_regular:
                fonts["body"] = str(compagnon_regular)
            else:
                fonts["body"] = str(font_families["compagnon"][0])
        elif body_fonts:
            fonts["body"] = str(body_fonts[0])
        elif all_fonts and all_fonts != [fonts.get("heading")]:
            fonts["body"] = str(all_fonts[0])
            
        # Add a third font style for captions or other elements if needed
        if "compagnon" in font_families and "body" in fonts and not fonts["body"].endswith(font_families["compagnon"][0].name):
            fonts["caption"] = str(font_families["compagnon"][0])
    
    # Load logos from the logos directory
    logos_dir = base_path / "logos"
    logos = {}
    
    if logos_dir.exists():
        # Find primary logo (preferring PNG format)
        primary_logos = list(logos_dir.glob("*nominal*.png"))
        if primary_logos:
            logos["primary"] = str(primary_logos[0])
        else:
            # Fallback to any PNG logo
            all_logos = list(logos_dir.glob("*.png"))
            if all_logos:
                logos["primary"] = str(all_logos[0])
    
    # Extra assets could be loaded from a separate directory if needed
    extra_assets = {}
    
    return BrandKit(
        primary_colors=primary_colors,
        fonts=fonts,
        logos=logos,
        extra_assets=extra_assets,
    )

# Asset catalogue & selection
# ---------------------------------------------------------------------------

def load_asset_catalogue(assets_dir: str | None = None, manifest: str | None = None) -> List[Asset]:
    """Load assets from a directory (filename‑based tags) or a JSON manifest."""
    catalogue: List[Asset] = []
    if manifest:
        data = json.loads(pathlib.Path(manifest).read_text("utf‑8"))
        for item in data:
            catalogue.append(Asset(name=item["name"], description=item["description"], use_cases=item["use_cases"], path_or_url=item["path"]))
    elif assets_dir:
        for fp in pathlib.Path(assets_dir).glob("*.png"):
            tags = re.split(r"[_\-]", fp.stem)  # crude tokenisation on filename
            catalogue.append(Asset(name=fp.stem, tags=tags, path_or_url=str(fp)))
    else:
        raise ValueError("Provide either assets_dir or manifest")
    return catalogue

# ------------- Asset scoring helpers ------------- #

def pick_assets_for_slide(copy_text: str, assets: Sequence[Asset], top_k: int = 3, model: str = "gpt-4.1-2025-04-14") -> List[Asset]:
    """Return the best‑fitting decorative assets for this copy block."""
    if _OPENAI:
        # Ask GPT‑4o to choose asset names (fast, accurate)
        sys_prompt = "Você é um designer de mídias sociais. Escolha os 3 elementos visuais que melhor reforçam a mensagem do slide, considerando as opções disponíveis (nomes dos arquivos). Responda apenas com uma lista em JSON: [\"asset1\", \"asset2\", ...]."
        asset_names = [a.name for a in assets]
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": f"COPIA DO SLIDE:\n{copy_text}\n\nOPÇÕES DE ASSET:\n{str(data)}"},
            ],
            "temperature": 0.2,
        }
        r = _OPENAI.chat.completions.create(**payload)  # type: ignore[attr-defined]
        chosen = json.loads(r.choices[0].message.content)
        return [a for a in assets if a.path_or_url in chosen][:top_k]

# ---------------------------------------------------------------------------
# LLM content plan (minor localisation tweaks)
# ---------------------------------------------------------------------------
OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"

def create_content_plan(main_prompt: str, style_md: str, copies: List[str], model: str) -> List[SlidePlan]:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    system_prompt = (
        "Você é um diretor de arte senior. Gere um plano JSON para um carrossel do Instagram, definindo para cada slide os campos heading (título/subtítulo curto) e body (texto corrido). Use exatamente, sem alterar, o copy fornecido. Segue guia de estilo:\n" + style_md
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": " ".join(main_prompt)},
        {"role": "user", "content": json.dumps(copies, ensure_ascii=False)},
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,
        response_format={"type": "json_object"}
    )
    content = response.choices[0].message.content
    raw_plan = json.loads(content)
    plan: List[SlidePlan] = []
    for idx, p in enumerate(raw_plan, start=1):
        plan.append(SlidePlan(slide=idx, heading=p["heading"], body=p["body"]))
    return plan

# ---------------------------------------------------------------------------
# Background generation (unchanged)
# ---------------------------------------------------------------------------

def generate_background_images(num: int, brand: BrandKit) -> List[Image.Image]:
    try:
        from openai import OpenAI  # type: ignore
        client = OpenAI()
        imgs: List[Image.Image] = []
        for _ in tqdm(range(num)):
            try:
                prompt = f"Abstract minimal background in brand palette {', '.join(brand.primary_colors[:2])}, soft shapes, flat design, 4K"
                res = client.images.generate(prompt=prompt, n=1, size="1024x1024")
                img_bytes = requests.get(res.data[0].url, timeout=30).content
                imgs.append(Image.open(io.BytesIO(img_bytes)).convert("RGBA"))
            except Exception:
                imgs.append(Image.new("RGBA", (1080, 1080), brand.primary_colors[0]))
        return imgs
    except Exception:
        # Fallback to solid color backgrounds if OpenAI is not available
        return [Image.new("RGBA", (1080, 1080), color) for color in brand.primary_colors[:num]]

# ---------------------------------------------------------------------------
# Layout & composition
# ---------------------------------------------------------------------------

_ANCHORS = [(60, 60), (800, 50), (50, 700), (700, 700)]  # where to drop decorative assets


def compose_slide(plan: SlidePlan,
                    bg: Image.Image,
                    brand: BrandKit,
                    catalogue: Sequence[Asset],
                    model: str = 'gpt-4.1-2025-04-14') -> Image.Image:
    """Compose slide with copy and chosen decorative assets."""
    slide = bg.resize((1080, 1080)).copy()
    draw = ImageDraw.Draw(slide)

    # Heading
    heading_font_path = brand.fonts.get("heading")
    if not heading_font_path:
        # Fallback to default font
        heading_font = ImageFont.load_default()
    else:
        heading_font = ImageFont.truetype(heading_font_path, size=92)
    draw.text((80, 180), plan.heading, font=heading_font, fill=brand.primary_colors[1])

    # Body text (auto‑wrap to 35 chars)
    body_font_path = brand.fonts.get("body")
    if not body_font_path:
        # Fallback to default font
        body_font = ImageFont.load_default()
    else:
        body_font = ImageFont.truetype(body_font_path, size=48)
    wrapped = textwrap.fill(plan.body, width=35)
    draw.text((80, 380), wrapped, font=body_font, fill="#3d3d3d")

    # Decorative assets
    chosen_assets = pick_assets_for_slide(plan.heading + " " + plan.body, catalogue, model)
    plan.suggested_assets = [a.name for a in chosen_assets]
    for anchor, asset in zip(_ANCHORS, chosen_assets):
        try:
            asset_img = _open_asset(asset.path_or_url, max_size=350)
            slide.alpha_composite(asset_img, dest=anchor)
        except Exception:
            continue

    # Logo bottom‑right
    logo_path = brand.logos.get("primary")
    if logo_path:
        try:
            logo = Image.open(logo_path).convert("RGBA").resize((120, 120))
            slide.alpha_composite(logo, dest=(900, 900))
        except Exception:
            pass  # Skip logo if it can't be loaded

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
    # Load brand kit from local files instead of Canva API
    brand = load_brand_kit(brand_id)

    # Input prompt & copies
    prompt_data = json.loads(pathlib.Path(prompt_json).read_text("utf‑8"))
    copies = prompt_data["copies"]
    main_prompt = prompt_data.get("prompt", "")

    style_md = pathlib.Path(style_file).read_text("utf‑8")

    plan = create_content_plan(main_prompt, style_md, copies, model)
    bgs = generate_background_images(len(plan), brand)

    # If assets_dir is not provided, use the default location
    if not assets_dir and not assets_manifest:
        assets_dir = f"data/{brand_id}/shapes/png"
    
    catalogue = load_asset_catalogue(assets_dir, assets_manifest)

    slides = [compose_slide(p, bg, brand, catalogue, model) for p, bg in zip(plan[:1], bgs[:1])]
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
    group = p.add_mutually_exclusive_group(required=False)  # Changed to not required
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


# Example usage with argparse.Namespace:
"""
args = argparse.Namespace(
    prompt_json="data/aqi/prompts/prompts.json",
    style_guideline="data/aqi/examples/brand_guidelines.md",
    brand_id="aqi",
    assets_dir="data/aqi/shapes/png",
    assets_manifest="data/aqi/shapes/assets_manifest_prod.json",
    output_dir="data/aqi/output"
)
"""

prompt_json="data/aqi/prompts/prompts.json"
style_file="data/aqi/examples/brand_guidelines.md"
brand_id="aqi"
assets_dir="data/aqi/shapes/png"
assets_manifest="data/aqi/shapes/assets_manifest_prod.json"
output_dir=out_dir = "data/aqi/output"
model = 'gpt-4.1-2025-04-14'


# Example command line usage:
"""
Example:
python agent.py \
    --prompt_json data/aqi/prompts/prompts.json \
    --style_guideline data/aqi/style_guidelines/style_guidelines.md \
    --brand_id aqi \
    --assets_dir data/aqi/shapes/png \
    --output_dir data/aqi/output
"""

# Alternative command line usage with assets manifest:
"""
Example:
python agent.py \
    --prompt_json data/aqi/prompts/prompts.json \
    --style_guideline data/aqi/style_guidelines/style_guidelines.md \
    --brand_id aqi \
    --assets_manifest data/aqi/shapes/assets_manifest_prod.json \
    --output_dir data/aqi/output
"""