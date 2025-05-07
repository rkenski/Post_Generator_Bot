# asset_descriptor.py — v0.3 (AQI‑tuned)
"""Automated shape‑descriptor bot for **AQI – Aquilo Que Importa**

Output format (one JSON object per PNG):
    {
      "name": "highlight_brush_01",
      "description": "Pincelada verde‑limão em forma de anel aberto, traço à mão que abraça o conteúdo com leveza e calor.",
      "use_cases": [
        "Enquadrar logotipo ou frase‑chave, criando sensação de abraço",
        "Marcar o início de uma história para guiar o olhar do leitor",
        "Trazer ritmo visual em transições suaves entre slides"
      ],
      "path": "verde/highlight_brush_01.png"
    }

--------------------------------------------------------------------------
CLI remains the same as v0.2.
"""
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
from asyncio import create_task, as_completed

from PIL import Image
from tqdm import tqdm

try:
    from openai import AsyncOpenAI  # OpenAI ≥ 1.15
except ImportError as e:  # pragma: no cover
    raise SystemExit("pip install --upgrade openai>=1.15") from e

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class Example:
    path: str
    description: str
    use_cases: List[str]

@dataclass
class Descriptor:
    name: str
    description: str
    use_cases: List[str]
    path: str

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_IMAGE_MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}


def _encode_data_url(img_path: str) -> str:
    ext = pathlib.Path(img_path).suffix.lower()
    mime = _IMAGE_MIME.get(ext, "image/png")
    with open(img_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{encoded}"

# --------------------------- System prompt --------------------------- #
_SYSTEM_PROMPT = (
    "Você é designer sênior do estúdio criativo AQI – Aquilo Que Importa. "
    "A AQI conecta marcas e pessoas à sua essência, valorizando hospitalidade, "
    "autenticidade, inovação e a solenidade aos detalhes. "
    "Ao receber uma forma PNG transparente da identidade visual da AQI, descreva‑a "
    "em Português de forma calorosa, poética e humana, evitando jargões corporativos. "
    "Depois proponha até três usos dessa forma em um carrossel de Instagram que "
    "realcem propósito, conexão e cuidado, nunca focando em vendas ou discurso "
    "de produto. Responda **estritamente** como JSON válido com as chaves: "
    "description (string) e use_cases (array de strings)."
)

# ---------------------------------------------------------------------------
# Message builder
# ---------------------------------------------------------------------------

def _build_messages(asset_path: str, examples: Sequence[Example]) -> List[Dict]:
    msgs: List[Dict] = [{"role": "system", "content": _SYSTEM_PROMPT}]

    for ex in examples:
        msgs.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Descreva esta forma e sugira usos:"},
                    {"type": "image_url", "image_url": {"url": _encode_data_url(ex.path)}},
                ],
            }
        )
        msgs.append(
            {
                "role": "assistant",
                "content": json.dumps({"description": ex.description, "use_cases": ex.use_cases}, ensure_ascii=False),
            }
        )

    # Query for current asset
    msgs.append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Descreva esta forma e sugira usos:"},
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
# Orchestration
# ---------------------------------------------------------------------------

async def main_async(args: argparse.Namespace) -> None:
    examples: List[Example] = []
    if args.examples_json:
        raw = json.loads(pathlib.Path(args.examples_json).read_text("utf‑8"))
        for item in raw:
            examples.append(Example(**item))
    if not examples:
        print("⚠️  Sem exemplos few‑shot – resultados podem ser mais genéricos.")

    asset_paths = [str(p) for p in pathlib.Path(args.assets_root).rglob("*.png")]
    if not asset_paths:
        sys.exit("Nenhum PNG encontrado em " + args.assets_root)

    client = AsyncOpenAI()
    sem = asyncio.Semaphore(args.concurrent_requests)
    results: List[Descriptor] = []

    async def worker(p):
        async with sem:
            desc = await _describe_asset(client, p, examples, args.model)
            results.append(desc)

    tasks = [create_task(worker(p)) for p in asset_paths]
    for fut in tqdm(as_completed(tasks), total=len(tasks), desc="Descrevendo", unit="img"):
        await fut

    with open(args.output, "w", encoding="utf‑8") as f:
        json.dump([d.__dict__ for d in results], f, ensure_ascii=False, indent=2)
    print(f"\n✅  {len(results)} descrições salvas em {args.output}")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Gera descrições humanizadas de formas PNG para o estúdio AQI.")
    p.add_argument("--assets_root", required=True)
    p.add_argument("--examples_json")
    p.add_argument("--output", default="assets_manifest_detailed.json")
    p.add_argument("--model", default=os.getenv("OPENAI_API_MODEL", "gpt‑4o‑mini"))
    p.add_argument("--concurrent_requests", type=int, default=5)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        sys.exit(1)

'''
args = argparse.Namespace(
    assets_root = "data/test/",
    examples_json = "data/aqi/examples/few_shot_examples.json",
    output = "data/aqi/assets_manifest_test.json",
    model = "gpt-4.1-2025-04-14",
    concurrent_requests = 5
)
'''
'''
Example:
python asset_descriptor.py \
    --assets_root data/aqi/png/ \
    --examples_json data/aqi/examples/few_shot_examples.json \
    --output data/aqi/assets_manifest_prod.json \
    --model gpt-4.1-2025-04-14
    --concurrent_requests 5
'''