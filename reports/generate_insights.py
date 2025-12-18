from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter, defaultdict
import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def _safe_load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _dominant_from_list(values: List[Optional[str]]) -> Optional[str]:
    cleaned = [v for v in values if v not in (None, "", "None")]
    if not cleaned:
        return None
    return Counter(cleaned).most_common(1)[0][0]


def _compute_transitions(values: List[Optional[str]]) -> List[Dict[str, Any]]:
    cleaned = [v if v not in (None, "", "None") else None for v in values]
    transitions = Counter()
    prev = None
    for v in cleaned:
        if v is None:
            continue
        if prev is not None and v != prev:
            transitions[(prev, v)] += 1
        prev = v
    out = []
    for (a, b), c in transitions.most_common(10):
        out.append({"from": a, "to": b, "count": c})
    return out


def build_insights(sync_json: Dict[str, Any]) -> Dict[str, Any]:
    timeline = sync_json.get("synchronized_timeline") or sync_json.get("timeline")
    if not isinstance(timeline, list) or len(timeline) == 0:
        raise ValueError("Timeline vacio o inexistente en el JSON sincronizado")

    times = []
    facial = []
    text = []

    for item in timeline:
        t = item.get("time_sec")
        fe = item.get("facial_emotion")
        te = item.get("text_emotion")

        if t is not None:
            try:
                times.append(float(t))
            except (TypeError, ValueError):
                pass

        facial.append(str(fe) if fe not in (None, "") else None)
        text.append(str(te) if te not in (None, "") else None)

    # Duracion segura
    duration = max(times) if times else 0.0

    def dominant(values):
        vals = [v for v in values if v not in (None, "", "None")]
        if not vals:
            return None
        return Counter(vals).most_common(1)[0][0]

    facial_dom = dominant(facial)
    text_dom = dominant(text)

    # Concordancia
    comparable = 0
    matches = 0
    for f, t in zip(facial, text):
        if f in (None, "", "None") or t in (None, "", "None"):
            continue
        comparable += 1
        if f == t:
            matches += 1

    agreement_rate = (matches / comparable) if comparable > 0 else None

    return {
        "duration_seconds": round(duration, 2),
        "dominant_facial_emotion": facial_dom,
        "dominant_text_emotion": text_dom,
        "agreement_rate": None if agreement_rate is None else round(agreement_rate, 4),
        "facial_emotion_distribution": dict(Counter(v for v in facial if v)),
        "text_emotion_distribution": dict(Counter(v for v in text if v)),
    }




def main():
    parser = argparse.ArgumentParser(description="Genera insights automáticos desde JSON sincronizado")
    parser.add_argument("--sync_json", required=True, help="Ruta al JSON sincronizado")
    parser.add_argument("--out", required=True, help="Ruta del JSON de salida (reporte)")
    args = parser.parse_args()

    sync_path = Path(args.sync_json)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data = _safe_load_json(sync_path)
    report = {
        "source_sync_json": str(sync_path),
        "insights": build_insights(data),
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logger.info(f"Reporte generado: {out_path}")


if __name__ == "__main__":
    main()
