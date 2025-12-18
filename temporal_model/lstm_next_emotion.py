from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional
from collections import Counter

import torch
import torch.nn as nn


def load_timeline(sync_path: Path) -> List[Dict[str, Any]]:
    with open(sync_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    tl = data.get("synchronized_timeline") or data.get("timeline")
    if not isinstance(tl, list):
        raise KeyError(f"JSON sin timeline valido. Claves: {list(data.keys())}")
    return tl


def extract_text_emotions(timeline: List[Dict[str, Any]]) -> List[str]:
    seq = []
    for item in timeline:
        e = item.get("text_emotion")
        if e in (None, "", "None"):
            continue
        seq.append(str(e))
    return seq


class NextEmotionLSTM(nn.Module):
    def __init__(self, n_classes: int, emb_dim: int = 16, hidden: int = 32):
        super().__init__()
        self.emb = nn.Embedding(n_classes, emb_dim)
        self.lstm = nn.LSTM(input_size=emb_dim, hidden_size=hidden, batch_first=True)
        self.fc = nn.Linear(hidden, n_classes)

    def forward(self, x):
        x = self.emb(x)
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.fc(last)


def build_dataset(ids: List[int], window: int) -> (torch.Tensor, torch.Tensor):
    X, y = [], []
    for i in range(len(ids) - window):
        X.append(ids[i:i+window])
        y.append(ids[i+window])
    return torch.tensor(X, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def train_lstm(sync_json: str, out_json: str, window: int = 10, epochs: int = 5) -> None:
    sync_path = Path(sync_json)
    timeline = load_timeline(sync_path)
    emotions = extract_text_emotions(timeline)

    min_required = window + 2
    if len(emotions) < min_required:
        # Reducir ventana automaticamente
        window = max(3, len(emotions) // 2)
        if window < 2:
            raise ValueError(
                f"Muy pocos datos para entrenar LSTM. Emociones disponibles: {len(emotions)}"
            )
    vocab = sorted(set(emotions))
    to_id = {e: i for i, e in enumerate(vocab)}
    to_em = {i: e for e, i in to_id.items()}

    ids = [to_id[e] for e in emotions]
    X, y = build_dataset(ids, window)

    model = NextEmotionLSTM(n_classes=len(vocab))
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=0.01)

    model.train()
    for ep in range(1, epochs + 1):
        opt.zero_grad()
        logits = model(X)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()

    # Evaluacion simple en el mismo set (demo academica)
    model.eval()
    with torch.no_grad():
        preds = model(X).argmax(dim=1)
        acc = (preds == y).float().mean().item()

    # Guardar reporte de predicciones
    out_path = Path(out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pred_labels = [to_em[int(p)] for p in preds[:50]]
    true_labels = [to_em[int(t)] for t in y[:50]]

    report = {
        "source_sync_json": str(sync_path),
        "model": {
            "type": "LSTM",
            "task": "next_text_emotion_prediction",
            "window": window,
            "epochs": epochs
        },
        "metrics": {
            "train_accuracy_demo": round(acc, 4)
        },
        "sample_predictions_first_50": [
            {"true": tl, "pred": pl} for tl, pl in zip(true_labels, pred_labels)
        ],
        "vocab": vocab
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="LSTM demo: predecir la siguiente emocion del texto")
    parser.add_argument("--sync_json", required=True, help="JSON sincronizado")
    parser.add_argument("--out", required=True, help="Salida del reporte del modelo (JSON)")
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    train_lstm(args.sync_json, args.out, window=args.window, epochs=args.epochs)


if __name__ == "__main__":
    main()
