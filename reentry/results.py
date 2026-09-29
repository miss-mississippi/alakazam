"""Машиночитаемые результаты.

Каждый скрипт шага пишет ключевые числа в results/<имя>.json. Документы
(README.md, REPORT_full.md) ссылаются на них невидимыми метками вида

    8.5<!--=step5.scenarios.film.total:.1f-->

и check_docs.py сверяет число перед меткой с результатом. Так цифры в тексте
не расходятся с кодом молча.

Ключ — путь через точку: первый сегмент — имя файла в results/, дальше —
ключи словаря; числовой сегмент — индекс списка.
"""

from __future__ import annotations

import datetime as _dt
import json
import math
from pathlib import Path

import numpy as np

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def _clean(x):
    """numpy -> python, NaN -> None, кортежи -> списки."""
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [_clean(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return None if math.isnan(v) else v
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


class Recorder:
    """Собирает числа по ходу скрипта и сохраняет их одним файлом.

        R = Recorder("step5")
        R["scenarios.film.total"] = 8.52
        R.save()
    """

    def __init__(self, name: str):
        self.name = name
        self.data: dict = {}

    def __setitem__(self, key: str, value):
        node = self.data
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = value

    def save(self, script: str | None = None) -> Path:
        RESULTS_DIR.mkdir(exist_ok=True)
        out = {"_meta": {"script": Path(script).name if script else None,
                         "generated_utc": _dt.datetime.now(_dt.timezone.utc)
                         .strftime("%Y-%m-%dT%H:%M:%SZ")}}
        out.update(_clean(self.data))
        path = RESULTS_DIR / f"{self.name}.json"
        path.write_text(json.dumps(out, ensure_ascii=False, indent=1,
                                   sort_keys=True) + "\n")
        return path


def load(name: str) -> dict:
    return json.loads((RESULTS_DIR / f"{name}.json").read_text())


def lookup(key: str, cache: dict | None = None):
    """Значение по ключу 'файл.путь.к.значению'. KeyError, если его нет."""
    parts = key.split(".")
    if cache is None:
        cache = {}
    if parts[0] not in cache:
        path = RESULTS_DIR / f"{parts[0]}.json"
        if not path.exists():
            raise KeyError(f"нет файла results/{parts[0]}.json")
        cache[parts[0]] = json.loads(path.read_text())
    node = cache[parts[0]]
    for p in parts[1:]:
        if isinstance(node, list):
            node = node[int(p)]
        else:
            node = node[p]
    return node
