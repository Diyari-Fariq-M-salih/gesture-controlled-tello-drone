import csv
import time
from typing import Dict, Iterable, List

class TelemetryLogger:
    def __init__(self, fields: Iterable[str], path: str):
        self.fields = list(fields)
        self.path = path
        self.rows: List[Dict[str, object]] = []

    def add(self, state: Dict[str, float]):
        row = {"t": time.time()}
        for k in self.fields:
            row[k] = state.get(k, None)
        self.rows.append(row)

    def export(self):
        with open(self.path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["t"] + self.fields)
            w.writeheader()
            w.writerows(self.rows)
