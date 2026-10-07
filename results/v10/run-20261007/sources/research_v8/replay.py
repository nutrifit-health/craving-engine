"""Причинный replay V9; общие вычисления выделены из исследовательской V8."""
from __future__ import annotations
from collections import defaultdict, deque
from datetime import datetime, timedelta
import heapq
import json
from pathlib import Path
import numpy as np
from research_v8.archive import RunArchive, digest
from research_v8.data_types import V8Record
from research_v8.memory import EncodedMemory
ROOT = Path(__file__).resolve().parents[1]

def graph_matrix(path: Path) -> tuple[np.ndarray, dict]:
    """Сохраняет связь матрицы с body IDs и первичным артефактом MaleCNS."""
    with np.load(path, allow_pickle=False) as data:
        matrix = np.zeros((len(data["kc_ids"]), len(data["pn_ids"])), dtype=float)
        matrix[data["rows"], data["columns"]] = data["weights"]
        metadata = {"source": json.loads(str(data["metadata"])), "kc_ids": data["kc_ids"].tolist(), "pn_ids": data["pn_ids"].tolist(), "artifact_sha256": digest(path)}
    return matrix, metadata


def score_records(
    records: list[V8Record], codes: np.ndarray | None, probabilities: np.ndarray,
    model_name: str, representation_id: str, archive: RunArchive, split: str,
    *, global_head: np.ndarray | None = None,
) -> tuple[list[dict], dict]:
    """Хронологический replay: оценочная метка добавляется после prediction."""
    if len(probabilities) != len(records) or (codes is not None and len(codes) != len(records)):
        raise ValueError("Число представлений, прогнозов и записей расходится")
    if any(row.split != split for row in records):
        raise ValueError("Смешанные split в replay")
    groups = defaultdict(list)
    for index, row in enumerate(records):
        groups[row.user_id].append((index, row))
    scores, continuation = [], []
    geometry = {"pairs": 0, "cosine_sum": 0.0, "zero_codes": 0}
    for user_id, history in sorted(groups.items()):
        history.sort(key=lambda item: (item[1].decision_at, item[1].event_id))
        memory = None if codes is None else EncodedMemory(codes.shape[1], representation_id, global_head=global_head)
        pending = []
        previous = None
        for index, row in history:
            while pending and pending[0][0] < row.decision_at:
                feedback_at, _, feedback = heapq.heappop(pending)
                if memory is not None:
                    memory.observe(feedback.event_id, feedback_at, feedback.outcome, feedback.confidence)
            p_pop = float(probabilities[index])
            probability = candidate = learning = p_pop
            gamma, familiarity, residual = 0.0, 0.0, 0.0
            components = {key: 0.0 for key in ("global_residual", "personal_residual", "learning_residual", "candidate_residual")}
            if memory is not None:
                snapshot = memory.predict(row.event_id, row.decision_at, codes[index], p_pop)
                probability, candidate, learning = snapshot.p_final, snapshot.p_candidate, snapshot.p_learning
                gamma, familiarity, residual = snapshot.gamma, snapshot.familiarity, snapshot.residual
                components = {key: getattr(snapshot, key) for key in components}
                current = codes[index]
                geometry["zero_codes"] += int(not np.any(current))
                if previous is not None:
                    geometry["pairs"] += 1
                    geometry["cosine_sum"] += float(previous @ current)
                previous = current
            # Ни один из этих оценочных исходов не передаётся predict/encode.
            scores.append({
                "model": model_name, "split": split, "user_id": user_id, "event_id": row.event_id,
                "decision_at": row.decision_at.isoformat(), "available_at": row.available_at.isoformat(),
                "outcome": row.evaluation_outcome, "observed_outcome": row.outcome,
                "observed_feedback": row.outcome is not None,
                "feedback_at": row.feedback_at.isoformat() if row.feedback_at is not None else None,
                "probability": probability, "candidate": candidate, "p_learning": learning,
                "p_population": p_pop, "gamma": gamma, "familiarity": familiarity, "residual": residual,
                "representation_id": representation_id,
                **components,
            })
            if row.outcome is not None:
                heapq.heappush(pending, (row.feedback_at, row.event_id, row))
            elif memory is not None:
                memory.discard(row.event_id)
        if memory is not None:
            archive.write_checkpoint(f"checkpoints/{split}/{model_name}/{user_id}.json.gz", memory.export_state())
            continuation.extend({"user_id": user_id, "event_id": r.event_id, "feedback_at": at.isoformat(), "observed_outcome": r.outcome, "confidence": r.confidence} for at, _, r in sorted(pending))
    archive.write_json(f"checkpoints/{split}/{model_name}/pending_events.json", continuation)
    if geometry["pairs"]:
        geometry["mean_adjacent_cosine"] = geometry["cosine_sum"] / geometry["pairs"]
    return scores, geometry


def save_predictions(archive: RunArchive, name: str, split: str, rows: list[dict], thresholds: dict) -> None:
    """Каждая строка хранит решение политики до просмотра её исхода."""
    queues = defaultdict(deque)
    path = archive.path / "predictions" / split / f"{name}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        for row in sorted(rows, key=lambda item: (item["decision_at"], item["user_id"], item["event_id"])):
            at = datetime.fromisoformat(row["decision_at"])
            decisions = {}
            for point, threshold in thresholds.items():
                sent = {}
                for limit in (2, 3, 7):
                    queue = queues[(row["user_id"], point, limit)]
                    while queue and at - queue[0] >= timedelta(days=7):
                        queue.popleft()
                    notify = row["probability"] >= threshold and len(queue) < limit
                    if notify:
                        queue.append(at)
                    sent[str(limit)] = notify
                decisions[point] = {"threshold": threshold, "detected": row["probability"] >= threshold, "notified": sent}
            stream.write(json.dumps({**row, "decisions": decisions}, ensure_ascii=False, allow_nan=False) + "\n")

