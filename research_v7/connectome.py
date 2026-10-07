"""Извлекает реальную ALPN→KC подматрицу MaleCNS; не моделирует весь мозг.

Числа синапсов служат неотрицательными структурными весами. Они не определяют
нейромедиатор, физиологический знак, проводимость или динамику пластичности.
Отображение человеческих признаков на ALPN — отдельная инженерная конструкция.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


SOURCE_ROOT = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/"
SOURCE_FILES = {
    "body-annotations-male-cns-v1.0-minconf-0.5.feather": "2177e246113e4cfbf1e7772ec37c6da1955ff22e8063d0b1f833101f99a9a3b2",
    "connectome-weights-male-cns-v1.0-minconf-0.5.feather": "e35da783d1c686b2b58b3b87cd6a403ae43bfcfba8bff28e08ef752c1a56afc1",
}
MAPPING_SEEDS = (11, 23, 47)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract(source_dir: Path, output: Path) -> dict:
    """Читает большие Feather порциями; сохраняет только выбранную подматрицу."""
    if output.exists():
        raise ValueError("Артефакт уже существует; перезапись запрещена")
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.feather as feather
    import pyarrow.ipc as ipc

    source_metadata = {}
    for name, expected in SOURCE_FILES.items():
        path = source_dir / name
        digest = file_sha256(path)
        if digest != expected:
            raise ValueError(f"Не совпадает SHA256 MaleCNS v1.0: {name}")
        source_metadata[name] = {"url": SOURCE_ROOT + name, "sha256": digest, "bytes": path.stat().st_size}
    annotations = feather.read_table(source_dir / next(iter(SOURCE_FILES))).to_pylist()
    kc = sorted((row for row in annotations if row["class"] == "Kenyon_Cell" and row["somaSide"] == "L"), key=lambda row: row["bodyId"])
    pn = sorted((row for row in annotations if row["class"] == "ALPN"), key=lambda row: row["bodyId"])
    if not kc or not pn:
        raise ValueError("Нет аннотированных KC или ALPN")
    kc_ids = np.asarray([row["bodyId"] for row in kc], dtype=np.int64)
    pn_ids = np.asarray([row["bodyId"] for row in pn], dtype=np.int64)
    kc_set, pn_set = pa.array(kc_ids), pa.array(pn_ids)
    pre, post, weights = [], [], []
    weights_path = source_dir / "connectome-weights-male-cns-v1.0-minconf-0.5.feather"
    with pa.memory_map(str(weights_path), "r") as source:
        reader = ipc.open_file(source)
        for index in range(reader.num_record_batches):
            batch = reader.get_batch(index)
            keep = pc.and_(pc.is_in(batch["body_pre"], value_set=pn_set), pc.is_in(batch["body_post"], value_set=kc_set))
            chosen = batch.filter(keep)
            if chosen.num_rows:
                pre.extend(chosen["body_pre"].to_pylist())
                post.extend(chosen["body_post"].to_pylist())
                weights.extend(chosen["weight"].to_pylist())
    if not weights or min(weights) <= 0:
        raise ValueError("Нет положительных анатомических связей")
    # ALPN без входа в выбранное левое MB не участвуют в искусственном адаптере.
    connected_pn = set(pre)
    pn = [row for row in pn if row["bodyId"] in connected_pn]
    pn_ids = np.asarray([row["bodyId"] for row in pn], dtype=np.int64)
    rows = np.searchsorted(kc_ids, post)
    columns = np.searchsorted(pn_ids, pre)
    matrix = np.zeros((len(kc), len(pn)), dtype=np.int64)
    np.add.at(matrix, (rows, columns), weights)
    rows, columns = np.nonzero(matrix)
    metadata = {
        "dataset": "male-cns:v1.0", "min_synapse_confidence": 0.5,
        "source_page": "https://male-cns.janelia.org/download/", "license": "CC-BY-4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "attribution": "MaleCNS v1.0, FlyEM/HHMI Janelia, University of Cambridge, MRC Laboratory of Molecular Biology, Google Research; Berg et al., Cell 2026",
        "modification_notice": "Из полной таблицы выделена ALPN→KC подматрица левого MB; оригинальные body IDs и числа синапсов сохранены. Нормировка, искусственный вход и перемешанный контроль создаются отдельно в replay.",
        "citation": "https://doi.org/10.1016/j.cell.2026.08.015",
        "selection": {"KC": "class=Kenyon_Cell AND somaSide=L", "PN": "class=ALPN, any somaSide, with edge to selected KC", "edge": "body_pre=PN AND body_post=KC, weight>0; no outcome-dependent selection"},
        "kc_count": len(kc), "pn_count": len(pn), "edges": len(rows),
        "synapse_count": int(matrix.sum()), "kc_without_alpn_input": int(np.sum(matrix.sum(axis=1) == 0)),
        "sources": source_metadata,
        "limitations": [
            "Участок ALPN→KC левого MB; прочие входы, APL, DAN, MBON и рекуррентная динамика не воспроизводятся.",
            "Положительные структурные веса — число синапсов, не подтверждённые возбуждающие проводимости.",
            "В аннотациях ALPN могут иметь неполную степень ручной проверки.",
            "Mapping человеческих ON/OFF каналов на ALPN искусственный, не доказанная сенсорная гомология.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        np.savez_compressed(stream, kc_ids=kc_ids, pn_ids=pn_ids,
                            kc_types=np.asarray([row["type"] or "" for row in kc]),
                            pn_types=np.asarray([row["type"] or "" for row in pn]),
                            rows=rows, columns=columns, weights=matrix[rows, columns],
                            metadata=np.asarray(json.dumps(metadata, ensure_ascii=False)))
    return metadata


def rewire(matrix: np.ndarray, seed: int = 20260923) -> tuple[np.ndarray, dict]:
    """Двойные обмены сохраняют оба бинарных degree и веса внутри каждой KC."""
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or not np.isfinite(matrix).all() or np.any(matrix < 0):
        raise ValueError("Ожидается конечная неотрицательная матрица")
    rows, columns = np.nonzero(matrix)
    weights = matrix[rows, columns].copy()
    original_columns = columns.copy()
    width = matrix.shape[1]
    occupied = set((rows * width + columns).tolist())
    rng = np.random.default_rng(seed)
    accepted, attempts = 0, 0
    target = 20 * len(rows)
    while len(rows) > 1 and accepted < target and attempts < 200 * len(rows):
        first, second = rng.integers(len(rows), size=2)
        attempts += 1
        r1, r2, c1, c2 = rows[first], rows[second], columns[first], columns[second]
        if r1 == r2 or c1 == c2 or r1 * width + c2 in occupied or r2 * width + c1 in occupied:
            continue
        occupied.remove(r1 * width + c1)
        occupied.remove(r2 * width + c2)
        occupied.add(r1 * width + c2)
        occupied.add(r2 * width + c1)
        columns[first], columns[second] = c2, c1
        accepted += 1
    rewired = np.zeros_like(matrix)
    rewired[rows, columns] = weights
    if not np.array_equal(np.sum(matrix > 0, axis=0), np.sum(rewired > 0, axis=0)) or not np.array_equal(np.sum(matrix > 0, axis=1), np.sum(rewired > 0, axis=1)):
        raise ValueError("Перемешивание нарушило число входов или выходов")
    return rewired, {"seed": seed, "accepted_swaps": accepted, "attempted_swaps": attempts, "edge_endpoint_changed_fraction": float(np.mean(columns != original_columns)) if len(rows) else 0, "preserved": ["KC_in_degree", "PN_out_degree", "per_KC_weight_multiset"], "not_preserved": ["PN_out_strength", "higher_order_motifs"]}


def load_projections(path: Path, active_channels: np.ndarray) -> tuple[dict, dict]:
    """Ровно те же adapters применяются к реальному и перемешанному графу."""
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
        matrix = np.zeros((len(data["kc_ids"]), len(data["pn_ids"])), dtype=float)
        matrix[data["rows"], data["columns"]] = data["weights"]
        pn_ids = data["pn_ids"].tolist()
    if not len(active_channels):
        raise ValueError("Training не содержит изменяющихся PN осей")
    randomized, null_metadata = rewire(matrix)
    row_norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = np.divide(matrix, row_norm, out=np.zeros_like(matrix), where=row_norm > 0)
    randomized = np.divide(randomized, row_norm, out=np.zeros_like(randomized), where=row_norm > 0)
    projections, mappings = {}, {}
    for seed in MAPPING_SEEDS:
        # Каждый человеческий канал представлен примерно одинаковым числом ALPN.
        assignment = np.resize(np.asarray(active_channels), matrix.shape[1]).copy()
        np.random.default_rng(seed).shuffle(assignment)
        adapter = np.zeros((matrix.shape[1], 32))
        adapter[np.arange(matrix.shape[1]), assignment] = 1
        projections[f"malecns_{seed}"] = matrix @ adapter
        projections[f"rewired_{seed}"] = randomized @ adapter
        mappings[str(seed)] = {"pn_ids": pn_ids, "human_on_off_channel": assignment.tolist()}
    return projections, {**metadata, "artifact_sha256": file_sha256(path), "rewiring": null_metadata, "mapping": mappings, "normalization": "L2 per KC incoming anatomical row before human-channel aggregation"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(extract(args.source_dir, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
