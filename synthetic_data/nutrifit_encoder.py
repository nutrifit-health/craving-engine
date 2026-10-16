"""Явный offline-вызов локального энкодера NutriFit, без HTTP и базы данных.

Нужен отдельный checkout NutriFit и уже установленный локальный tsx.
Без вызова encode_histories модуль ничего не запускает и не ищет backend.
"""

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from synthetic_data.contexts import CONTEXT_WINDOWS, history_contexts
from synthetic_data.types import TrueDailyState, UserHealthContext


def _source_hashes(backend_root: Path) -> dict[str, str]:
    repository_root = backend_root.parents[1]
    backend_paths = [
        "src/modules/Insights/services/neuromorphic/encoder.ts",
        "src/modules/Insights/services/neuromorphic/constants.ts",
        "src/modules/Insights/constants/index.ts",
        "src/modules/Insights/types/craving.ts",
        "scripts/craving-research/encoder.ts",
        "scripts/craving-research/types.ts",
    ]
    paths = [backend_root / name for name in backend_paths] + [
        repository_root / "packages/api-contract/src/metrics/stress.ts",
        repository_root / "packages/api-contract/src/insights/craving-schemas.ts",
    ]
    hashes = {
        str(path.relative_to(repository_root)): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in paths
    }
    for name in ("nutrifit_encoder.py", "contexts.py", "types.py"):
        path = Path(__file__).with_name(name)
        hashes["craving-engine/synthetic_data/" + name] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    return hashes


def encode_histories(
    histories: list[list[tuple[TrueDailyState, UserHealthContext]]],
    *,
    backend_root: Path,
) -> tuple[list[np.ndarray], dict[str, object]]:
    """Кодирует один пакет историй текущим TS encoder, без Python-замены формул."""
    backend_root = Path(backend_root).expanduser().resolve()
    repository_root = backend_root.parents[1]
    runner = backend_root / "node_modules/.bin/tsx"
    if not runner.is_file():
        raise RuntimeError(
            "Локальный apps/backend/node_modules/.bin/tsx не установлен; автоматическая загрузка запрещена"
        )
    sources = _source_hashes(backend_root)
    contexts: list[dict[str, object]] = []
    exclusions: list[dict[str, int]] = []
    identities = [history[0][1].userId for history in histories if history]
    if len(set(identities)) != len(identities):
        raise ValueError("Пакет содержит повторяющиеся идентификаторы историй")
    for history in histories:
        prepared, excluded = history_contexts(history)
        contexts.extend(prepared)
        exclusions.append(excluded)
    payload = json.dumps(
        {"contexts": contexts},
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )
    completed = subprocess.run(
        [str(runner), str(backend_root / "scripts/craving-research/encoder.ts")],
        cwd=backend_root,
        input=payload,
        text=True,
        capture_output=True,
        check=False,
        timeout=180,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Production encoder не выполнился: " + completed.stderr[-6000:]
        )
    decoded = json.loads(completed.stdout)
    if not isinstance(decoded, dict):
        raise TypeError("Production encoder вернул неизвестный формат")
    rows, runtime = decoded.get("rows"), decoded.get("runtime")
    if (
        not isinstance(rows, list)
        or len(rows) != len(contexts)
        or not isinstance(runtime, dict)
    ):
        raise ValueError("Нарушено соответствие метаданных строк входному пакету")
    if (
        runtime.get("contextWindows") != CONTEXT_WINDOWS
        or runtime.get("minimumPopulatedSections") != 2
    ):
        raise ValueError(
            "Окна или минимальная готовность backend изменились; требуется обновить явное отображение моста"
        )
    matrix = np.asarray(decoded.get("vectors"), dtype=np.float32)
    if not contexts and matrix.size == 0:
        matrix = matrix.reshape((0, 16))
    if (
        matrix.shape != (len(contexts), 16)
        or not np.isfinite(matrix).all()
        or np.any((matrix < 0) | (matrix > 1))
    ):
        raise ValueError(
            "Production encoder вернул неправильный размер или диапазон PN"
        )
    if _source_hashes(backend_root) != sources:
        raise RuntimeError("Исходники изменились во время пакетного кодирования")
    runtime_hashes = runtime.get("apiContractSha256")
    if not isinstance(runtime_hashes, dict) or not runtime_hashes:
        raise ValueError("Отсутствует provenance реально загруженного api-contract")
    for filename, digest in runtime_hashes.items():
        if not isinstance(filename, str) or not isinstance(digest, str):
            raise TypeError("Некорректный provenance api-contract")
        path = Path(filename).resolve()
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError("Runtime api-contract изменился во время кодирования")
        sources[str(path.relative_to(repository_root))] = digest

    vectors: list[np.ndarray] = []
    per_history: list[dict[str, object]] = []
    offset = 0
    total_ready = 0
    for history, excluded in zip(histories, exclusions):
        count = len(history)
        vectors.append(matrix[offset : offset + count].copy())
        section_counts = []
        ready_mask = []
        for index, row in enumerate(rows[offset : offset + count], start=offset):
            if (
                not isinstance(row, dict)
                or row.get("populatedSections") != contexts[index]["populatedSections"]
            ):
                raise ValueError("Число секций выходной строки не соответствует входу")
            sections = row["populatedSections"]
            ready = row.get("ready")
            if (
                not isinstance(sections, int)
                or not isinstance(ready, bool)
                or ready != (sections >= 2)
            ):
                raise ValueError("Некорректная готовность строки")
            section_counts.append(sections)
            ready_mask.append(ready)
        total_ready += sum(ready_mask)
        per_history.append(
            {
                "user_id": history[0][1].userId if history else None,
                "rows": count,
                "ready_rows": sum(ready_mask),
                "populated_sections": section_counts,
                "ready_mask": ready_mask,
                "excluded_record_appearances": excluded,
            }
        )
        offset += count
    metadata: dict[str, object] = {
        "adapter_version": "explicit-local-ts-observed-no-vitals-v2",
        "encoder": "apps/backend/src/modules/Insights/services/neuromorphic/encoder.ts#encodeCravingContext",
        "source_sha256": sources,
        "encoder_input_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "runtime": runtime,
        "context_windows": CONTEXT_WINDOWS,
        "readiness": {
            "minimum_populated_sections": 2,
            "rows": len(contexts),
            "ready_rows": total_ready,
            "ready_fraction": total_ready / len(contexts) if contexts else None,
            "per_history": per_history,
            "policy": "PN возвращены для всех строк; readiness является метаданными, production service здесь не запускался",
        },
        "input_sources": [row["inputSources"] for row in rows],
        "excluded_branches": [
            "HRV RMSSD/SDNN, resting HR, индивидуальные baseline и objective stress: vitals=null; computeStressScore не исполняется",
            "Паспортные возраст, рост, истинные витальные, latent sensitivities, GI/GL и сегодняшние исходы не передаются",
            "Оценка жира Deurenberg и BMR Mifflin без наблюдаемых параметров; доступны только BodyMeasurementRecord",
            "HealthTimeline alcoholUnits24h отсутствует в наблюдаемом контракте: 0; алкоголь из записей еды сохраняется",
            "База данных, tenant isolation, metrics snapshot projection, notification/service gating и производственный MB не исполняются",
            "История логирования до начала переданной траектории и fallback поиска HealthTimeline старше 30 дней отсутствуют",
        ],
        "mapping_limitations": [
            "Наивные даты трактуются как условный UTC; географическая таймзона и DST не моделируются",
            "Окно еды 48 ч, тренировок 24 ч согласно текущим константам backend; сон/чек-ин 36 ч и замер тела 30 дней",
            "Без durable ID повторные события объединяются по времени и типу еды/спорту; коллизии разных событий не различимы",
            "HealthHabitsRecord — накопительный снимок воды текущего дня, доступный при получении контекста; targetWaterMl и готовые счётчики привычек исключены",
            "Engagement рассчитывается по наблюдаемым еде, тренировкам, сну, чек-инам, замерам тела и снимкам воды за 30 дней",
            "Перенос сна, чек-ина и замера тела выполняется целыми записями, а не покомпонентной витриной latest metrics",
            "Шкала fatigueScore должна быть 0..10; неоднозначные значения большей шкалы отклоняются без автоматического угадывания",
            "Генерация корпуса и инженерное соответствие энкодеру не являются клинической проверкой",
        ],
    }
    return vectors, metadata
