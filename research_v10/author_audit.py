"""Воспроизводимая диагностика точного листинга ToxaBes без его публикации."""

import argparse
import copy
import hashlib
import importlib.util
import json
import platform
import random
import sys
import time
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path

import networkx
import torch
from torch.nn import functional

SOURCE_URL = "https://habr.com/ru/articles/1045696/"
REVIEWED_LISTING_SHA256 = (
    "00e0da6b8c25d4d6b58ad5fd73b3dd96f40486c4cf69b2a58718dac6e4bbce35"
)
SMALL_CONFIG = {
    "num_columns": 8,
    "in_dim": 40,
    "col_out": 20,
    "top_out": 3,
    "graph_k": 2,
    "graph_p": 0.15,
    "time_steps": 50,
}


class CodeBlocks(HTMLParser):
    """Извлекает содержимое pre без нормализации пробелов внутри листинга."""

    def __init__(self):
        super().__init__()
        self.blocks = []
        self.current = []
        self.in_pre = False

    def handle_starttag(self, tag, attrs):
        if tag == "pre":
            self.in_pre = True

    def handle_endtag(self, tag):
        if tag == "pre":
            self.blocks.append("".join(self.current))
            self.current = []
            self.in_pre = False

    def handle_data(self, data):
        if self.in_pre:
            self.current.append(data)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def extract_snapshot(html_path, source_path):
    """Сохраняет только временный модуль, возвращает публичную provenance-запись."""
    parser = CodeBlocks()
    html = html_path.read_bytes()
    parser.feed(html.decode("utf-8"))
    starts = (
        "#!/usr/bin/env python3",
        "class LIFNeuron",
        "class SpikeColumn",
        "def make_small_world",
        "class STDP",
        "class DynamicTopologyNetwork",
    )
    if len(parser.blocks) < 7 or not all(
        block.startswith(start) for block, start in zip(parser.blocks[:6], starts)
    ):
        raise ValueError(
            "Структура статьи изменилась: листинг необходимо прочитать заново"
        )
    source = "\n\n".join(parser.blocks[:6]) + "\n"
    source_path.write_text(source, encoding="utf-8")
    return {
        "url": SOURCE_URL,
        "html_sha256": sha256(html),
        "listing_sha256": sha256(source.encode()),
        "block_sha256": [sha256(b.encode()) for b in parser.blocks[:7]],
        "listing_blocks": list(range(6)),
        "training_block": 6,
        "extraction": "HTMLParser: pre text, blocks 0..5 joined by two LF; final LF",
        "listing_bytes": len(source.encode()),
        "execution": "Только прочитанный модуль модели; скрипт обучения не выполняется",
    }


def load_author(source_path):
    """Загружает прочитанный локальный snapshot без изменения кода автора."""
    if sha256(source_path.read_bytes()) != REVIEWED_LISTING_SHA256:
        raise ValueError(
            "SHA256 не совпадает с прочитанным snapshot: выполнение запрещено"
        )
    spec = importlib.util.spec_from_file_location("v10_toxabes_snapshot", source_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def max_delta(left, right):
    return float((left - right).abs().max().item())


def tensor_stats(value):
    value = value.detach().cpu()
    return {
        "min": float(value.min()),
        "max": float(value.max()),
        "l2": float(value.norm()),
        "nonzero": int(torch.count_nonzero(value)),
    }


def optimizer_for(model):
    """Повторяет группы, скорости и decay AdamW из скрипта статьи."""
    return torch.optim.AdamW(
        [
            {"params": model.readout.parameters(), "lr": 5e-4},
            {"params": model.input_scale, "lr": 2.5e-5},
            {"params": model.inter_weights, "lr": 5e-5},
        ],
        lr=5e-4,
        weight_decay=1e-4,
    )


def training_probe(model, x, y):
    """Разделяет изменения внутри forward, backward и optimizer.step."""
    model.train()
    optimizer = optimizer_for(model)
    optimized = {id(p) for group in optimizer.param_groups for p in group["params"]}
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    optimizer.zero_grad(set_to_none=True)
    logits, rates = model(x, learn=True)
    after_forward = {name: p.detach().clone() for name, p in model.named_parameters()}
    rate_loss = ((rates - model.target_rate) ** 2).mean()
    loss = functional.cross_entropy(logits, y) + 0.05 * rate_loss
    loss.backward()
    gradients = {
        name: None if p.grad is None else tensor_stats(p.grad)
        for name, p in model.named_parameters()
    }
    clipped = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
    optimizer.step()
    parameters = {
        name: {
            "requires_grad": p.requires_grad,
            "in_optimizer": id(p) in optimized,
            "gradient": gradients[name],
            "forward_delta": max_delta(before[name], after_forward[name]),
            "optimizer_delta": max_delta(after_forward[name], p.detach()),
        }
        for name, p in model.named_parameters()
    }
    return {
        "completed": True,
        "logits_shape": list(logits.shape),
        "finite": bool(torch.isfinite(logits).all()),
        "loss": float(loss.detach()),
        "pre_clip_gradient_norm": float(clipped),
        "rate_loss": float(rate_loss),
        "rate_loss_requires_grad": rate_loss.requires_grad,
        "current_rates_requires_grad": rates.requires_grad,
        "parameters": parameters,
    }


@torch.no_grad()
def eval_probe(base, x, other):
    """Сопоставляет повтор, порядок и разбиение batches при одинаковом checkpoint."""
    base.eval()
    repeat = copy.deepcopy(base)
    avg_before = repeat.spike_rate_avg.clone()
    first, rate_first = repeat(x)
    avg_after = repeat.spike_rate_avg.clone()
    second, rate_second = repeat(x)
    ab, ba = copy.deepcopy(base), copy.deepcopy(base)
    a_first, _ = ab(x)
    ab(other)
    ba(other)
    a_after_b, _ = ba(x)
    fixed = copy.deepcopy(base)
    checkpoint_avg = fixed.spike_rate_avg.clone()
    fixed_first, _ = fixed(x)
    fixed.spike_rate_avg.copy_(checkpoint_avg)
    fixed(other)
    fixed.spike_rate_avg.copy_(checkpoint_avg)
    fixed_again, _ = fixed(x)
    combined = copy.deepcopy(base)
    combined_logits, _ = combined(torch.cat((x, other)))
    return {
        "repeated_logits_max_delta": max_delta(first, second),
        "repeated_raw_rate_max_delta": max_delta(rate_first, rate_second),
        "eval_avg_max_delta": max_delta(avg_before, avg_after),
        "order_logits_max_delta": max_delta(a_first, a_after_b),
        "restored_avg_logits_max_delta": max_delta(fixed_first, fixed_again),
        "batch_partition_logits_max_delta": max_delta(
            a_first, combined_logits[: len(x)]
        ),
        "repeated_prediction_changes": int((first.argmax(1) != second.argmax(1)).sum()),
        "order_prediction_changes": int(
            (a_first.argmax(1) != a_after_b.argmax(1)).sum()
        ),
    }


@torch.no_grad()
def feedback_capture(model, x):
    """Наблюдает входы, спайки и состояния колонок через немодифицирующие hooks."""
    observations = []

    def capture(column, args, output):
        spikes, state = output
        observations.append(
            torch.cat(
                [
                    args[0].flatten(),
                    spikes.flatten(),
                    *[v.flatten() for v in state.values()],
                ]
            ).clone()
        )

    handles = [column.register_forward_hook(capture) for column in model.columns]
    model.eval()
    try:
        logits, rates = model(x)
    finally:
        for handle in handles:
            handle.remove()
    return logits, rates, torch.cat(observations)


@torch.no_grad()
def feedback_probe(base, x):
    """Сравнивает крайние межколоночные веса и удаление всех рёбер."""
    low, high, absent = (copy.deepcopy(base) for _ in range(3))
    low.inter_weights.fill_(-2)
    high.inter_weights.fill_(2)
    absent.topology.zero_()
    low_logits, low_rates, low_trace = feedback_capture(low, x)
    high_logits, high_rates, high_trace = feedback_capture(high, x)
    zero_logits, _, zero_trace = feedback_capture(absent, x)
    return {
        "column_input_spike_state_max_delta": max_delta(low_trace, high_trace),
        "no_edges_column_input_spike_state_max_delta": max_delta(low_trace, zero_trace),
        "logits_extreme_weight_max_delta": max_delta(low_logits, high_logits),
        "logits_no_edges_max_delta": max_delta(low_logits, zero_logits),
        "modulated_rates_max_delta": max_delta(low_rates, high_rates),
        "interpretation": "Связи модулируют признаки readout; обратной подачи в LIF нет",
    }


@torch.no_grad()
def stdp_probe(base, x, other, seed):
    """Парные проходы с одинаковыми RNG отделяют STDP от стохастики и AdamW."""
    on, off = copy.deepcopy(base), copy.deepcopy(base)
    on.train()
    off.train()
    start_weights = on.inter_weights.clone()
    pairs = []
    for step, batch in enumerate((x, other, x)):
        seed_all(seed + step)
        on_logits, _ = on(batch, learn=True)
        seed_all(seed + step)
        off_logits, _ = off(batch, learn=False)
        pairs.append(
            {
                "logits_max_delta": max_delta(on_logits, off_logits),
                "weights_max_delta": max_delta(on.inter_weights, off.inter_weights),
            }
        )
    carry, reset = copy.deepcopy(on), copy.deepcopy(on)
    trace_before = carry.pre_trace.clone()
    post_trace_before = carry.post_trace.clone()
    reset.pre_trace.zero_()
    reset.post_trace.zero_()
    before_second = carry.inter_weights.clone()
    seed_all(seed + 10)
    carry(other, learn=True)
    seed_all(seed + 10)
    reset(other, learn=True)
    return {
        "paired_batches": pairs,
        "total_stdp_weight_delta": max_delta(start_weights, on.inter_weights),
        "off_weight_delta": max_delta(start_weights, off.inter_weights),
        "trace_before_next_batch": tensor_stats(trace_before),
        "trace_after_next_batch": tensor_stats(carry.pre_trace),
        "post_trace_before_next_batch": tensor_stats(post_trace_before),
        "post_trace_after_next_batch": tensor_stats(carry.post_trace),
        "carried_vs_reset_trace_delta": max_delta(carry.pre_trace, reset.pre_trace),
        "carried_vs_reset_post_trace_delta": max_delta(
            carry.post_trace, reset.post_trace
        ),
        "carried_vs_reset_weight_delta": max_delta(
            carry.inter_weights, reset.inter_weights
        ),
        "next_batch_stdp_delta": max_delta(before_second, carry.inter_weights),
        "training_step": int(on.training_step),
        "topology_changed": bool(torch.any(on.topology != base.topology)),
    }


@torch.no_grad()
def range_probe(base, author):
    """Измеряет sigmoid-диапазон и достижимость порога формирования связей."""
    effective = torch.sigmoid(base.inter_weights) * base.topology
    edge = base.topology.bool()
    transformed = torch.sigmoid(torch.tensor([-2.0, 0.0, 2.0], device=effective.device))
    default_max = 102 / 128
    stdp = author.STDP(2).to(effective.device)
    strongest = torch.full((2,), default_max, device=effective.device)
    for _ in range(1000):
        stdp.update_connection_potential(strongest, strongest)
    return {
        "initial_effective_active_edges": tensor_stats(effective[edge]),
        "initial_negative_effective_edges": int((effective < 0).sum()),
        "sigmoid_raw_minus2_zero_plus2": transformed.cpu().tolist(),
        "small_signed_activity_bounds": [-0.2, 0.8],
        "default_signed_activity_bounds": [-26 / 128, default_max],
        "default_correlation_upper_bound": default_max**2,
        "formation_threshold": base.stdp.formation_threshold,
        "max_activity_potential_after_1000": float(stdp.connection_potential.max()),
        "formation_candidates_after_1000": int(
            (stdp.connection_potential > stdp.formation_threshold).sum()
        ),
        "interpretation": "Положительный effective вес; signed активность допускает отрицательное влияние. "
        "Порог формирования 0.8 недостижим при стандартной доле 102/128 "
        "и нулевом начальном potential: корреляция не выше (102/128)^2",
    }


def run_device(author, device, seeds):
    rows = []
    for seed in seeds:
        seed_all(seed)
        base = author.DynamicTopologyNetwork(**SMALL_CONFIG).to(device)
        x = torch.randn(4, SMALL_CONFIG["in_dim"], device=device) * 1.5 + 1
        other = torch.randn(3, SMALL_CONFIG["in_dim"], device=device) * 1.5 - 1
        y = torch.arange(4, device=device) % SMALL_CONFIG["top_out"]
        encoded = base._poisson_encode(x)
        rows.append(
            {
                "seed": seed,
                "config": SMALL_CONFIG,
                "encoding_requires_grad": encoded.requires_grad,
                "training": training_probe(copy.deepcopy(base), x, y),
                "evaluation": eval_probe(copy.deepcopy(base), x, other),
                "feedback": feedback_probe(base, x),
                "stdp": stdp_probe(base, x, other, seed),
                "effective_weights": range_probe(base, author),
            }
        )
    seed_all(seeds[0])
    full = author.DynamicTopologyNetwork().to(device)
    x = torch.randn(2, 784, device=device)
    y = torch.tensor([0, 1], device=device)
    smoke = training_probe(full, x, y)
    return {
        "device": str(device),
        "small_runs": rows,
        "article_dimensions_batch2": smoke,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--html", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--devices", nargs="+", default=["cpu", "mps"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 27])
    parser.add_argument("--execute-reviewed-snapshot", action="store_true")
    args = parser.parse_args()
    if not args.execute_reviewed_snapshot:
        parser.error("Для выполнения необходимо явно подтвердить чтение snapshot")
    if not args.source.resolve().is_relative_to(Path("/tmp").resolve()):
        parser.error("Полный чужой листинг разрешён только во временном /tmp")
    torch.set_num_threads(1)
    started = time.perf_counter()
    provenance = extract_snapshot(args.html, args.source)
    author = load_author(args.source)
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(UTC).isoformat(),
        "source": provenance,
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "networkx": networkx.__version__,
            "platform": platform.platform(),
            "executable": sys.executable,
            "cpu_threads": torch.get_num_threads(),
            "mps_available": torch.backends.mps.is_available(),
            "dependency_scope": "torch/networkx установлены только в локальный .venv; аудит не меняет pyproject",
        },
        "invocation": sys.argv,
        "results": [],
        "limitations": [
            "Синтетические диагностические batches; точность MNIST не измерялась",
            "40 эпох, benchmark NutriFit и основной протокол V10 не выполнялись",
            "Повторяемость CPU задаётся seed; MPS не обещает побитовую идентичность",
            "Жёсткие пороги блокируют gradient flow, но это совместимо с reservoir design",
            "Численные ablations отражают конкретные batches, не качество обученной модели",
        ],
    }
    for name in args.devices:
        if name == "mps" and not torch.backends.mps.is_available():
            report["results"].append({"device": name, "skipped": "MPS недоступен"})
            continue
        try:
            report["results"].append(run_device(author, torch.device(name), args.seeds))
        except Exception as error:
            report["results"].append({"device": name, "error": repr(error)})
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n"
            )
            raise
        print(f"Завершена диагностика {name}: seeds={args.seeds}", flush=True)
    report["elapsed_seconds"] = time.perf_counter() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"Отчёт: {args.output}; время {report['elapsed_seconds']:.2f} с", flush=True)


if __name__ == "__main__":
    main()
