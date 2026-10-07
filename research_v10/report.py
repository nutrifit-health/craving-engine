"""Пересоздаёт двуязычный отчёт и графики из сохранённых результатов V10."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    run = root / "results/v10/run-20261007"
    summary = json.loads((run / "summary.json").read_text())
    checks = json.loads(
        (root / "results/v10/recheck-20261007-retry1/verification.json").read_text()
    )
    few = json.loads(
        (root / "results/v10/recheck-20261007-retry1/few-shot-paired.json").read_text()
    )
    cf = json.loads((root / "results/v10/verification/counterfactual.json").read_text())
    cost = json.loads((run / "development_costs.json").read_text())
    diagnostic = json.loads(
        (root / "results/v10/verification/representation-diagnostic.json").read_text()
    )
    contrasts = json.loads((run / "development_comparisons.json").read_text())
    scenarios = {
        name: json.loads((run / f"scenarios/{name}/report.json").read_text())
        for name in summary["scenarios"]
    }
    names = {
        "population": "Population GBDT",
        "scalar": "Scalar",
        "direct": "KC direct",
        "lif_direct": "LIF + direct",
        "lif_stdp": "LIF + STDP",
        "lif_rstdp": "LIF + error-modulated STDP",
        "lif_direct_matched": "LIF + direct (matched)",
        "lif_stdp_matched": "LIF + STDP (matched)",
    }
    families = ["population", "scalar", "direct", "lif_direct", "lif_stdp", "lif_rstdp"]

    def ci(value):
        return (
            f"{value['mean']:+.6f} [{value['ci95'][0]:+.6f}; {value['ci95'][1]:+.6f}]"
        )

    def table(headers, rows):
        return "\n".join(
            [
                "| " + " | ".join(headers) + " |",
                "| " + " | ".join(["---"] * len(headers)) + " |",
            ]
            + ["| " + " | ".join(map(str, row)) + " |" for row in rows]
        )

    metrics = table(
        ["Model", "Brier ↓", "AUC ↑", "Recall", "FPR", "J", "Caught / false, 2 per 7d"],
        [
            [
                names[f],
                f"{summary['development'][f]['gated']['macro']['brier']:.6f}",
                f"{summary['development'][f]['gated']['macro']['auc']:.6f}",
                f"{100 * summary['development'][f]['gated']['macro']['recall']:.2f}%",
                f"{100 * summary['development'][f]['gated']['macro']['false_positive_rate']:.2f}%",
                f"{summary['development'][f]['gated']['macro']['youden_j']:.6f}",
                f"{summary['development'][f]['gated']['notifications']['caught_events']} / {summary['development'][f]['gated']['notifications']['false_notifications']}",
            ]
            for f in families
        ],
    )
    scenario_table = table(
        ["Scenario", "ΔBrier vs KC direct, 95% CI", "ΔBrier vs matched LIF direct"],
        [
            [
                s,
                ci(v["comparisons"]["lif_rstdp_vs_direct_gated"]["brier_delta"]),
                ci(
                    v["comparisons"]["lif_rstdp_vs_lif_direct_matched_gated"][
                        "brier_delta"
                    ]
                ),
            ]
            for s, v in scenarios.items()
        ],
    )
    cf_table = table(
        [
            "Model",
            "After 1 response: ΔBrier, 95% CI",
            "After 2 responses: ΔBrier, 95% CI",
        ],
        [
            [
                names[f],
                ci(cf["families"][f]["learning_vs_withheld"]["1"]["gated_brier_delta"]),
                ci(cf["families"][f]["learning_vs_withheld"]["2"]["gated_brier_delta"]),
            ]
            for f in ["scalar", "direct", "lif_direct_matched", "lif_rstdp"]
        ],
    )
    cost_table = table(
        [
            "Model",
            "Numeric bytes / user",
            "JSON checkpoint bytes, min–max",
            "Replay µs / decision",
            "Encoding µs / decision",
        ],
        [
            [
                names[f],
                cost[f]["states"][0]["numeric_bytes"],
                f"{min(s['checkpoint_json_bytes'] for s in cost[f]['states']):,}–{max(s['checkpoint_json_bytes'] for s in cost[f]['states']):,}",
                f"{cost[f]['microseconds_per_decision']:.1f}",
                f"{cost[f].get('encoding_seconds', 0) / 7200 * 1e6:.1f}",
            ]
            for f in families
            if cost[f]["states"]
        ],
    )

    plots = root / "results/v10/plots"
    plots.mkdir(exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 140,
        }
    )
    fig, ax = plt.subplots(figsize=(9, 4.2), layout="constrained")
    values = [summary["development"][f]["gated"]["macro"]["brier"] for f in families]
    ax.barh(
        [names[f] for f in families],
        values,
        color=["#8b98a5", "#22866a", "#2b6fa6", "#b18b50", "#986d8c", "#bb6354"],
    )
    ax.invert_yaxis()
    ax.set_xlim(0, 0.19)
    ax.set_xlabel("Macro Brier score (lower is better)")
    ax.set_title("V10: frozen V9 development, 40 synthetic users")
    for i, v in enumerate(values):
        ax.text(v + 0.00015, i, f"{v:.6f}", va="center", fontsize=9)
    fig.savefig(plots / "development-brier.png")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4.4), layout="constrained")
    for i, (s, v) in enumerate(scenarios.items()):
        d = v["comparisons"]["lif_rstdp_vs_direct_gated"]["brier_delta"]
        m = d["mean"]
        lo, hi = d["ci95"]
        ax.errorbar(
            m, i, xerr=[[m - lo], [hi - m]], fmt="o", capsize=4, color="#2b6fa6"
        )
    ax.axvline(0, color="#888", lw=1)
    ax.set_yticks(range(len(scenarios)), list(scenarios))
    ax.invert_yaxis()
    ax.set_xlabel(
        "Paired ΔBrier: LIF + error-STDP minus KC direct (95% user bootstrap CI)"
    )
    ax.set_title("New controlled scenarios: 3 seeds × 40 users; negative favours STDP")
    fig.savefig(plots / "scenario-deltas.png")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    x = [0, 1, 2, 5, 10, 20]
    for f in ["scalar", "direct", "lif_direct_matched", "lif_rstdp"]:
        for ax, policy in zip(axes, ["gated", "ungated"]):
            y = [
                scenarios["abrupt"]["adaptation"][f][str(n)][
                    f"probability_mse_{policy}"
                ]["mean"]
                for n in x
            ]
            ax.plot(x, y, marker="o", label=names[f], markersize=3)
    for ax, title in zip(axes, ["Original safety gate", "Ungated diagnostic"]):
        ax.set_title(title)
        ax.set_xlabel("Received responses after change")
        ax.set_xticks(x)
        ax.set_ylabel("Error against generator probability (MSE)")
        ax.grid(alpha=0.15)
    axes[1].legend(fontsize=8)
    fig.suptitle("Abrupt shift: 120 synthetic users with 60 days of prior experience")
    fig.savefig(plots / "adaptation.png")
    plt.close(fig)

    reproduce = """```bash
    python3.12 -m venv .venv
    source .venv/bin/activate
    python -m pip install -r requirements-v10.txt
    python -m pip install --no-deps -e .
    python -m pytest research_v8 research_v9 research_v10/test_mechanisms.py -q
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \\
      python -m research_v10.experiment --output-dir runs/v10-new
    python -m research_v10.verify --run runs/v10-new --output-dir runs/v10-new-check
    python -m research_v10.followup --run runs/v10-new --output runs/v10-new-counterfactual.json
    ```
    """
    links = """- [Frozen configurations and thresholds](../results/v10/run-20261007/frozen.json)
    - [All tuning attempts](../results/v10/run-20261007/tuning.json)
    - [Main summary](../results/v10/run-20261007/summary.json)
    - [Paired development comparisons](../results/v10/run-20261007/development_comparisons.json)
    - [Independent verification and admission](../results/v10/recheck-20261007-retry1/verification.json)
    - [Paired few-shot comparisons](../results/v10/recheck-20261007-retry1/few-shot-paired.json)
    - [Counterfactual / cold-start control](../results/v10/verification/counterfactual.json)
    - [Representation diagnosis](../results/v10/verification/representation-diagnostic.json)
    - [Author audit and provenance](../results/v10-author-audit/summary.json)
    - [Complete artifact hashes](../results/v10/run-20261007/artifact_manifest.json)
    """
    ru = f"""# Research V10: помогли ли LIF и локальная пластичность?
    
    [English](results-v10.md) · [README](../README.ru.md) · [Протокол](protocol-v10.ru.md)
    
    Дата: 7 октября 2026 года. **В проверенной реализации преимущество LIF + STDP не подтверждено.** Простая скалярная поправка дала меньшую ошибку вероятностей на прежней задаче. На новых задачах отдельные небольшие эффекты не выполнили заранее заданные условия. Это результат конкретной конечной серии, а не доказательство бесполезности спайковых сетей.
    
    Полезный положительный результат: в контролируемой задаче первые изменения качества после одного-двух новых ответов обнаружились и у обычной памяти. Спайки для этого не оказались необходимыми. Освоение нового режима за два ответа и польза на реальных людях не установлены.
    
    ## Что выполнено
    
    - Точный листинг [ToxaBes](https://habr.com/ru/articles/1045696/) проверен на CPU и MPS, seeds 7/17/27. Полное обучение MNIST (40 эпох) и заявленная автором accuracy не воспроизводились: проверены исполняемые механизмы.
    - Самостоятельный V9 повторён: пять основных macro-метрик всех шести моделей совпали с архивом точно. Прежний отрицательный допуск сохранился.
    - 35 заранее заданных конфигураций рассмотрены на 10 tuning-пользователях; пороги выбраны на 10 других calibration-пользователях. Зафиксированы 8 вариантов, включая 2 matched-контроля. Затем оценены 40 известных V9 development-пользователей.
    - Восемь новых контролируемых сценариев, 3 seeds, 40 пользователей × 120 решений на seed: 115200 сценарных решений на вариант. Одни и те же 120 синтетических пользователей переиспользованы между сценариями; это не 960 независимых людей.
    - Основная серия заняла {summary["seconds"]:.2f}с. Независимый повтор всех 8 × 7200 development-прогнозов совпал точно; проверены {checks["manifest_files_verified"]} артефакта. Проверки: 38 тестов V8/V9 и 31 подслучай, 14 тестов механизмов V10, 7 тестов листинга автора; Ruff, сборка sdist/wheel, импорт из wheel и pip check прошли.
    
    ## Что перенесено из рекомендации
    
    Внутренний LIF повторяет принцип накопления тока, утечки, порога и сброса. Перебраны 5 профилей времени/усиления и 2 скорости обучения. Сравнены обновление error×rate, парная STDP и парная STDP×prediction error. Последний вариант — наша инженерная гипотеза о третьем факторе, не готовое правило из статьи.
    
    Общая вероятность GBDT и входная KC-кодировка одинаковы. Личная память, задержки, ограничения поправки и гейт одинаковы; matched-контроли используют те же LIF-параметры и learning rate. Самостоятельная MB-модель V9 воспроизведена отдельно и не подменяется GBDT в историческом отчёте.
    
    Реализация V10 отличается от авторской сети: фиксированные KC-входы, 32 детерминированных внутренних такта на событие, персональный readout и сохранённая eligibility для позднего ответа. У автора — кортикальные модули, Poisson-кодирование и межмодульная пластичность. У нас пост-импульсы задаются общим p0; рекуррентная спайковая память между ежедневными событиями и обучение encoder через surrogate gradient не реализованы. Глобальная модель не переобучается спайковым методом.
    
    ## Проверка листинга автора
    
    | Проверка | Наблюдение | Следствие |
    | --- | --- | --- |
    | Gradient / AdamW | `input_scale` и веса колонок не получили gradient; readout и interweights обновляются | Листинг совместим с фиксированным reservoir, но не с обучением всех названных параметров обычным backprop |
    | Дополнительный rate loss | `requires_grad=False` | Эта часть loss не даёт регуляризирующего градиента |
    | Повтор eval | Logits менялись примерно на 0.0015–0.0028; восстановление EMA устранило расхождение | Оценка зависит от изменяемого `spike_rate_avg`; argmax в проверенных batches не изменился |
    | Короткая STDP-серия | Веса изменились примерно на 3–5×10⁻⁸ за 3 batches; logits в этих float32 проходах не изменились | Нельзя приписывать показанную accuracy именно STDP; длительное обучение здесь не проверялось |
    | Связность | Изменение interweights меняет выходные признаки, но не входы/спайки/mem/syn колонок | В листинге нет обратной подачи этой модуляции в LIF |
    | Создание связей | Максимальная корреляция 0.63501 при стандартном threshold 0.8 | Стандартный механизм формирования новых связей не достигает порога; проверено 1000 максимальными обновлениями |
    
    Точные численные проверки, SHA256 листинга и ограничения сохранены в [аудите](../results/v10-author-audit/summary.json). Чужой полный листинг не включён в распространяемый код.
    
    ## Прежняя задача: вероятности и предупреждения
    
    {metrics}
    
    В этой таблице первичная метрика — Brier (меньше лучше). Личный scalar уменьшил Brier, но снизил AUC и J относительно population. Поэтому даже лучший Brier не означает, что задача полезных предупреждений решена. V10 выбирает пороги на 10 пользователях, V9 выбирал на 20: recall/FPR нельзя напрямую сопоставлять между версиями как эффект модели.
    
    LIF+rSTDP против обычной KC-памяти: ΔBrier **{ci(contrasts["lif_rstdp_vs_direct_gated"]["brier_delta"])}**. Против scalar: **{ci(contrasts["lif_rstdp_vs_scalar_gated"]["brier_delta"])}**. Против LIF с теми же параметрами, но direct-обновлением: **{ci(contrasts["lif_rstdp_vs_lif_direct_matched_gated"]["brier_delta"])}**. Здесь положительная разница означает ухудшение.
    
    ![Ошибка вероятностей на V9 development](../results/v10/plots/development-brier.png)
    
    ## Новые контролируемые сценарии
    
    {scenario_table}
    
    Это разности gated Brier кандидата и соответствующего baseline, с парными 95% интервалами по людям. Шум/задержка/пропуски накладываются на abrupt, поэтому их контроль — abrupt. При delivery 72h используется порядок V9 «прогноз раньше ответа с тем же timestamp»; первый использующий ответ ежедневный прогноз происходит через 96 часов.
    
    В noise15 есть небольшой выигрыш относительно KC direct, но он меньше заданного 0.002, является вторичным сравнением и не проходит как общее подтверждение. В report50 ошибка выше. При report10 гейт не разрешил личную коррекцию и gated-прогнозы совпали. Наблюдаемые эффекты не обосновывают выбор исключительно удачного сценария.
    
    ![Парные разности в сценариях](../results/v10/plots/scenario-deltas.png)
    
    ## Один-два новых ответа: что можно утверждать
    
    К моменту резкой смены режима у моделей уже есть 60 дней опыта. Первое сравнение с population не изолирует новые ответы. Поэтому после основной оценки выполнен диагностический контроль без подбора: те же модели и события, но у контрольной копии отключены только ответы после смены режима. Предшествующие прогнозы совпали точно.
    
    {cf_table}
    
    Отрицательная разница означает пользу продолжения обучения относительно собственной копии без новых ответов. В этой синтетической задаче эффект после 1–2 ответов есть у нескольких методов. Он **не специфичен для спайков**. Прямое сравнение rSTDP с KC direct после двух ответов: ΔBrier **{ci(few["abrupt"]["direct"]["2"]["gated_brier_delta"])}** — преимущество не установлено.
    
    Для совершенно новой памяти gated-прогноз после первых 1–2 ответов совпадает с population: действует минимум 12 эффективных наблюдений и другие условия допуска. Ungated-результат — отдельная диагностика, выбранные настройки не подбирались специально для неё. Улучшение относительно замороженной памяти не означает полного освоения нового режима; порог «достаточного качества» такой адаптации не был задан.
    
    ![Кривые ошибки относительно вероятности генератора](../results/v10/plots/adaptation.png)
    
    ## Почему результат мог оказаться таким
    
    После оценки, без изменения весов или повторного подбора, измерена геометрия выбранного следа. Средний cosine eligibility/rate равен {diagnostic["mean_eligibility_rate_cosine"]:.3f}; в {100 * diagnostic["negative_alignment_fraction"]:.1f}% событий направления противоположны. Парная STDP не обязана совпадать с направлением уменьшения ошибки логистического readout. Умножение произвольного знакового следа на prediction error само по себе такого совпадения не гарантирует. Это конкретная зацепка для проектирования третьего фактора, а не объяснение всех SNN.
    
    Выбранная LIF-кодировка активирует в среднем {diagnostic["active_units_mean"]:.1f} элементов против 101 исходного KC-кода. Следовательно, она также меняет геометрию признаков; выигрыш от спайков не следует из одного уменьшения активности.
    
    Все варианты ограничены поправкой ±0.7 логита. Диагностика с известной вероятностью генератора показывает недостижимую при таком ограничении часть адаптации. Этот evaluator-only предел сохранён в verification.json; модель скрытую вероятность не получает. Нельзя автоматически ослабить ограничение после просмотра результата и считать это прежним экспериментом.
    
    ## Стоимость и размер состояния
    
    {cost_table}
    
    Измерено на локальном macOS arm64, один CPU-поток для NumPy, 180 дней V9. Replay включает Python-обвязку и финальную сериализацию состояния, но не обучение и вычисление общего GBDT p0. Encoding — отдельная добавочная цена LIF; базовое KC-преобразование в этой колонке не учитывается. Это исследовательское измерение, не SLA и не нагрузочный тест миллиона пользователей.
    
    Для KC/LIF три числовых массива по 2020 float64 по-прежнему занимают 48480 байт (47.34 KiB). Полный JSON больше и зависит от истории/pending/seen. V10 дополнительно сохраняет eligibility ожидающего ответа; память процесса, gzip, квантизация и production-хранилище не измерены. Цель 16 KiB полного состояния не достигнута.
    
    ## Воспроизводимость, ошибки и границы
    
    Первый независимый checker упал из-за требования точного равенства `sigmoid(logit(p0)) == p0`. Измеренный round-trip максимум 1.11×10⁻¹⁶, gated-прогноз совпал точно. Исправлен только checker: допуск 1e−14 для ungated round-trip, сохранены [первый журнал](../results/v10/verification/recheck.log) и [успешный повтор](../results/v10/verification/recheck-retry1.log). Модели, выбранные настройки и основной архив не менялись. Исправленный checker имеет отдельный SHA256; снимок main сохраняет его прежний вариант.
    
    Данные синтетические. V9 development уже известен. Новые seeds проверяют перенос внутри специально заданного генератора; параметры перенесены с 2020 на 33 координаты без нового подбора. Полные метки calibration идеализированы. Несколько вторичных сравнений не имеют поправки на множественность. Вывод ограничен выбранным диапазоном LIF и конкретным readout; 40 эпох MNIST, нейроморфное железо, клиническая польза и предотвращение событий не проверялись.
    
    ## Вывод и следующий исследовательский вопрос
    
    `recommendation_supported=false`: условия не выполнены относительно matched LIF direct, KC direct и scalar. Внедрять этот SNN-вариант как улучшение текущего предиктора оснований нет. Рекомендация дала полезный эксперимент и выявила проблему выбора сигнала обучения. Следующее отдельное исследование может заранее сравнить иное, обоснованное правило eligibility/модулятора и сохранение временной динамики между событиями с обычной памятью. Для такого шага нужен новый протокол и новая оценочная когорта, а не дальнейший подбор на результатах V10.
    
    ## Повторить эксперимент
    
    {reproduce}
    
    Для аудита автора сначала сохраните HTML статьи во временный файл, прочитайте извлечённый листинг и используйте `python -m research_v10.author_audit --help`. Аудитор принимает только проверенный SHA256 и требует `--execute-reviewed-snapshot`; тесты листинга читают путь из `V10_AUTHOR_SOURCE` и пропускаются, если временного файла нет. V10 runtime не требует Torch: он нужен для отдельного аудита автора.
    
    Для новых запусков нужны новые каталоги. [Точные зависимости](../requirements-v10.txt) относятся к проверенной платформе; переносимость всего диапазона версий не заявляется.
    
    ## Артефакты
    
    {links}
    """
    en = f"""# Research V10: did LIF and local plasticity help?
    
    [Русский](results-v10.ru.md) · [README](../README.md) · [Protocol](protocol-v10.md)
    
    Date: October 7, 2026. **The tested LIF + STDP implementation did not establish an advantage.** A scalar personal correction had lower probability error on the original task. Small effects in individual new scenarios did not meet the preregistered conditions. This finite experiment does not establish universal uselessness of spiking networks.
    
    A useful positive result: ordinary personal memory also showed initial improvements after one or two new responses in the controlled task. Spikes were not necessary for that effect. Complete adaptation after two responses and real-user effectiveness were not established.
    
    ## Completed work
    
    - Exact [ToxaBes listing](https://habr.com/ru/articles/1045696/) audited on CPU/MPS with seeds 7/17/27. We tested executable mechanisms, not the full 40-epoch MNIST training or reported accuracy.
    - Standalone V9 reproduced: five principal macro metrics for all six models matched the archive exactly; admission remained negative.
    - 35 fixed configurations screened on 10 tuning users, thresholds selected on 10 separate calibration users, 8 frozen variants including 2 matched controls evaluated on 40 previously examined V9 development users.
    - Eight new controlled scenarios × 3 seeds × 40 users × 120 decisions: 115200 scenario decisions per variant. The same 120 synthetic users recur across scenarios; they are not 960 independent people.
    - Main series: {summary["seconds"]:.2f}s. Independent replay reproduced all 8 × 7200 development rows exactly; {checks["manifest_files_verified"]} artifacts verified. Checks passed: 38 V8/V9 tests +31 subtests, 14 V10 mechanism tests, 7 author-listing tests, Ruff, sdist/wheel build, wheel imports and pip check.
    
    ## What was transferred
    
    LIF implements current integration, leakage, threshold and reset. Five time/gain profiles and two learning rates were tested. Learning rules: error×rate, pair-based STDP, and STDP×prediction error. The latter is our engineering third-factor hypothesis, not a ready-made rule copied from the article.
    
    Principal ablations share GBDT p0 and the original KC input representation. Personal memory, delays, correction bounds and gating are shared; matched controls use exactly the same LIF profile and rate. Standalone MB V9 was reproduced separately; its historical results are not redefined as a GBDT hybrid.
    
    V10 differs from the author's network: fixed KC inputs, 32 deterministic ticks per event, a personal readout and stored eligibility for delayed responses. The article uses cortical modules, Poisson encoding and inter-module plasticity. Our post spikes derive from population p0. We do not implement recurrent spiking state between daily events or encoder training with surrogate gradients. The global model is not retrained as an SNN.
    
    ## Exact-listing audit
    
    | Check | Observation | Implication |
    | --- | --- | --- |
    | Gradients / AdamW | No gradient for input_scale or column weights; readout/interweights update | Compatible with a fixed reservoir, not end-to-end training of every named parameter |
    | Additional rate loss | requires_grad=False | It contributes no regularizing gradient |
    | Repeated evaluation | Logits changed about 0.0015–0.0028; restoring EMA removed the change | spike_rate_avg mutates during evaluation; tested batches had no argmax changes |
    | Short STDP run | Weight changes about 3–5×10⁻⁸ across 3 batches; no float32 logit difference | Does not attribute reported accuracy to STDP or rule out a long-run effect |
    | Connectivity intervention | Interweights affect output features, not column inputs/spikes/mem/syn | No feedback of this modulation into LIF dynamics |
    | Connection formation | Maximum correlation 0.63501, default threshold 0.8 | Default formation threshold is unreachable; 1000 maximal updates checked |
    
    Exact measurements, listing SHA256 and limitations are in the [audit](../results/v10-author-audit/summary.json). The full third-party listing is not redistributed.
    
    ## Original task: probabilities and warnings
    
    {metrics}
    
    Brier is primary here; lower is better. Scalar memory reduces Brier but lowers AUC/J relative to population. Better probability error does not establish better warnings. V10 selects thresholds on 10 users versus 20 in V9, so cross-version recall/FPR differences cannot be attributed solely to models.
    
    LIF+rSTDP versus KC direct: ΔBrier **{ci(contrasts["lif_rstdp_vs_direct_gated"]["brier_delta"])}**; versus scalar: **{ci(contrasts["lif_rstdp_vs_scalar_gated"]["brier_delta"])}**; versus matched LIF direct: **{ci(contrasts["lif_rstdp_vs_lif_direct_matched_gated"]["brier_delta"])}**. Positive differences indicate deterioration.
    
    ![Development probability error](../results/v10/plots/development-brier.png)
    
    ## New controlled scenarios
    
    {scenario_table}
    
    Differences use gated Brier with paired 95% user-bootstrap intervals. Noise/delay/reporting interventions modify abrupt, their reference scenario. Under V9 timestamp ordering, a 72-hour response first influences the 96-hour daily prediction.
    
    The noise15 improvement against KC direct is small, below the 0.002 threshold and exploratory. Report50 is worse. In report10 the gate does not allow a personal correction and gated forecasts coincide. Selecting only the favourable scenario would not support an overall claim.
    
    ![Scenario paired differences](../results/v10/plots/scenario-deltas.png)
    
    ## One or two new responses
    
    Models already have 60 days of experience at the abrupt change. Comparing only with population does not isolate new responses. A post-evaluation diagnostic, without retuning, therefore compares each model with its own copy receiving no post-change responses. Pre-change predictions match exactly.
    
    {cf_table}
    
    Negative differences favour continued learning over its own control. Several methods benefit after 1–2 responses in this synthetic task. The effect is **not specific to spikes**. Direct rSTDP-versus-KC comparison after two responses: ΔBrier **{ci(few["abrupt"]["direct"]["2"]["gated_brier_delta"])}**; no advantage established.
    
    For entirely new personal memory, gated forecasts after 1–2 responses equal population: at least 12 effective observations and other admission conditions are required. Ungated scores are diagnostics; their configurations were selected by gated Brier. Better performance than a frozen memory does not establish complete adaptation; no adequate-quality threshold for such adaptation was prespecified.
    
    ![Adaptation against generator probability](../results/v10/plots/adaptation.png)
    
    ## Mechanistic diagnosis
    
    Post-evaluation geometry, without retuning: mean eligibility/rate cosine={diagnostic["mean_eligibility_rate_cosine"]:.3f}; directions oppose in {100 * diagnostic["negative_alignment_fraction"]:.1f}% of events. Pair-based STDP need not align with the logistic readout's error-reducing direction. Multiplying an arbitrary signed trace by prediction error does not guarantee alignment. This motivates a better-founded third factor; it is not an explanation of every SNN.
    
    The selected LIF code activates {diagnostic["active_units_mean"]:.1f} units on average versus 101 original KC units. LIF also changes feature geometry; fewer spikes alone do not demonstrate an advantage.
    
    All variants cap corrections at ±0.7 logit. Evaluator-only oracle diagnostics quantify error unreachable under this bound. Models never receive the hidden probability. Relaxing the bound after viewing results would constitute a new experiment.
    
    ## Cost and state
    
    {cost_table}
    
    Measured locally on macOS arm64, one NumPy CPU thread, 180 V9 days. Replay includes Python orchestration and final state serialization but excludes fitting and computing shared GBDT p0. Encoding is additional LIF work; the common KC transformation is excluded from that column. This is not an SLA or a million-user load test.
    
    KC/LIF retain three 2020-value float64 arrays: 48480 bytes (47.34 KiB). Complete JSON is larger and depends on history/pending/seen. V10 also stores pending eligibility. Process heap, compressed checkpoints, quantization and production storage were not measured. Complete 16 KiB state was not achieved.
    
    ## Reproducibility and limitations
    
    The first independent checker required exact `sigmoid(logit(p0)) == p0`. Measured maximum round-trip error was 1.11×10⁻¹⁶; gated forecasts matched exactly. Only the checker changed to 1e−14 tolerance for the ungated round-trip. [Original failure](../results/v10/verification/recheck.log) and [successful retry](../results/v10/verification/recheck-retry1.log) are preserved. Models, selected settings and the main archive were unchanged. The corrected checker has a separate SHA256; the main snapshot retains its earlier version.
    
    All data are synthetic. V9 development is previously examined. New seeds test a specified generator; settings transfer from 2020 to 33 coordinates without retuning. Complete calibration labels are idealized. Secondary comparisons have no multiplicity correction. Conclusions cover the selected LIF range and readout only. Full MNIST training, neuromorphic hardware, clinical benefit and prevention were not tested.
    
    ## Decision
    
    `recommendation_supported=false`: conditions failed against matched LIF direct, KC direct and scalar. This SNN variant is not justified as an improvement to the predictor. The recommendation led to a useful experiment and exposed learning-signal design issues. A separate future study could prespecify a better-founded eligibility/modulator and temporal state across events against ordinary memory. That requires a new protocol and evaluation cohort, not further tuning on V10 results.
    
    ## Reproduction
    
    {reproduce}
    
    For the author audit, save article HTML temporarily, read the extracted listing and inspect `python -m research_v10.author_audit --help`. Execution is restricted to the reviewed SHA256 and requires `--execute-reviewed-snapshot`. Listing tests read `V10_AUTHOR_SOURCE` and skip if the temporary file is missing. V10 runtime needs no Torch; Torch is used for the separate author audit.
    
    Use fresh output directories. [Exact dependencies](../requirements-v10.txt) describe the tested platform, not universal compatibility.
    
    ## Artifacts
    
    {links}
    """
    for english, russian in {
        "| Model |": "| Модель |",
        "Caught / false, 2 per 7d": "Поймано / ложных, лимит 2 за 7 дней",
        "| Scenario |": "| Сценарий |",
        "After 1 response: ΔBrier, 95% CI": "После 1 ответа: ΔBrier, 95% ДИ",
        "After 2 responses: ΔBrier, 95% CI": "После 2 ответов: ΔBrier, 95% ДИ",
        "Numeric bytes / user": "Числовые байты на пользователя",
        "JSON checkpoint bytes, min–max": "Байты JSON, минимум–максимум",
        "Replay µs / decision": "Replay, мкс на решение",
        "Encoding µs / decision": "Кодирование, мкс на решение",
        "Frozen configurations and thresholds": "Зафиксированные настройки и пороги",
        "All tuning attempts": "Все попытки подбора",
        "Main summary": "Основной результат",
        "Paired development comparisons": "Парные сравнения на development",
        "Independent verification and admission": "Независимая проверка и решение",
        "Paired few-shot comparisons": "Парные сравнения после первых ответов",
        "Counterfactual / cold-start control": "Контроль новых ответов и холодного старта",
        "Representation diagnosis": "Диагностика представления",
        "Author audit and provenance": "Аудит автора и происхождение листинга",
        "Complete artifact hashes": "Контрольные суммы всех артефактов",
    }.items():
        ru = ru.replace(english, russian)
    for filename, body in [("results-v10.ru.md", ru), ("results-v10.md", en)]:
        # Убираем отступ шаблона, сохраняя строки подставленных таблиц.
        rendered = "\n".join(
            line.removeprefix("    ")
            for line in body.splitlines()
        )
        (root / "docs" / filename).write_text(rendered.strip() + "\n")
    print("Bilingual reports and 3 figures generated")


if __name__ == "__main__":
    main()
