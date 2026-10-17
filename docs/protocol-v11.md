[Русский](protocol-v11.ru.md) · [Documentation](README.md)

# Craving Engine V11 — Compact Contextual Personalization vs Scalar and Anatomical Control
Research protocol, 2026-10-16. Task: [V11 task](https://gov.nutrifit.health/tasks/6ad2b10a2858f3e61a4f22f5). Previous results: [V9](https://gov.nutrifit.health/documents/6ab46925560e2cbddbef9645), [V10](https://gov.nutrifit.health/tasks/6ac6b3a156965bd66f60880f).

## 1. Research Question and Scope of Inference
Does compact contextual personalization (a small weight vector $w_u \in \mathbb{R}^d$ for $d \in \{2, 4, 8\}$ beyond a scalar intercept $b_u$) provide a practical and statistically significant improvement in probability accuracy (Macro Brier) over a 1D scalar correction ($d=0$), and does the anatomical Kenyon cell graph (MaleCNS v1.0) provide additional benefit over a matched random control of the same dimensionality?

This is a controlled synthetic experiment across 180-day trajectories. It does not establish clinical efficacy, real-world relapse prevention, or performance on real users. Lack of significant improvement indicates the insufficiency of the evaluated representations and settings in this simulator, not proof that feedback lacks signal in general.

## 2. Representations and Evaluated Families
All personal variants operate on top of a single frozen population model $p_0(x)$ (GBDT), trained strictly on `training`. Five families are compared:

1. **`population`**: raw $p_0(x)$ prediction without personalization ($b_u=0, w_u=\emptyset$).
2. **`scalar`**: existing 1D scalar memory ($d=0, w_u=\emptyset$, intercept $b_u$ only). Primary personal baseline.
3. **`compact_linear`**: compact linear projection $X \to \text{PCA}_d$. Features are centered and scaled based on the training set.
4. **`compact_kc_anatomical`**: $X \to \text{MaleCNS}_{2019} \to \text{top-}k \to \text{PCA}_d$. MaleCNS v1.0 anatomical graph with $k$-WTA sparsity.
5. **`compact_kc_rewired`**: $X \to \text{Rewired}_{2019} \to \text{top-}k \to \text{PCA}_d$. Matched random control preserving degree distribution and weight multiset.

Input feature count is read dynamically from `schema.json`. All projectors and scaling statistics are fitted strictly on `training`. Signed PCA coordinates are used directly, without dual-channel splitting or per-instance normalization. The intercept $b_u$ is kept separate from the context vector norm.

## 3. Memory Model and Update Algorithm
The memory implements delayed direct update with residual clipping, fully inheriting the temporal semantics of `research_v8/memory.py`:

- **User State:** $b_{\text{fast}}, b_{\text{slow}} \in \mathbb{R}$ and $w_{\text{fast}}, w_{\text{slow}} \in \mathbb{R}^d$, scalar $\text{exposure} \in \mathbb{R}_{\ge 0}$, internal `clock`, queue `pending`, set `seen`, gate observation history.
- **Clock advancement:** before every prediction and on receiving feedback $\Delta t = (t - \text{clock})$. All components decay:
  $b_{\text{fast}}, w_{\text{fast}} \leftarrow 2^{-\Delta t / \tau_{\text{fast}}}$; $b_{\text{slow}}, w_{\text{slow}} \leftarrow 2^{-\Delta t / \tau_{\text{slow}}}$; $\text{exposure} \leftarrow 2^{-\Delta t / \tau_{\text{exp}}}$.
  Fixed parameters: $\tau_{\text{fast}} = 14$ days, $\tau_{\text{slow}} = 90$ days, $\tau_{\text{exp}} = 30$ days.
- **Prediction:** logistic residual $\operatorname{residual} = \operatorname{clip}(b_{\text{fast}} + b_{\text{slow}} + (w_{\text{fast}} + w_{\text{slow}})^\top z(x), -0.7, 0.7)$.
  Familiarity: $\text{fam} = 1 - \exp(-\text{exposure} / 3.0)$.
  Gate $\gamma \in [0, 1]$ computed over a 60-day window (min. 12 effective samples) of empirical gain over $p_0$.
  Final prediction: $p_{\text{final}} = \sigma(\operatorname{logit}(p_0) + \gamma \cdot \text{fam} \cdot \operatorname{residual})$.
  Snapshot saved in `pending[event_id]` with $p_{\text{learning}} = \sigma(\operatorname{logit}(p_0) + \operatorname{residual})$ and vector $z(x)$.
- **Feedback handling:** on receiving outcome $y \in \{0, 1\}$ with confidence $c \in (0, 1]$:
  Clock advances to feedback time $at$. Delay $\text{age} = (at - t_{\text{decision}})$.
  Error $e = (y - p_{\text{learning}}) \cdot c$.
  Consolidation $\kappa = \text{exposure} / (\text{exposure} + 3.0)$ (computed prior to adding current feedback).
  Delay dampening: $D_{\text{fast}} = 2^{-\text{age} / \tau_{\text{fast}}}$, $D_{\text{slow}} = 2^{-\text{age} / \tau_{\text{slow}}}$.
  Updates:
  $b_{\text{fast}} \leftarrow b_{\text{fast}} \cdot (1 - \min(1.0, \eta_{\text{fast}} \lambda c)) + \eta_{\text{fast}} D_{\text{fast}} e$;
  $b_{\text{slow}} \leftarrow b_{\text{slow}} \cdot (1 - \min(1.0, \eta_{\text{slow}} \lambda c)) + \eta_{\text{slow}} D_{\text{slow}} \kappa e$;
  $w_{\text{fast}} \leftarrow w_{\text{fast}} \cdot (1 - \min(1.0, \eta_{\text{fast}} \lambda c)) + \eta_{\text{fast}} D_{\text{fast}} e z(x)$;
  $w_{\text{slow}} \leftarrow w_{\text{slow}} \cdot (1 - \min(1.0, \eta_{\text{slow}} \lambda c)) + \eta_{\text{slow}} D_{\text{slow}} \kappa e z(x)$;
  $\text{exposure} \leftarrow \text{exposure} + c \cdot 2^{-\text{age} / \tau_{\text{exp}}}$.
  Fixed $\lambda = 0.01$.

At $d=0$, the module must be numerically identical to existing `scalar` (predictions and components tolerance $\le 10^{-12}$).

## 4. Data and Four Cohorts
The generator produces exactly 4 disjoint cohorts (180 days each):
- `training`: 60 users (seed 20261101) — fitting $p_0$ and projectors;
- `tuning`: 20 users (seed 20261102) — tuning dimensionality and learning rates;
- `calibration`: 20 users (seed 20261103) — diagnostic replay without tuning;
- `holdout`: 40 users (seed 20261104) — single final evaluation.

Full synthetic outcomes are accessible only to the evaluator. Training observed rows contain only causal features and observed labels. In replay, feedback arrives strictly sequentially at `feedback_at`.

## 5. Tuning Procedure on Tuning Cohort
Grid for compact families:
- $d \in \{2, 4, 8\}$;
- $\eta_{\text{fast}} \in \{0.05, 0.1, 0.2\}$, $\eta_{\text{slow}} = \eta_{\text{fast}} / 4$.
- Single pair $(d^*, \eta^*)$ selected by minimizing average Macro Brier across the 3 compact families. Ties within 0.0002 broken by smaller $d$, then smaller $\eta_{\text{fast}}$.

`scalar` is tuned on the same $\eta_{\text{fast}} \in \{0.05, 0.1, 0.2\}$ grid on its own Macro Brier on `tuning`.

All parameters are frozen prior to opening `holdout`.

## 6. Confirmatory Statistical Plan on Holdout
Analysis unit is user. For user $u$, difference in mean Brier: $\Delta_u = \text{Brier}_u(\text{A}) - \text{Brier}_u(\text{B})$. Mean across users: $\hat{\Delta} = \frac{1}{N} \sum_{u=1}^N \Delta_u$.

Exactly two confirmatory contrasts:
1. **Primary contrast:** $\text{compact\_linear} - \text{scalar}$ ($H_{0,1}: \mu_\Delta \ge 0$ vs $H_{1,1}: \mu_\Delta < 0$).
2. **Anatomical contrast:** $\text{compact\_kc\_anatomical} - \text{compact\_kc\_rewired}$ ($H_{0,2}: \mu_\Delta \ge 0$ vs $H_{1,2}: \mu_\Delta < 0$).

**Testing procedure:**
- 20,000 bootstrap resamples of users with fixed seed 20261105.
- 95% confidence interval: percentile bootstrap of differences.
- One-sided p-value: bootstrap of centered differences under $H_0$ ($\tilde{\Delta}_u = \Delta_u - \hat{\Delta}$):
  $$p = \frac{1 + \sum_{b=1}^B \mathbb{I}(t_b^* \le \hat{\Delta})}{B + 1}$$
- Multiple testing adjustment: Holm procedure for the two p-values at $\alpha = 0.05$.

**Success criterion for primary contrast:**
- Point estimate $\hat{\Delta} \le -0.0020$ (selected engineering significance criterion);
- 95% bootstrap CI upper bound $< 0$;
- $p < \alpha_{\text{Holm}}$.

Confirmation of anatomical benefit requires $\hat{\Delta} < 0$ and significance after Holm adjustment. Non-significance does not establish equivalence. All other comparisons are exploratory.

## 7. Few-Shot Analysis and Metrics
- Evaluated on non-overlapping bins of received feedback counts:
  $K \in \{0\}, \{1\text{–}2\}, \{3\text{–}5\}, \{6\text{–}10\}, \{11+\}$.
  Active users, decision count, proportion with $\gamma > 0$, Brier, and Log-loss reported per bin.
- Macro AUC includes only users with both outcome classes; fraction of evaluated users ($N_{\text{eval}} / N_{\text{total}}$) reported. Pooled AUC reported across all pooled pairs.
- Peak state size: maximum serialized JSON checkpoint size over the 180-day trajectory per user.
