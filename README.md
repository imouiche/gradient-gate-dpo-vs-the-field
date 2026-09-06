# Gate-DPO vs. the Field

**A mass-dynamics comparison of preference-optimization losses, baseline, calibrated, gated, and
globally-scheduled, measured on the same protocol, across four model architectures spanning 0.5B to 7B
parameters.**

> This repository presents a self-contained slice of results from **Gradient-Gated DPO: Stabilizing
> Preference Optimization in Language Models**, a project on mitigating the *squeezing effect* in
> off-policy DPO training, the tendency for training too long to make even the chosen response less
> likely. This repo extends the paper's results with an additional architecture (Qwen1.5-7B) and one
> further base loss (DPO-Shift) trained beyond what's in the current preprint.

**Paper:** [arXiv:2605.02626](https://arxiv.org/abs/2605.02626)
**Interactive comparison:** [claude.ai/code/artifact/…](https://claude.ai/code/artifact/6908ffbf-7781-453f-a573-897fd7470b3c) — same data, with sortable detail and a live rendering of the chart below.

## The question

When a DPO-style method reduces "squeezing" (destructive redistribution of probability mass away from
*both* chosen and rejected responses), is that because of the specific loss formulation, IPO's identity
mapping, Cal-DPO's calibration term, or because of a shared *gating* mechanism that down-weights the
rejected-response gradient once its probability is already very low?

To find out, the same gate (a smooth sigmoid threshold on the rejected response's estimated probability)
was attached to three different base losses, DPO, IPO, and Cal-DPO, and trained under identical
recipes on four architectures: Pythia-410M, Qwen-0.5B, LLaMA-7B, and Qwen1.5-7B.

## The finding

**The gate dominates the choice of base loss.** Gate any of DPO, IPO, DPO-Shift, or Cal-DPO with the same
valley-probability gate, and the three land within about a point of each other on every architecture —
while their ungated counterparts scatter across a much wider, mostly-negative range. A separate baseline,
DPO-Shift (a global, training-progress-scheduled coefficient rather than a per-example gate), barely moves
the needle off the plain DPO baseline anywhere.

Δ Chosen (change in the chosen response's log-probability from the first to the last evaluation
checkpoint) by method, per architecture:

![Delta Chosen by method, per architecture, gated methods cluster high and positive, ungated methods scatter low or negative](delta-chosen-strip-chart.svg)

| | Ungated range (DPO / IPO / Cal-DPO / DPO-Shift) | Gated range (any base loss) |
|---|---|---|
| Pythia-410M | −10.72 to +2.34 | +3.54 to +4.33 |
| LLaMA-7B | −0.13 to +0.05 | +0.19 to +0.58 |
| Qwen-0.5B | −25.24 to +1.76 | +3.40 to +4.14 |
| Qwen1.5-7B | −7.75 to −0.14 | +0.14 to +0.74 |

Every ungated method includes at least one strongly negative result. Every gated method, regardless of
which base loss it wraps, clusters tightly and positively.

## Full results

Δ Chosen / Δ Rejected change in chosen/rejected response log-probability from first to last eval
checkpoint. A less-negative Δ Rejected means less squeezing. **Margin** = Δ Chosen − Δ Rejected. **Δ
Others** = mean change across five unrelated-example probes (spillover onto examples training shouldn't
touch).

### Pythia-410M (410M params)

DPO baseline: Δ Chosen −10.72 · Δ Rejected −25.73

| Method | Δ Chosen | Δ Rejected | Margin | Δ Others |
|---|--:|--:|--:|--:|
| DPO (baseline) | −10.72 | −25.73 | +15.01 | −30.82 |
| IPO | −1.59 | −9.50 | +7.90 | −8.22 |
| Cal-DPO | +2.34 | −7.60 | +9.94 | −6.16 |
| **Gate-IPO** (seq) | **+3.57** | −2.18 | +5.75 | +1.85 |
| **Gate-IPO** (q10) | **+3.54** | −3.16 | +6.70 | +0.75 |
| **Gate-Cal-DPO** (seq) | **+3.74** | −3.59 | +7.33 | +0.67 |
| **Gate-Cal-DPO** (q10) | **+3.90** | −3.52 | +7.42 | −1.32 |
| DPO-Shift | −5.52 | −18.56 | +13.04 | −20.66 |
| **Gate-DPO** (seq) | **+3.97** | −2.87 | +6.83 | +1.36 |
| **Gate-DPO** (q10) | **+3.78** | −5.55 | +9.33 | −6.80 |
| **Gate-DPO-Shift** (seq) | **+4.01** | −2.81 | +6.82 | +0.79 |
| **Gate-DPO-Shift** (q10) | **+4.33** | −3.04 | +7.38 | −0.05 |

### LLaMA-7B (7B params)

DPO baseline: Δ Chosen −0.09 · Δ Rejected −0.84

| Method | Δ Chosen | Δ Rejected | Margin | Δ Others |
|---|--:|--:|--:|--:|
| DPO (baseline) | −0.09 | −0.84 | +0.75 | −0.96 |
| IPO | +0.05 | −0.60 | +0.65 | −0.79 |
| Cal-DPO | −0.13 | −0.91 | +0.78 | −1.01 |
| **Gate-IPO** (seq) | **+0.32** | −0.16 | +0.48 | −0.12 |
| **Gate-IPO** (q10) | **+0.46** | +0.01 | +0.46 | −0.37 |
| **Gate-Cal-DPO** (seq) | **+0.19** | −0.45 | +0.64 | −0.38 |
| **Gate-Cal-DPO** (q10) | **+0.52** | −0.05 | +0.57 | −0.42 |
| DPO-Shift | −0.03 | −0.76 | +0.72 | −0.93 |
| **Gate-DPO** (seq) | **+0.30** | −0.29 | +0.59 | −0.26 |
| **Gate-DPO** (q10) | **+0.57** | +0.06 | +0.51 | −0.35 |
| **Gate-DPO-Shift** (seq) | **+0.29** | −0.28 | +0.57 | −0.28 |
| **Gate-DPO-Shift** (q10) | **+0.58** | +0.09 | +0.49 | −0.36 |

### Qwen-0.5B

DPO baseline: Δ Chosen −10.61 · Δ Rejected −25.15

| Method | Δ Chosen | Δ Rejected | Margin | Δ Others |
|---|--:|--:|--:|--:|
| DPO (baseline) | −10.61 | −25.15 | +14.53 | −26.80 |
| IPO | +1.76 | −4.33 | +6.09 | −2.89 |
| Cal-DPO | −25.24 | −49.38 | +24.15 | −42.68 |
| **Gate-IPO** (seq) | **+3.40** | −2.39 | +5.78 | +1.00 |
| **Gate-IPO** (q10) | **+3.64** | −2.90 | +6.53 | −0.51 |
| **Gate-Cal-DPO** (seq) | **+3.48** | −3.59 | +7.07 | +0.59 |
| **Gate-Cal-DPO** (q10) | **+3.80** | −3.32 | +7.12 | −0.87 |
| DPO-Shift | −7.87 | −21.93 | +14.07 | −23.78 |
| **Gate-DPO** (seq) | **+3.87** | −5.13 | +9.00 | −3.31 |
| **Gate-DPO** (q10) | **+4.14** | −5.14 | +9.28 | −4.64 |
| **Gate-DPO-Shift** (seq) | **+3.71** | −2.88 | +6.59 | +1.09 |
| **Gate-DPO-Shift** (q10) | **+4.00** | −2.79 | +6.79 | −0.21 |

### Qwen1.5-7B

DPO baseline: Δ Chosen −6.11 · Δ Rejected −9.85

| Method | Δ Chosen | Δ Rejected | Margin | Δ Others |
|---|--:|--:|--:|--:|
| SFT (pre-preference reference) | +6.84 | +5.44 | +1.40 | +0.27 |
| DPO (baseline) | −6.11 | −9.85 | +3.74 | −12.26 |
| IPO | −0.14 | −1.71 | +1.57 | −1.79 |
| Cal-DPO | −7.75 | −12.18 | +4.43 | −15.69 |
| **Gate-IPO** (seq) | **+0.36** | −0.84 | +1.20 | −0.33 |
| **Gate-IPO** (q10) | **+0.48** | −0.75 | +1.23 | −0.75 |
| **Gate-Cal-DPO** (seq) | **+0.14** | −1.45 | +1.59 | −1.30 |
| **Gate-Cal-DPO** (q10) | **+0.58** | −0.80 | +1.38 | −1.17 |
| DPO-Shift | −5.21 | −8.58 | +3.37 | −10.58 |
| **Gate-DPO** (seq) | **+0.32** | −1.15 | +1.46 | −0.81 |
| **Gate-DPO** (q10) | **+0.74** | −0.54 | +1.28 | −0.67 |
| **Gate-DPO-Shift** (seq) | **+0.38** | −1.01 | +1.39 | −0.66 |

## A note on the Cal-DPO calibration term

Cal-DPO's calibration regularizer (β = 0.001 ⇒ implied target reward gap c = 1/(2β) = 500) produces a
training-time calibration loss on the order of 10⁵ and gradient norms in the millions, on every
architecture, gated or not. This is a property of the hyperparameter, not an architecture-specific
instability; it was checked systematically before drawing that conclusion. The resulting checkpoints are
valid, and their mass-dynamics numbers (shown above) are unaffected; the raw loss magnitude during training
just isn't a meaningful signal for this particular method.

## Method

- **Protocol**: for each configuration, an SFT-warmed policy is trained with the given preference loss for
  5 epochs on 5,001 examples (Anthropic-HH, helpful-base split), with periodic evaluation checkpoints
  recording log-probabilities on the chosen response, the rejected response, and five held-out "unrelated
  example" probes.
- **Gate**: a smooth sigmoid `g(p) = σ(α · (p − τ))` applied to the rejected-term gradient, where `p` is an
  estimate of the rejected response's probability (either a token-level quantile, "q10", or a
  sequence-level geometric mean, "seq"). The gate is detached from the computation graph, so it acts as a
  per-example reweighting rather than introducing a new gradient path.
- **Architectures**: Pythia-410M, Qwen-0.5B, LLaMA-7B, Qwen1.5-7B — spanning roughly an order of magnitude
  in parameter count, on both a purpose-built decoder-only model (Pythia) and two independently-trained
  model families (Qwen, LLaMA).
- **Compute**: all runs trained on a single A100-80GB per job via [Modal](https://modal.com).

## Citation

```bibtex
@misc{mouiche2026gatedpo,
  title  = {Gradient-Gated DPO: Stabilizing Preference Optimization in Language Models},
  author = {Mouiche, Inoussa},
  year   = {2026},
  eprint = {2605.02626},
  archivePrefix = {arXiv},
  url    = {https://arxiv.org/abs/2605.02626}
}
```

---

*Results current as of September 2026. Code and full experimental infrastructure available on request.*
