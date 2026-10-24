# GS native-substrate positive control — result

**Question.** The step-1 in-line attack-angle proxy was validated only in the ANL/
adenylating superfamily (SfaB panel, Pα). Our expanded pool — and the "only GS lands in
the both-feasible corner" headline — is glutamine-synthetase fold (acyl-phosphate, Pγ). Does
the proxy transfer? Test: fold PuuA (P78061) and GlnA3 (Q9KZC7) with their **native acyl
donor L-glutamate** under the identical step-1 protocol (protein + ATP + 2 Mg + acyl donor,
PG route, no amine — strict parallel to the adipate run) and compute the same metric:
γ-carboxylate O → Pγ → O3B(leaving) angle, <4 Å engagement filter, mean in-line angle among
engaged poses, conclusive only if ≥5.

**Setup.** L-glutamate (C5H8NO4⁻, L config), adipate→glutamate swap into each enzyme's own
step-1 YAML (same protein/MSA/ATP/2Mg). Boltz-2 v2.2.1 stock, exact/stock precision, seed 42,
50 diffusion samples, on lambda1 V100 (the accelerated H100 kit was queue-saturated; exact mode
is byte-identical to stock, and a 0.5 Å / degree-scale readout is insensitive to V100↔H100 noise).

## Result (γ-carboxylate O → Pγ)

| enzyme | substrate | min (Å) | median (Å) | poses <4 Å | candidate angle | conclusive (≥5)? |
|---|---|---:|---:|---:|---:|:--:|
| PuuA (P78061)  | **L-glutamate (native)** | 4.09 | 5.72 | **0 / 50** | — | **no** |
|                | adipate (target)         | 3.20 | 4.91 | 7 / 50 | 162.0° | yes |
| GlnA3 (Q9KZC7) | **L-glutamate (native)** | 3.92 | 4.41 | **1 / 50** | 161° (that pose) | **no** |
|                | adipate (target)         | 3.25 | 3.85 | 37 / 50 | 159.2° | yes |

The correct (γ) carboxylate points at Pγ in 90% (PuuA) / 100% (GlnA3) of poses, so glutamate
binds in the right place with the right carboxylate — it just sits ~0.5–1.5 Å beyond the 4 Å bar.
Figure: `control_glutamate_vs_adipate.png`.

## Interpretation

1. **Direction transfers; absolute engagement does not.** When the native γ-carboxylate does
   approach Pγ it attacks **in-line (161°)**, matching adipate (159–162°). So the in-line geometry
   the proxy rewards is physically real on the GS fold. But the **4 Å near-attack / engagement
   calibration** (from ANL adenylation, Pα) does **not** carry over: even the true substrate
   under-engages (median 4.4 / 5.7 Å; 0–1 poses <4 Å). By our own ≥5-engaged "conclusive" rule
   the **native reaction scores INCONCLUSIVE.**

2. **Engagement paradox.** Under identical conditions the **non-native adipate engages MORE than
   the native glutamate** (37 vs 1 and 7 vs 0 poses <4 Å). If `n_within_4A` / `near_attack_fraction`
   tracked catalytic competence, the native substrate should dominate — it doesn't. So **engagement
   count is not a trustworthy discriminator on the GS fold**, and the "both-feasible corner"
   *engagement* framing is weaker than the combined figure implies.

## Consequences for the project

- **Keep** reporting the GS in-line **angle** as a directional signal (it is reproduced for the true
  substrate), but **drop the absolute near-attack / engagement framing for cross-family** comparison.
  Do **not** elevate the GS "both-feasible corner" to a validated prediction — it stays hypothesis-grade,
  exactly as the transferability caveat warned.
- The within-family GS *ranking by angle* is not refuted (angle direction transfers); the *engagement
  counts* that feed near-attack fraction are.

## Caveats / what would upgrade this

- **No amine co-substrate.** This control (like the adipate step-1) omitted the amine. The real GS
  Michaelis complex has NH₃ (canonical) / putrescine (PuuA) bound, which may pull glutamate into the
  <4 Å pocket. Motivated next step: **amine-complete complex** (glutamate + native amine + ATP + 2 Mg).
  If engagement then reaches native-like <4 Å, the proxy transfers given a *complete* complex. Note the
  within-condition paradox (adipate > native, both amine-free) stands regardless.
- **Single seed (42).** Distances sit near the threshold; a few seeds would firm up the counts.

Data: `control_geometry.json`; inputs `inputs/*.yaml`; models `outputs/by_seed/<name>/s42/`.
