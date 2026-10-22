#!/usr/bin/env python
"""How much validation can the measured panel actually support? Measured.

Throughout this work the ranking has been described as "unvalidated against
activity". That is true but unquantified, and unquantified limits have been
wrong here before. The only measured ground truth available is the 48-member
assayed panel, with titre on the nylon-6,6 L1 diad. This asks what that panel
can and cannot establish.

THE OBVIOUS SEQUENCE-LEVEL PREDICTOR, TESTED HONESTLY. If homology to a known
active predicted activity, candidate prioritisation would be easy: score every
shortlist protein by identity to AS9/AS17/AS23. Leave-one-out on the panel
(each protein scored by its identity to the OTHER strong producers):

    AUROC                         0.596
    permutation p (20k shuffles)  0.300          <- not significant
    bootstrap 95% CI              0.234 - 0.848  <- spans chance
    Spearman vs titre             +0.310, p=0.032

    baseline, mean identity to the whole panel   AUROC 0.333
    baseline, SEQUENCE LENGTH ALONE              AUROC 0.659   <- beats it

So: no. Identity to the known actives does not separate them, and a trivial
length baseline scores higher. The nominally significant Spearman is a false
comfort - the permutation test on the actual discrimination task gives p=0.30.

THE DEEPER POINT, WHICH IS ABOUT THE PANEL AND NOT ABOUT ANY SCORE. There are
3 positives. The bootstrap CI on AUROC spans 0.23-0.85, so this panel could
not detect even a strongly predictive score, let alone rank 292 candidates
with one. Any model tuned until it ranks AS9/AS17/AS23 highly is fitting three
data points - which is exactly what the upstream project records about its own
structural score ("Parameters were selected on this same panel, so this is
retrospective enrichment").

Nothing is therefore fitted here, and no sequence-level ranker is shipped. The
honest conclusion is a sample-size statement: validating a candidate ranking
needs more measured actives, not a better model.

The panel is also genuinely diverse - median pairwise identity 0.331, maximum
0.489 - so none of this is a near-duplicate leakage artifact. That part is
reassuring: the diversity is real, there just are not enough positives.

Usage:
    python tools/validate_against_activity.py [activity threshold]
"""
from __future__ import annotations
from amide_synthesis.infrastructure.paths import PROJECT

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PANEL = Path(os.environ.get(
    'AMIDE_SYNTHETASE_PANEL',
    '/homes/rzhu/Enzyme_agent/exp_data/amide_synthetase/assay_data/'
    'panel_activity.csv'))
STRONG_UGML = float(os.environ.get('STRONG_PRODUCER_UGML', 0.5))
PERMUTATIONS = 20000


def pairwise_identity(sequences):
    """Global-alignment identity, normalised by the shorter sequence."""
    from Bio import Align
    aligner = Align.PairwiseAligner(scoring='blastp')
    aligner.mode = 'global'
    n = len(sequences)
    out = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            alignment = aligner.align(sequences[i], sequences[j])[0]
            matches = sum(1 for a, b in zip(alignment[0], alignment[1])
                          if a == b and a != '-')
            value = matches / min(len(sequences[i]), len(sequences[j]))
            out[i, j] = out[j, i] = value
    return out


def leave_one_out(identity, strong, n):
    """Identity to the strong producers OTHER than self."""
    return np.array([max([identity[i, s] for s in strong if s != i], default=0.0)
                     for i in range(n)])


def main(threshold=STRONG_UGML):
    from sklearn.metrics import roc_auc_score
    from scipy.stats import spearmanr

    if not PANEL.exists():
        raise SystemExit(f'panel not found: {PANEL}')
    panel = pd.read_csv(PANEL)
    panel = panel[panel.sequence.notna()].reset_index(drop=True)
    activity = panel.activity_N66L1_ugml_avg.to_numpy()
    label = (activity >= threshold).astype(int)
    strong = np.flatnonzero(label)

    print(f'panel n={len(panel)}  positives={len(strong)} '
          f'(>= {threshold} ug/ml): {panel.id[strong].tolist()}')
    if len(strong) < 2:
        raise SystemExit('need at least two positives for leave-one-out')

    identity = pairwise_identity(panel.sequence.tolist())
    upper = identity[np.triu_indices(len(panel), 1)]
    print(f'pairwise identity: median {np.median(upper):.3f} '
          f'max {upper.max():.3f}  (no near-duplicates -> no leakage)')

    score = leave_one_out(identity, strong, len(panel))
    observed = roc_auc_score(label, score)
    rho, p_rho = spearmanr(score, activity)

    rng = np.random.default_rng(0)
    null = np.array([roc_auc_score(rng.permutation(label), score)
                     for _ in range(PERMUTATIONS)])
    boot = []
    for _ in range(5000):
        idx = rng.choice(len(label), len(label), replace=True)
        if len(set(label[idx])) > 1:
            boot.append(roc_auc_score(label[idx], score[idx]))

    print('\nleave-one-out, identity to the other strong producers')
    print(f'  AUROC                  {observed:.3f}')
    print(f'  permutation p          {(null >= observed).mean():.3f}'
          f'   ({PERMUTATIONS} shuffles)')
    print(f'  bootstrap 95% CI       {np.percentile(boot, 2.5):.3f}'
          f' - {np.percentile(boot, 97.5):.3f}')
    print(f'  Spearman vs titre      {rho:+.3f}  p={p_rho:.3f}')

    mean_identity = np.array([
        np.mean([identity[i, j] for j in range(len(panel)) if j != i])
        for i in range(len(panel))])
    print('\nbaselines')
    print(f'  mean identity to panel AUROC {roc_auc_score(label, mean_identity):.3f}')
    print(f'  sequence length        AUROC '
          f'{roc_auc_score(label, panel.length_aa.to_numpy()):.3f}')

    print(f'\nwith {len(strong)} positives the CI spans chance: this panel cannot '
          'validate\nany candidate ranking. The limit is the number of measured '
          'actives, not the model.')


if __name__ == '__main__':
    main(float(sys.argv[1]) if len(sys.argv) > 1 else STRONG_UGML)
