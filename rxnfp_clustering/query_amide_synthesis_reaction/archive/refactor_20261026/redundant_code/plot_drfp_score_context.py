#!/usr/bin/env python3
"""Put shortlisted DRFP scores in context of all 480 eligible amidations."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent


def main():
    source = PROJECT / 'results/modelseed_nylon_top100'
    out = PROJECT / 'figures/drfp_score_context'
    out.mkdir(parents=True, exist_ok=True)
    population_path, shortlist_path = source / 'all_amide_scores.tsv', source / 'top5_reactions.tsv'
    population = pd.read_csv(population_path, sep='\t', low_memory=False)
    shortlist = pd.read_csv(shortlist_path, sep='\t')
    values = population.drfp_tanimoto.to_numpy()
    shortlist['percentile_strictly_below'] = [100 * np.mean(values < v - 1e-12) for v in shortlist.drfp_tanimoto]
    shortlist['retrieval_rank_top_percent'] = 100 * shortlist.amide_rank / len(population)
    shortlist[['rxn_id', 'drfp_tanimoto', 'amide_rank', 'percentile_strictly_below', 'retrieval_rank_top_percent']].to_csv(out / 'score_context.tsv', sep='\t', index=False)
    plt.rcParams.update({'svg.fonttype': 'none', 'pdf.fonttype': 42, 'font.size': 10})
    fig, ax = plt.subplots(figsize=(6.5, 3.1), layout='constrained')
    ax.hist(values, bins=np.linspace(0, 1, 31), color='#d4d9dd', edgecolor='white', label=f'All {len(values)} eligible reactions')
    median = float(np.median(values))
    ax.axvline(median, color='#2166ac', linewidth=2, linestyle='--', label=f'Median: {median:.3f}')
    ax.scatter(shortlist.drfp_tanimoto, np.zeros(len(shortlist)), marker='v', s=70, color='#b2182b', label='Five shortlisted reactions', zorder=4, clip_on=False)
    ax.set(xlim=(0, 1), xlabel='DRFP Tanimoto similarity to nylon-6,6', ylabel='Reactions')
    ax.set_title('Moderate scores, high relative similarity', loc='left', weight='bold', fontsize=12)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc='upper right')
    for ext in ['png', 'svg', 'pdf']: fig.savefig(out / ('drfp_score_context.' + ext), dpi=220, bbox_inches='tight')
    plt.close(fig)
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), population_size=len(population),
                    median=median, selected_range=[float(shortlist.drfp_tanimoto.min()), float(shortlist.drfp_tanimoto.max())],
                    score_scope='Full native reactions with cofactors; route-matched nylon targets; saved ModelSEED pH-7 convention',
                    percentile_definition='Percent of eligible reactions with strictly lower DRFP Tanimoto; ties share a percentile',
                    input_sha256={str(p): digest(p) for p in [population_path, shortlist_path, Path(__file__).resolve()]},
                    output_sha256={p.name: digest(p) for p in out.iterdir() if p.is_file() and p.name != 'manifest.json'})
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(out / 'drfp_score_context.png')


if __name__ == '__main__': main()
