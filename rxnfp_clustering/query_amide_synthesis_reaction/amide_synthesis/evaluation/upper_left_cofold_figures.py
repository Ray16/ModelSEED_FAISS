"""Show labeled short-distance/high-angle candidates in aligned metric panels."""
import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render(source, out, min_angle=150, max_distance=4):
    data = pd.read_csv(source, sep='\t')
    if data.accession.duplicated().any():
        raise ValueError('Candidate identities must be unique')
    mask = (data.step1_weighted_angle_deg > min_angle) & (data.step1_plot_harmonic_distance_A < max_distance)
    selected = data[mask].sort_values(['step1_weighted_angle_deg', 'step1_plot_harmonic_distance_A'],
                                    ascending=[False, True]).reset_index(drop=True)
    if selected.empty:
        raise ValueError('No candidates meet the declared upper-left criteria')
    out.mkdir(parents=True, exist_ok=True)
    selected.to_csv(out/'upper_left_candidates.tsv', sep='\t', index=False)
    families = selected.family.value_counts().index.tolist()
    palette = ['#4477AA', '#EE6677', '#228833', '#AA3377', '#EE7733', '#009988']
    colors = {name: palette[i % len(palette)] for i, name in enumerate(families)}
    row_colors = selected.family.map(colors)
    height = max(6.5, len(selected)*.28 + 3.2)
    plt.rcParams.update({'font.size': 10, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(1, 2, figsize=(11.5, height), sharey=True,
                             gridspec_kw={'wspace': .12})
    rows = np.arange(len(selected))
    metrics = ['step1_weighted_angle_deg', 'step1_plot_harmonic_distance_A']
    limits = [(min_angle - 1, max(min_angle + 5, selected[metrics[0]].max() + 2.2)),
              (max(0, selected[metrics[1]].min() - .12), max_distance + .12)]
    labels = ['Proximity-weighted attack angle (°)', 'Harmonic mean attack distance (Å)']
    for i, (ax, metric, label, limit) in enumerate(zip(axes, metrics, labels, limits)):
        for row in rows[::2]:
            ax.axhspan(row-.5, row+.5, color='#f2f5f8', zorder=0)
        ax.scatter(selected[metric], rows, c=row_colors, s=48, edgecolor='white', linewidth=.6, zorder=3)
        for row, value in enumerate(selected[metric]):
            ax.annotate(f'{value:.1f}' if i == 0 else f'{value:.2f}', (value, row),
                        xytext=(7, 0), textcoords='offset points', va='center', fontsize=9)
        ax.set(xlabel=label, xlim=limit, ylim=(len(selected)-.5, -.5))
        ax.set_title('A  Weighted attack angle' if i == 0 else 'B  Harmonic distance',
                     weight='bold', loc='left', pad=12)
        ax.grid(axis='x', alpha=.15)
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.tick_params(axis='y', length=0)
        ax.axvline(min_angle if i == 0 else max_distance, color='#888888', linestyle='--', linewidth=1)
    axes[0].set_yticks(rows, selected.accession.tolist())
    for label, color in zip(axes[0].get_yticklabels(), row_colors):
        label.set_color(color)
    fig.suptitle(f'Step 1 upper-left candidates: {len(selected)} enzymes', fontsize=16, weight='bold', y=.97)
    fig.text(.5, .935, f'Weighted angle >{min_angle:g}° and harmonic distance <{max_distance:g} Å · one row per enzyme · sorted by angle',
             ha='center', fontsize=10)
    handles = [Line2D([0], [0], marker='o', linestyle='', color=colors[family],
                      label=f'{family} (n={int(selected.family.eq(family).sum())})') for family in families]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .05), ncol=1,
               frameon=False, fontsize=9, handletextpad=.6)
    fig.text(.5, .018, '50 poses per candidate. Angle = Σ(θ/d) / Σ(1/d); harmonic distance = n / Σ(1/d).\n'
             'These are step 1 ensemble summaries; step 2 results are retained in the accompanying table.',
             ha='center', fontsize=9)
    fig.subplots_adjust(left=.16, right=.97, top=.885, bottom=.19)
    for extension in ['png', 'pdf', 'svg']:
        fig.savefig(out/f'upper_left_side_by_side.{extension}', dpi=220, facecolor='white')
    plt.close(fig)
    display = selected[['accession', 'family', 'reaction', 'step1_weighted_angle_deg',
                        'step1_plot_harmonic_distance_A', 'step1_near_attack_fraction', 'step2_near_attack_fraction']]
    page = ('<!doctype html><meta charset="utf-8"><title>Upper-left enzyme candidates</title>'
            '<style>body{font:15px/1.5 system-ui;margin:30px;color:#183348}img{max-width:1100px;width:100%}'
            'td,th{padding:7px;border-bottom:1px solid #ddd;text-align:left}table{border-collapse:collapse;font-size:12px}</style>'
            f'<h1>{len(selected)} upper-left candidates</h1><p>Step 1 proximity-weighted angle &gt;{min_angle:g}° '
            f'and harmonic distance &lt;{max_distance:g} Å. Rows are ordered by weighted angle.</p>'
            '<img src="upper_left_side_by_side.svg" alt="Aligned weighted-angle and harmonic-distance panels with accession labels">'
            '<p><a href="upper_left_side_by_side.pdf">PDF</a> · <a href="upper_left_side_by_side.png">PNG</a> · '
            '<a href="upper_left_candidates.tsv">All selected candidates and step 2 metrics</a></p>'
            + display.to_html(index=False, escape=True, float_format=lambda v: f'{v:.3f}')
            + '<p>Near-attack fractions use distance &lt;4 Å and angle &gt;150° for step 1, and '
            'distance &lt;3.5 Å and angle 90–120° for pocket-valid step 2 poses. '
            'Ensemble means do not establish individual-pose occupancy or enzyme activity.</p>')
    (out/'report.html').write_text(page)
    receipt = dict(created_utc=datetime.now(timezone.utc).isoformat(), source=str(source),
                   source_sha256=digest(source), generator_sha256=digest(Path(__file__)),
                   criteria=dict(step1_weighted_angle_deg_greater_than=min_angle,
                                 step1_harmonic_distance_A_less_than=max_distance),
                   candidates=len(selected), family_counts=selected.family.value_counts().to_dict(),
                   outputs={p.name: digest(p) for p in out.iterdir() if p.name != 'provenance.json'})
    (out/'provenance.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plot-data', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--min-angle', type=float, default=150)
    parser.add_argument('--max-distance', type=float, default=4)
    args = parser.parse_args()
    render(args.plot_data.resolve(), args.out_dir.resolve(), args.min_angle, args.max_distance)


if __name__ == '__main__':
    main()
