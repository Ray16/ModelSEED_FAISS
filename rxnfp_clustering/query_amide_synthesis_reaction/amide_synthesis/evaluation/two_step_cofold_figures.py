"""Integrate both modeled reaction stages into a labeled structural shortlist."""
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


def draw(data, path, title, colors):
    height = max(7, len(data)*.28 + 3)
    fig, axes = plt.subplots(1, 6, figsize=(20, height), sharey=True,
                             gridspec_kw={'width_ratios': [1, 1, 1.15, 1, 1, 1.15], 'wspace': .2})
    row_colors = data.family.map(colors)
    distance_values = data[['step1_plot_harmonic_distance_A', 'step2_plot_harmonic_distance_A']].to_numpy()
    distance_limits = (max(0, np.nanmin(distance_values)-.2), np.nanmax(distance_values)+.5)
    for stage in (1, 2):
        columns = [f'step{stage}_plot_weighted_angle_deg', f'step{stage}_plot_harmonic_distance_A',
                   f'step{stage}_near_attack_fraction']
        for index, column in enumerate(columns):
            ax = axes[(stage-1)*3+index]
            for row in range(0, len(data), 2):
                ax.axhspan(row-.5, row+.5, color='#f0f4f7', zorder=0)
            values = data[column].to_numpy(float)
            if index == 2:
                ax.barh(np.arange(len(data)), values, color=row_colors, height=.5, alpha=.8)
                ax.set(xlim=(0, 1.18), xlabel='Near-attack fraction')
                ax.set_xticks([0, .5, 1], ['0%', '50%', '100%'])
            else:
                ax.scatter(values, np.arange(len(data)), c=row_colors, s=34, edgecolors='white', linewidths=.4, zorder=3)
                if index == 0:
                    ax.set(xlim=(max(0, min(values.min()-6, 145 if stage == 1 else 85)),
                                 max(values.max()+14, 166 if stage == 1 else 125)), xlabel='Weighted angle (°)')
                    ax.axvspan(150 if stage == 1 else 90, 180 if stage == 1 else 120,
                               color='#228833', alpha=.08, zorder=0)
                else:
                    ax.set(xlim=distance_limits, xlabel='Harmonic distance (Å)')
                    ax.axvline(4 if stage == 1 else 3.5, color='#888888', linestyle='--', linewidth=.8)
            for row, value in enumerate(values):
                label = f'{value:.0%}' if index == 2 else f'{value:.1f}' if index == 0 else f'{value:.2f}'
                ax.annotate(label, (value, row), xytext=(5, 0), textcoords='offset points', va='center', fontsize=8)
            ax.set_ylim(len(data)-.5, -.5)
            ax.set_title(f'Step {stage}\n'+['Weighted angle', 'Harmonic distance', 'Near-attack poses'][index],
                         fontsize=11, weight='bold', pad=10)
            ax.spines[['top', 'right', 'left']].set_visible(False)
            ax.tick_params(axis='y', length=0)
            ax.grid(axis='x', alpha=.12)
    axes[0].set_yticks(np.arange(len(data)), data.accession)
    for label, color in zip(axes[0].get_yticklabels(), row_colors):
        label.set_color(color)
    fig.suptitle(title, fontsize=17, weight='bold', y=.97)
    fig.text(.5, .935, 'Same enzyme on each row · ordered by min(step 1 near-attack fraction, step 2 near-attack fraction)',
             ha='center', fontsize=11)
    families = data.family.value_counts()
    handles = [Line2D([0], [0], marker='o', linestyle='', color=colors[name], label=f'{name} (n={count})')
               for name, count in families.items()]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .058), ncol=2, frameon=False, fontsize=9)
    fig.text(.5, .018, 'Step 1 near attack: d <4 Å and θ >150°. Step 2: d <3.5 Å and 90° ≤ θ ≤120°. 50 poses/stage; pocket violations excluded from step 2.\n'
             'The minimum fraction is a structural prioritization heuristic, not a reaction probability or validated activity score. Step 2 uses forced pocket restraints.',
             ha='center', fontsize=9)
    fig.subplots_adjust(left=.095, right=.985, top=.88, bottom=.18)
    for extension in ['png', 'svg', 'pdf']:
        fig.savefig(path.with_suffix('.'+extension), dpi=220, facecolor='white')
    plt.close(fig)


def render(source, original, out, top=30):
    data = pd.read_csv(source, sep='\t')
    manifest_path = source.parent.parent.parent/'manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    title = manifest.get('substrates', 'Two-stage cofold cohort').capitalize()
    if manifest.get('sequence_count', len(data)) != len(data):
        raise ValueError('Plot table differs from the declared cohort size')
    metrics = [f'step{s}_plot_{metric}' for s in [1, 2] for metric in ['weighted_angle_deg', 'harmonic_distance_A']]
    metrics += ['step1_near_attack_fraction', 'step2_near_attack_fraction']
    if data.accession.duplicated().any() or not np.isfinite(data[metrics].to_numpy(float)).all():
        raise ValueError('Unique candidates with complete two-stage geometry are required')
    if not data[['step1_near_attack_fraction', 'step2_near_attack_fraction']].apply(lambda v: v.between(0, 1)).all().all():
        raise ValueError('Invalid near-attack fraction')
    data['two_step_min_fraction'] = data[['step1_near_attack_fraction', 'step2_near_attack_fraction']].min(axis=1)
    data = data.sort_values(['two_step_min_fraction', 'step1_near_attack_fraction', 'step2_near_attack_fraction', 'accession'],
                            ascending=[False, False, False, True]).reset_index(drop=True)
    data['two_step_geometry_rank'] = data.index+1
    selected = data.head(top).copy()
    original_ids = pd.read_csv(original, sep='\t').accession
    previous = data[data.accession.isin(original_ids)].copy()
    if len(previous) != len(original_ids):
        raise ValueError('Original comparison group identity mismatch')
    families = pd.concat([selected, previous]).family.value_counts().index
    palette = ['#4477AA', '#EE6677', '#228833', '#AA3377', '#EE7733', '#009988']
    colors = {name: palette[i % len(palette)] for i, name in enumerate(families)}
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 9, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    draw(selected, out/'two_step_side_by_side',
         f'{title}: combined two-step shortlist · top {len(selected)} of {len(data)}', colors)
    draw(previous, out/'original_upper_left_both_steps',
         f'Original step 1 upper-left group: both steps for the same {len(previous)} candidates', colors)
    data.to_csv(out/'all_candidates_two_step.tsv', sep='\t', index=False)
    selected.to_csv(out/'top_candidates_two_step.tsv', sep='\t', index=False)
    previous.to_csv(out/'original_upper_left_both_steps.tsv', sep='\t', index=False)
    display_columns = ['two_step_geometry_rank', 'accession', 'family', 'reaction', 'step1_plot_weighted_angle_deg',
                       'step1_plot_harmonic_distance_A', 'step1_near_attack_fraction', 'step2_plot_weighted_angle_deg',
                       'step2_plot_harmonic_distance_A', 'step2_near_attack_fraction', 'two_step_min_fraction']
    named = data[data.family.str.contains('SbnC|GlnA2', case=False, na=False)]
    page = ('<!doctype html><meta charset="utf-8"><title>Integrated two-step cofold results</title>'
            '<style>body{font:15px/1.5 system-ui;margin:30px;color:#183348}img{width:100%;max-width:1800px}'
            'table{border-collapse:collapse;font-size:12px}td,th{padding:7px;border-bottom:1px solid #ddd;text-align:left}</style>'
            f'<h1>{html.escape(title)}: both modeled reaction steps</h1>'
            f'<p>All {len(data)} candidates completed both stages, with 50 poses per stage. The main shortlist now considers both stages.</p>'
            '<p>Order: minimum of the two near-attack pose fractions, descending; ties use step 1 fraction, step 2 fraction, then accession. '
            'This heuristic emphasizes the weaker stage. It is not a reaction probability, and activity comparisons across families remain uncalibrated. '
            'Weighted angles and harmonic distances are also shown for each stage. Step 2 angles are evaluated against 90–120°; larger is not always better.</p>'
            '<img src="two_step_side_by_side.svg" alt="Aligned angle, distance and near-attack fraction panels for both steps">'
            '<p><a href="two_step_side_by_side.pdf">Main figure PDF</a> · <a href="two_step_side_by_side.png">Main figure PNG</a> · '
            '<a href="top_candidates_two_step.tsv">Top-candidate data</a> · <a href="all_candidates_two_step.tsv">All candidates</a> · '
            '<a href="../../evidence/substrate_pair_recheck_20261105.json">Actual model-input substrate recheck</a></p>'
            '<h2>Main shortlist</h2>'+selected[display_columns].to_html(index=False, escape=True, float_format=lambda x:f'{x:.3f}')
            +'<h2>Original upper-left group: both stages</h2><p>The same original candidates are retained here, including those with strong step 1 and weak step 2 geometry.</p>'
            '<img src="original_upper_left_both_steps.svg" alt="Both stages for the original upper-left group">'
            '<p><a href="original_upper_left_both_steps.pdf">Comparison PDF</a> · '
            '<a href="original_upper_left_both_steps.tsv">Comparison data</a></p>'
            '<h2>SbnC and GlnA2 annotations</h2>'+named[display_columns].to_html(index=False, escape=True, float_format=lambda x:f'{x:.3f}')
            +f'<p>Substrate audit: actual input configurations for all {len(data)} candidates/stages matched C5 glutarate in step 1 and cadaverine with '
            'activated glutaryl intermediate in step 2. See the linked verification receipt for activation-route counts. These activation-route assignments '
            'are campaign modeling assumptions. Step 2 uses per-enzyme forced pocket restraints; violating poses are excluded from its summaries.</p>')
    (out/'report.html').write_text(page)
    receipt = dict(created_utc=datetime.now(timezone.utc).isoformat(), source_sha256=digest(source),
                   original_group_sha256=digest(original), generator_sha256=digest(Path(__file__)),
                   candidates=len(data), displayed_candidates=len(selected),
                   ordering='min(step1_near_attack_fraction, step2_near_attack_fraction) descending; ties step1, step2 descending then accession',
                   top_family_counts=selected.family.value_counts().to_dict(),
                   outputs={p.name:digest(p) for p in out.iterdir() if p.name!='provenance.json'})
    (out/'provenance.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plot-data', type=Path, required=True)
    parser.add_argument('--original-group', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--top', type=int, default=30)
    args = parser.parse_args()
    if args.top < 1:
        parser.error('--top must be positive')
    render(args.plot_data.resolve(), args.original_group.resolve(), args.out_dir.resolve(), args.top)


if __name__=='__main__':
    main()
