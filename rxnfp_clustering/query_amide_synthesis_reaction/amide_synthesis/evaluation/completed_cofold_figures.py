"""Visualize a complete two-stage cofold cohort without changing saved analysis."""
import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render(root, out):
    source = root / 'analysis/final'
    complete = json.loads((source / 'complete.json').read_text())
    manifest = json.loads((root / 'manifest.json').read_text())
    if complete['status'] != 'complete' or complete['manifest_sha256'] != digest(root / 'manifest.json'):
        raise ValueError('A completed, matching final analysis is required')
    candidates = pd.read_csv(source / 'candidates.tsv', sep='\t')
    poses = pd.read_csv(source / 'poses.tsv', sep='\t')
    if (len(candidates) != manifest['sequence_count'] or candidates.accession.duplicated().any()
            or len(poses) != complete['poses'] or set(poses.accession) != set(candidates.accession)):
        raise ValueError('Cohort identity/count mismatch')
    counts = poses.groupby(['accession', 'stage']).size()
    if len(counts) != 2 * len(candidates) or not counts.eq(50).all() or set(poses.stage) != {1, 2}:
        raise ValueError('Every candidate must have 50 poses in each stage')
    if poses.duplicated(['accession', 'stage', 'model']).any():
        raise ValueError('Duplicate pose identity')
    rows = []
    for (accession, stage), group in poses.groupby(['accession', 'stage']):
        if stage == 2:
            group = group[~group.pocket_satisfied.astype(str).str.lower().eq('false')]
        distance = group.distance_A.to_numpy(float)
        angle = group.angle_deg.to_numpy(float)
        if len(group) and (not np.isfinite(distance).all() or not np.isfinite(angle).all()
                           or np.any(distance <= 0) or np.any((angle < 0) | (angle > 180))):
            raise ValueError('Invalid pose geometry')
        rows.append(dict(accession=accession, stage=stage, valid=len(group),
                         harmonic_distance_A=len(group) / np.sum(1 / distance) if len(group) else np.nan,
                         weighted_angle_deg=np.sum(angle / distance) / np.sum(1 / distance) if len(group) else np.nan,
                         near_attack_fraction=np.mean((distance < 4) & (angle > 150)) if stage == 1
                         else np.mean((distance < 3.5) & (angle >= 90) & (angle <= 120)) if len(group) else np.nan))
    aggregate = pd.DataFrame(rows)
    table = candidates.copy()
    for stage in (1, 2):
        part = aggregate[aggregate.stage.eq(stage)].drop(columns='stage').rename(
            columns={c: f'step{stage}_plot_{c}' for c in aggregate.columns if c not in ['accession', 'stage']})
        table = table.merge(part, on='accession', validate='one_to_one')
        if not np.allclose(table[f'step{stage}_near_attack_fraction'],
                           table[f'step{stage}_plot_near_attack_fraction'], equal_nan=True):
            raise ValueError('Recomputed near-attack fractions disagree with final analysis')
    if not np.allclose(table.step1_weighted_angle_deg, table.step1_plot_weighted_angle_deg):
        raise ValueError('Weighted angles disagree with final analysis')
    if not np.array_equal(table.step2_valid_poses, table.step2_plot_valid):
        raise ValueError('Pocket-valid pose counts disagree with final analysis')
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / 'candidate_plot_data.tsv', sep='\t', index=False)
    reactions = sorted(table.reaction.unique())
    palette = ['#4477AA', '#EE6677', '#228833', '#CCBB44', '#66CCEE',
               '#AA3377', '#BBBBBB', '#000000', '#EE7733', '#009988']
    colors = {r: palette[i % len(palette)] for i, r in enumerate(reactions)}
    plt.rcParams.update({'font.size': 10, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.8))
    for reaction, group in table.groupby('reaction'):
        options = dict(s=29, color=colors[reaction], alpha=.68, edgecolors='white', linewidths=.35)
        axes[0].scatter(group.step1_plot_harmonic_distance_A, group.step1_weighted_angle_deg,
                        label=f'{reaction} (n={len(group)})', **options)
        axes[1].scatter(group.step2_plot_harmonic_distance_A, group.step2_plot_weighted_angle_deg, **options)
        axes[2].scatter(group.step1_near_attack_fraction, group.step2_near_attack_fraction, **options)
    for i, threshold in enumerate([4, 3.5]):
        ax = axes[i]
        lower = max(.1, table[f'step{i+1}_plot_harmonic_distance_A'].min() * .8)
        upper = max(threshold * 1.2, table[f'step{i+1}_plot_harmonic_distance_A'].max() * 1.2)
        ax.set(xscale='log', xlim=(lower, upper), ylim=(0, 182),
               xlabel='Harmonic mean attack distance (Å; log scale)',
               ylabel='Proximity-weighted attack angle (°)')
        ax.axvline(threshold, color='#777777', ls='--', lw=1)
        bottom, height = (150, 30) if i == 0 else (90, 30)
        if threshold > lower:
            ax.add_patch(Rectangle((lower, bottom), threshold-lower, height,
                                  color='#228833', alpha=.10, zorder=0))
        ax.set_title(f'{chr(65+i)}  Step {i+1}: ensemble geometry', loc='left', weight='bold')
    axes[2].set(xlim=(-.025, 1.025), ylim=(-.025, 1.025),
                xlabel='Step 1 near-attack pose fraction', ylabel='Step 2 near-attack pose fraction',
                title='C  Geometry across both stages')
    for ax in axes:
        ax.grid(alpha=.15)
        ax.spines[['top', 'right']].set_visible(False)
    title = manifest.get('substrates', manifest['campaign_id']).capitalize()
    fig.suptitle(f'{title}: all {len(table)} candidates · 50 poses per stage', fontsize=15, weight='bold')
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=5,
               bbox_to_anchor=(.5, .055), fontsize=8, frameon=False)
    fig.text(.5, .018, 'Shaded regions are pose-level reference criteria; ensemble means do not establish near-attack occupancy or enzyme activity.',
             ha='center', fontsize=9)
    fig.subplots_adjust(left=.055, right=.985, top=.86, bottom=.29, wspace=.31)
    for extension in ['png', 'svg', 'pdf']:
        fig.savefig(out / f'completed_geometry.{extension}', dpi=220, facecolor='white')
    plt.close(fig)

    interactive = make_subplots(rows=1, cols=3, subplot_titles=[
        'Step 1: weighted angle vs distance', 'Step 2: weighted angle vs distance', 'Near-attack fractions in both stages'])
    for reaction, group in table.groupby('reaction'):
        custom = group[['accession', 'family', 'step1_engaged_poses', 'step1_near_attack_fraction',
                        'step2_near_attack_fraction', 'step2_valid_poses']].to_numpy()
        hover = ('<b>%{customdata[0]}</b><br>%{customdata[1]}<br>x=%{x:.3f}; y=%{y:.3f}'
                 '<br>Step 1 close poses: %{customdata[2]}/50<br>Step 1 near attack: %{customdata[3]:.2f}'
                 '<br>Step 2 near attack: %{customdata[4]:.2f}<br>Step 2 valid poses: %{customdata[5]}<extra></extra>')
        coordinates = [(group.step1_plot_harmonic_distance_A, group.step1_weighted_angle_deg),
                       (group.step2_plot_harmonic_distance_A, group.step2_plot_weighted_angle_deg),
                       (group.step1_near_attack_fraction, group.step2_near_attack_fraction)]
        for col, (x, y) in enumerate(coordinates, 1):
            interactive.add_trace(go.Scatter(x=x, y=y, mode='markers', name=f'{reaction} (n={len(group)})',
                                            legendgroup=reaction, showlegend=col == 1, customdata=custom,
                                            hovertemplate=hover, marker=dict(color=colors[reaction], size=8, opacity=.7)), row=1, col=col)
    for col in [1, 2]:
        interactive.update_xaxes(type='log', title_text='Harmonic distance (Å)', row=1, col=col)
        interactive.update_yaxes(range=[0, 182], title_text='Weighted angle (°)', row=1, col=col)
        interactive.add_vline(x=4 if col == 1 else 3.5, line_dash='dash', line_color='#888888', row=1, col=col)
    interactive.update_xaxes(title_text='Step 1 near-attack fraction', range=[-.025, 1.025], row=1, col=3)
    interactive.update_yaxes(title_text='Step 2 near-attack fraction', range=[-.025, 1.025], row=1, col=3)
    interactive.update_layout(height=600, template='plotly_white', margin=dict(t=55, b=120),
                              legend=dict(orientation='h', y=-.25))
    first = int(table.step1_near_attack_fraction.gt(0).sum())
    second = int(table.step2_near_attack_fraction.gt(0).sum())
    both = int((table.step1_near_attack_fraction.gt(0) & table.step2_near_attack_fraction.gt(0)).sum())
    display = table[['accession', 'reaction', 'family', 'step1_weighted_angle_deg', 'step1_plot_harmonic_distance_A',
                     'step1_near_attack_fraction', 'step2_plot_weighted_angle_deg', 'step2_plot_harmonic_distance_A',
                     'step2_near_attack_fraction', 'step2_valid_poses']].sort_values(['reaction', 'accession'])
    primary = ''
    upper = root / 'analysis/upper_left'
    if (upper / 'upper_left_side_by_side.svg').exists():
        selected = json.loads((upper / 'provenance.json').read_text())
        if selected['source_sha256'] != digest(out / 'candidate_plot_data.tsv'):
            raise ValueError('Primary figure belongs to a different candidate table')
        relative = html.escape(Path(os.path.relpath(upper, out)).as_posix())
        primary = (f'<h2>Main visualization: {selected["candidates"]} upper-left candidates</h2>'
                   f'<a href="{relative}/report.html"><img src="{relative}/upper_left_side_by_side.svg" '
                   'style="width:100%;max-width:1100px" alt="Weighted angle and harmonic distance side by side, with enzyme accessions"></a>'
                   f'<p><a href="{relative}/report.html">Candidate identities and metrics</a> · '
                   f'<a href="{relative}/upper_left_side_by_side.pdf">Download main figure (PDF)</a></p>')
    page = ('<!doctype html><meta charset="utf-8"><title>Completed cofold geometry</title>'
            '<style>body{font:15px/1.5 system-ui;margin:30px;color:#183348}table{border-collapse:collapse;font-size:12px}'
            'td,th{padding:7px;border-bottom:1px solid #ddd;text-align:left}input{padding:10px;width:420px}</style>'
            f'<h1>{html.escape(title)}: completed cofold results</h1>'
            f'<p><b>{len(table)} candidates · {len(poses):,} structures · both stages complete.</b></p>'
            + primary + '<h2>Full cohort: both stages</h2>'
            + f'<p>At least one near-attack pose: step 1, {first}; step 2, {second}; both stages, {both}. '
            'These counts include candidates with only one qualifying pose; they are not activity-hit counts.</p>'
            '<p>Hover over a point for its accession and metrics. Click a reaction in the legend to hide it; double-click to isolate it.</p>'
            + interactive.to_html(full_html=False, include_plotlyjs=True)
            + '<p>Weighted angle = Σ(angle/distance) / Σ(1/distance). Harmonic distance = n / Σ(1/distance). '
            'Step 1 near attack: distance &lt;4 Å and angle &gt;150°. Step 2 near attack: distance &lt;3.5 Å and angle 90–120°. '
            'Step 2 uses activated intermediate plus diamine, with per-enzyme pocket restraints derived from step 1. '
            'Pocket-violating poses are excluded from step 2 summaries. A larger step 2 angle is not always better. '
            'Ensemble mean coordinates need not occur in a single pose. Geometry does not establish enzyme activity; '
            'comparisons across enzyme families remain uncalibrated.</p>'
            '<p><a href="completed_geometry.png">PNG</a> · <a href="completed_geometry.pdf">PDF</a> · '
            '<a href="completed_geometry.svg">SVG</a> · <a href="candidate_plot_data.tsv">All plotted data</a></p>'
            '<h2>All candidates</h2><input id="filter" placeholder="Filter by accession, family or reaction">'
            + display.to_html(index=False, table_id='candidates', float_format=lambda value: f'{value:.3f}', escape=True)
            + '<script>document.getElementById("filter").addEventListener("input",function(){'
            'const q=this.value.toLowerCase();document.querySelectorAll("#candidates tbody tr").forEach('
            'row=>row.hidden=!row.textContent.toLowerCase().includes(q));});</script>')
    (out / 'report.html').write_text(page)
    receipt = dict(created_utc=datetime.now(timezone.utc).isoformat(), candidates=len(table), structures=len(poses),
                   step1_any_near_attack=first, step2_any_near_attack=second, both_stages_any_near_attack=both,
                   excluded_step2_pocket_violations=int(table.step2_pocket_violations.sum()),
                   source_sha256={p.name: digest(p) for p in [source / 'candidates.tsv', source / 'poses.tsv', source / 'complete.json']},
                   generator_sha256=digest(Path(__file__)),
                   outputs={p.name: digest(p) for p in out.iterdir() if p.suffix in ['.tsv', '.png', '.svg', '.pdf', '.html']})
    (out / 'provenance.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path)
    args = parser.parse_args()
    render(args.run_dir.resolve(), args.out_dir or args.run_dir / 'analysis/visualizations')


if __name__ == '__main__':
    main()
