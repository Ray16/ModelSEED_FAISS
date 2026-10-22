#!/usr/bin/env python
"""Project the nylon-6,6 queries into ModelSEED reaction space and mark where they land.

Shows three things: where each mechanistic framing of the target reaction falls,
where its nearest ModelSEED reactions are, and where the TOP SCREENED CANDIDATES
sit (labelled with the enzyme they resolve to). The candidate overlay appears
once 03/04/05 have run; before that the map still draws.

t-SNE cannot embed new points into an existing layout, so the projection is
recomputed over all corpus reactions plus the queries (PCA 50 -> t-SNE,
perplexity 50, seed 0). Coordinates are therefore NOT comparable point-for-point
with figures/tecrdb_shift/fig_tsne_ecfp4_drfp.png, and t-SNE does not preserve
cosine rank - the cosine ranking is authoritative, the map is for orientation.

The coordinate cache is keyed by the CONTENT of its inputs (corpus hash + query
SMILES), not by row count, so a changed corpus can never silently reuse a stale
layout.
"""
from __future__ import annotations

import hashlib
import sys

import numpy as np
import pandas as pd

from amide_synthesis.infrastructure.paths import ROOT, CACHE, FIG, RES
sys.path.insert(0, str(ROOT / '_lib'))
import _style  # noqa: F401,E402  sets 18 pt / dpi 300
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from amide_synthesis.reaction_search import encode_and_search as search
from amide_synthesis.reaction_search import corpus as _corpus


from amide_synthesis.infrastructure.workers import worker_count, describe
WORKERS = worker_count()

GRAY = '#B8BEC5'
ORANGE = '#D55E00'
BLUE = '#0072B2'
GREEN = '#009E73'
PURPLE = '#8D5AB5'
INK = '#252C34'

SHOWN = {
    'direct_condensation': ('Direct condensation', ORANGE),
    'atp_amp_ligase': ('ATP-dependent ligase', BLUE),
    'coa_aminolysis': ('Acyl-CoA aminolysis', GREEN),
}
NEIGHBOUR_TOP = 25
N_CANDIDATES = 10


def coordinates(x_full, xq, corpus_hash, query_smiles):
    key = hashlib.sha256(
        (corpus_hash + '|' + '|'.join(query_smiles)).encode()).hexdigest()[:16]
    path = CACHE / f'tsne_{key}.npy'
    if path.exists():
        coords = np.load(path)
        if len(coords) == len(x_full) + len(xq):
            print(f'reusing cached layout {path.name}', flush=True)
            return coords
    stacked = np.vstack([x_full, xq])
    print(f'PCA(50) on {stacked.shape}', flush=True)
    reduced = PCA(n_components=50, svd_solver='randomized',
                  random_state=0).fit_transform(stacked)
    print(f't-SNE on {WORKERS} workers', flush=True)
    import inspect
    iter_kw = ('max_iter' if 'max_iter' in inspect.signature(TSNE).parameters
               else 'n_iter')
    coords = TSNE(n_components=2, perplexity=50, init='pca', n_jobs=WORKERS,
                  random_state=0, verbose=1, **{iter_kw: 1000}).fit_transform(reduced)
    np.save(path, coords)
    return coords


def place_labels(ax, fig, points, texts, color, ink):
    """Annotate points so no two labels overlap, nor cover a marker.

    Matplotlib has no label de-collision. Each label is tried at a ring of
    candidate offsets (increasing radius, several angles) and the first position
    whose box clears every previously placed label, every marker and the axes
    edge is kept; a label with no free position is dropped rather than stacked
    on another, since the marker stays visible either way.

    Geometry is computed numerically from a ONE-OFF text measurement rather than
    by drawing each trial: this figure rasterises 33k background points, so a
    canvas redraw per trial costs minutes.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axes_box = ax.get_window_extent(renderer)
    scale = fig.dpi / 72.0                       # points -> pixels
    pad = 7.0

    sizes = []
    for label in texts:
        probe = ax.text(0, 0, label, fontsize=18, fontweight='semibold')
        extent = probe.get_window_extent(renderer)
        probe.remove()
        sizes.append((extent.width + 16, extent.height + 12))   # + bbox padding

    anchors = [ax.transData.transform(point) for point in points]
    marker_boxes = [(px - 14, py - 14, px + 14, py + 14) for px, py in anchors]

    def clear(box, others):
        x0, y0, x1, y1 = box
        for ox0, oy0, ox1, oy1 in others:
            if (x0 - pad < ox1 and x1 + pad > ox0
                    and y0 - pad < oy1 and y1 + pad > oy0):
                return False
        return True

    angles = [25, -25, 155, -155, 70, -70, 110, -110, 0, 180, 45, -45, 135, -135,
              90, -90]
    radii = [34, 52, 72, 96, 124, 156, 192, 232, 280]
    placed_boxes, placed = [], 0
    for (px, py), (width, height), label in zip(anchors, sizes, texts):
        best = None
        for radius in radii:
            for angle in angles:
                dx = radius * scale * np.cos(np.radians(angle))
                dy = radius * scale * np.sin(np.radians(angle))
                cx, cy = px + dx, py + dy
                x0 = cx if dx > 5 else (cx - width if dx < -5 else cx - width / 2)
                box = (x0, cy - height / 2, x0 + width, cy + height / 2)
                if (box[0] < axes_box.x0 + 3 or box[2] > axes_box.x1 - 3
                        or box[1] < axes_box.y0 + 3 or box[3] > axes_box.y1 - 3):
                    continue
                if clear(box, placed_boxes) and clear(box, marker_boxes):
                    best = (dx, dy, box)
                    break
            if best:
                break
        if not best:
            continue
        dx, dy, box = best
        placed_boxes.append(box)
        placed += 1
        ax.annotate(
            label, xy=ax.transData.inverted().transform((px, py)),
            xytext=(dx / scale, dy / scale), textcoords='offset points',
            fontsize=18, color=ink, fontweight='semibold', zorder=9,
            ha='left' if dx > 5 else 'right' if dx < -5 else 'center',
            va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor=color, linewidth=1.1, alpha=0.95),
            arrowprops=dict(arrowstyle='-', color=color, linewidth=1.1,
                            shrinkA=0, shrinkB=4))
    return placed


def load_candidates():
    """Top candidate reactions with the enzyme each resolves to, if available."""
    path = RES / 'candidates_ranked.tsv'
    if not path.exists():
        return pd.DataFrame()
    table = pd.read_csv(path, sep='\t')
    # Rank by cosine here, not by `priority`: this figure is about position in
    # embedding space, and the priority ordering demotes name-route evidence,
    # which would hide the highest-scoring hits (BibC at 0.91, PubC at 0.90).
    table = (table.sort_values('cosine_similarity', ascending=False)
                  .drop_duplicates('rxn_id'))
    return table.head(N_CANDIDATES)


from amide_synthesis.infrastructure.freshness import tracked_stage

@tracked_stage('figure')
def main():
    from amide_synthesis.infrastructure.paths import ensure_directories
    ensure_directories()
    queries = search.build_queries()
    xq = search.unit(search.featurise(queries.rxn_smiles.to_numpy()))
    full, x_full = _corpus.load()

    coords = coordinates(x_full, xq, _corpus.manifest()['corpus_sha256'],
                         list(queries.rxn_smiles))
    background, query_xy = coords[:len(x_full)], coords[len(x_full):]
    position = {rxn_id: i for i, rxn_id in enumerate(full.rxn_id)}

    neighbour_idx, top1 = {}, {}
    for name in SHOWN:
        qi = int(np.flatnonzero(queries['query'].to_numpy() == name)[0])
        sims = x_full @ xq[qi]
        neighbour_idx[name] = np.argsort(-sims)[:NEIGHBOUR_TOP]
        top1[name] = float(sims.max())

    candidates = load_candidates()

    fig = plt.figure(figsize=(25, 13.5))
    ax = fig.add_axes([0.03, 0.07, 0.52, 0.85])
    zooms = {name: fig.add_axes([0.595, y, 0.175, 0.255])
             for name, y in zip(SHOWN, (0.655, 0.365, 0.075))}
    legend_ax = fig.add_axes([0.795, 0.28, 0.21, 0.45]); legend_ax.axis('off')

    ax.scatter(*background.T, s=5, color=GRAY, alpha=0.35, linewidths=0,
               rasterized=True, zorder=1)

    handles = [Line2D([], [], marker='o', ls='none', ms=9, color=GRAY)]
    labels = [f'ModelSEED\n({len(full):,} reactions)']
    for name, (label, color) in SHOWN.items():
        qi = int(np.flatnonzero(queries['query'].to_numpy() == name)[0])
        idx = neighbour_idx[name]
        panel = zooms[name]
        panel.scatter(*background.T, s=14, color=GRAY, alpha=0.40, linewidths=0,
                      rasterized=True, zorder=1)
        for axis, (nsize, qsize) in ((ax, (70, 560)), (panel, (120, 760))):
            axis.scatter(*background[idx].T, s=nsize, facecolors='none',
                         edgecolors=color, linewidths=1.9, zorder=3)
            axis.scatter(*query_xy[qi], s=qsize, marker='*', color=color,
                         edgecolors='white', linewidths=1.8, zorder=6)
        handles.append(Line2D([], [], marker='*', ls='none', ms=22, color=color,
                              markeredgecolor='white'))
        labels.append(f'{label}\n(encoded query)')
        handles.append(Line2D([], [], marker='o', ls='none', ms=11,
                              markerfacecolor='none', markeredgecolor=color,
                              markeredgewidth=1.9))
        labels.append(f'its {NEIGHBOUR_TOP} nearest\nModelSEED reactions')

        focus = np.vstack([background[idx], query_xy[qi][None, :]])
        pad = 0.22 * max(float(np.ptp(focus[:, 0])), float(np.ptp(focus[:, 1])), 1.0)
        panel.set_xlim(focus[:, 0].min() - pad, focus[:, 0].max() + pad)
        panel.set_ylim(focus[:, 1].min() - pad, focus[:, 1].max() + pad)
        panel.set_xticks([]); panel.set_yticks([])
        panel.set_title(f'{label} · top-1 cosine {top1[name]:.2f}',
                        fontsize=18, pad=10, color=color)
        for spine in panel.spines.values():
            spine.set_edgecolor(color); spine.set_linewidth(1.4)

    ax.set_xlabel('t-SNE 1', labelpad=12)
    ax.set_ylabel('t-SNE 2', labelpad=12)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect('equal', adjustable='box'); ax.margins(0.08)
    fig.suptitle('Nylon-6,6 amide formation in ModelSEED reaction space | ECFP4 + DRFP',
                 x=0.03, ha='left', y=0.97, fontsize=22, fontweight='semibold')
    # ---- top candidate ENZYMES, drawn at the reaction that retrieved them
    if not candidates.empty:
        points, texts = [], []
        for row in candidates.itertuples():
            if row.rxn_id not in position:
                continue
            points.append(tuple(background[position[row.rxn_id]]))
            gene = (str(row.gene).split()[0]
                    if pd.notna(row.gene) and str(row.gene) != 'nan'
                    else str(row.uniprot))
            texts.append(f'{gene}  {row.cosine_similarity:.2f}')
        if points:
            xs, ys = zip(*points)
            ax.scatter(xs, ys, s=320, marker='D', color=PURPLE,
                       edgecolors='white', linewidths=1.7, zorder=7)
            shown = place_labels(ax, fig, points, texts, PURPLE, INK)
            print(f'labelled {shown}/{len(points)} candidates without overlap',
                  flush=True)
            handles.append(Line2D([], [], marker='D', ls='none', ms=13,
                                  color=PURPLE, markeredgecolor='white'))
            labels.append(f'Top {len(points)} candidate enzymes\n'
                          f'(at the reaction that found them)')

    legend_ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(0, 1),
                     borderaxespad=0, frameon=False, labelspacing=1.0,
                     handletextpad=0.9, fontsize=18)

    for text in fig.findobj(match=plt.Text):
        if text.get_text():
            assert text.get_fontsize() >= 18, (text.get_text(), text.get_fontsize())
    out = FIG / 'fig_tsne_nylon_query.png'
    fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    print('Saved', out, flush=True)

    pd.DataFrame({'query': queries['query'], 'tsne_1': query_xy[:, 0],
                  'tsne_2': query_xy[:, 1]}).to_csv(
        RES / 'query_tsne_coords.tsv', sep='\t', index=False)


if __name__ == '__main__':
    main()
