#!/usr/bin/env python3
"""Compare native reactions with nylon-6,6 using explicitly labeled methods.

Run with the palm Python. Display three embedding/metric combinations in one
compact red-blue heatmap, retaining every native cofactor and selecting the
corresponding ATP route for the nylon target. The full nine-method score matrix
remains available as data. CphA1's symbolic polymer scores remain missing.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import html
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from drfp import DrfpEncoder
from rdkit import Chem, rdBase
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator as fpg
from rdkit.Chem.MolStandardize import rdMolStandardize

from draw_native_reactions import SCHEMES, Canvas, molecule, validate
from amide_synthesis.reaction_search.encode_and_search import load_species
from amide_synthesis.reaction_search.acylation import count_free_amines

PROJECT = Path(__file__).resolve().parent
UNCHARGER = rdMolStandardize.Uncharger()
ROW_KEYS = ['DdaG', 'SfaB', 'CphA1_Asp', 'CphA1_Arg', 'McbA', 'PylC']
ROW_LABELS = ['Nylon target', 'DdaG', 'SfaB', 'CphA1 · Asp', 'CphA1 · Arg', 'McbA', 'PylC']
METHODS = [
    ('drfp', 'tanimoto', 'DRFP\nTanimoto'), ('drfp', 'cosine', 'DRFP\ncosine'),
    ('ecfp4_concat', 'tanimoto', 'ECFP4 sides\nTanimoto'),
    ('ecfp4_diff', 'cosine', 'ECFP4 diff\ncosine'),
    ('atompair_diff', 'cosine', 'AtomPair diff\ncosine'),
    ('maccs_diff', 'cosine', 'MACCS diff\ncosine'),
    ('ecfp4_drfp', 'cosine', 'ECFP4 diff + DRFP\ncosine'),
    ('bert_ft', 'cosine', 'rxnfp FT\ncosine'),
    ('bert_pretrained', 'cosine', 'rxnfp pretrained\ncosine'),
]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def neutral(smiles):
    m = Chem.MolFromSmiles(smiles)
    if m is None or any(a.GetAtomicNum() == 0 for a in m.GetAtoms()):
        raise ValueError('Fingerprint inputs require complete, parseable molecules')
    return Chem.MolToSmiles(UNCHARGER.uncharge(m))


def counts(smiles):
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    return Counter(a.GetAtomicNum() for a in m.GetAtoms()), Chem.GetFormalCharge(m)


def reaction(left, right):
    left, right = list(map(neutral, left)), list(map(neutral, right))
    lc = sum((counts(s)[0] for s in left), Counter())
    rc = sum((counts(s)[0] for s in right), Counter())
    if lc != rc or sum(counts(s)[1] for s in left) != sum(counts(s)[1] for s in right):
        raise ValueError('Unbalanced reaction input')
    return '.'.join(sorted(left)) + '>>' + '.'.join(sorted(right))


def build_inputs():
    species = {k: neutral(s) for k, s in load_species().items()}
    acid, amine = species['ADIPATE'], species['HMDA']
    product = neutral('O=C(O)CCCCC(=O)NCCCCCCN')
    assert count_free_amines(Chem.MolFromSmiles(amine)) == 2
    records = [
        dict(name='target_amp', rxn_smiles=reaction([acid, amine, species['ATP']], [product, species['AMP'], species['PPI']])),
        dict(name='target_adp', rxn_smiles=reaction([acid, amine, species['ATP']], [product, species['ADP'], species['PI']])),
    ]
    for scheme in SCHEMES:
        validate(scheme)
        if scheme['enzyme'] == 'SuCphA1':
            continue
        left = [s for _, s in scheme['reactants']]
        right = [scheme['product'][1]]
        extra = ([species['PPI']] if scheme['key'] == 'SfaB' else
                 [species['ADP'], species['PI']] if scheme['key'] == 'PylC' else
                 [species['AMP'], species['PPI']])
        records.append(dict(name=scheme['key'] + '_full', rxn_smiles=reaction(left + [species['ATP']], right + extra)))
    return records, acid, amine, product


def fingerprint_arrays(records):
    rxns = [r['rxn_smiles'] for r in records]
    drfp = np.asarray(DrfpEncoder.encode(rxns, n_folded_length=2048, radius=3), dtype=np.float32)
    generators = {'ecfp4': fpg.GetMorganGenerator(radius=2, fpSize=2048),
                  'atompair': fpg.GetAtomPairGenerator(fpSize=2048)}
    arrays = {'drfp': drfp}
    for name, generator in generators.items():
        differences, concats = [], []
        for rxn in rxns:
            sums, bits = [], []
            for side in rxn.split('>>'):
                mols = [Chem.MolFromSmiles(s) for s in side.split('.')]
                sums.append(sum((generator.GetCountFingerprintAsNumPy(m) for m in mols), np.zeros(2048)))
                bits.append(np.maximum.reduce([generator.GetFingerprintAsNumPy(m) for m in mols]))
            differences.append(sums[1] - sums[0])
            concats.append(np.concatenate(bits))
        arrays[name + '_diff'] = np.asarray(differences, dtype=np.float32)
        if name == 'ecfp4':
            arrays['ecfp4_concat'] = np.asarray(concats, dtype=np.float32)
    arrays['maccs_diff'] = np.asarray([
        sum((np.asarray(MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(s))) for s in r['rxn_smiles'].split('>>')[1].split('.')), np.zeros(167))
        - sum((np.asarray(MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(s))) for s in r['rxn_smiles'].split('>>')[0].split('.')), np.zeros(167))
        for r in records], dtype=np.float32)
    arrays['ecfp4_drfp'] = np.hstack([arrays['ecfp4_diff'], drfp])
    return arrays


def similarity(a, b, metric):
    a, b = a.astype(float), b.astype(float)
    if metric == 'tanimoto':
        assert set(np.unique(a)) <= {0, 1} and set(np.unique(b)) <= {0, 1}
        union = np.logical_or(a, b).sum()
        return float(np.logical_and(a, b).sum() / union) if union else np.nan
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denominator) if denominator else np.nan


def target_diagram(out, acid, amine, product):
    canvas = Canvas(1400, 300)
    canvas.text(20, 8, 'Nylon-6,6 diad target', 27, bold=True)
    for s, x, label in [(acid, 15, 'Adipate'), (amine, 340, 'Hexamethylenediamine')]:
        canvas.mol(molecule(s), x, 55, 295, 180)
        canvas.text(x + 147, 250, label, 19, anchor='middle')
    canvas.text(325, 140, '+', 29, anchor='middle')
    canvas.line(665, 155, 775, 155)
    canvas.line(761, 146, 775, 155); canvas.line(761, 164, 775, 155)
    canvas.text(720, 115, 'Condensation', 16, anchor='middle')
    canvas.text(720, 177, '+ H₂O', 18, anchor='middle')
    mol = molecule(product)
    match = mol.GetSubstructMatch(Chem.MolFromSmarts('C(=O)N'))
    canvas.mol(mol, 800, 55, 580, 180, mol.GetBondBetweenAtoms(match[0], match[2]).GetIdx())
    canvas.text(1090, 250, 'Nylon-6,6 monoamide diad', 19, anchor='middle')
    canvas.save(out / 'target_reaction')


def heatmap(out, panels, methods):
    """One compact view; each native reaction uses its matching ATP route."""
    plt.rcParams.update({'font.size': 10, 'svg.fonttype': 'none', 'pdf.fonttype': 42})
    columns = [0, 6, 7]
    # panels[0] = AMP-query, panels[1] = ADP-query; CphA1 remains missing.
    selections = [('DdaG', 0, 1), ('SfaB', 0, 2), ('CphA1', 1, 3),
                  ('McbA', 0, 5), ('PylC', 1, 6)]
    matrix = np.asarray([panels[p][r, columns] for _, p, r in selections])
    fig, ax = plt.subplots(figsize=(6.8, 3.8), layout='constrained')
    cmap = plt.get_cmap('RdBu_r').copy()
    cmap.set_bad('#eeeeee')
    available = matrix[np.isfinite(matrix)]
    vmin = -1 if np.any(available < -1e-6) else 0
    img = ax.imshow(np.ma.masked_invalid(matrix), cmap=cmap, vmin=vmin, vmax=1, aspect='auto')
    ax.set_title('Similarity to nylon-6,6 diad formation', fontsize=13, weight='bold', pad=12)
    ax.set_yticks(range(5), ['DdaG · AMP', 'SfaB · AMP', 'CphA1 · ADP', 'McbA · AMP', 'PylC · ADP'])
    ax.set_xticks(range(3), ['DRFP\nTanimoto', 'ECFP4 diff + DRFP\ncosine', 'rxnfp FT\ncosine'], fontsize=10)
    ax.tick_params(length=0)
    for spine in ax.spines.values(): spine.set_visible(False)
    for i in range(5):
        for j in range(3):
            value = matrix[i, j]
            ax.text(j, i, '—' if np.isnan(value) else f'{value:.3f}', ha='center', va='center', fontsize=12,
                    color='white' if np.isfinite(value) and (value < .23 or value > .77) else '#183348')
    fig.colorbar(img, ax=ax, fraction=.035, pad=.025, label='Similarity')
    fig.supxlabel('Cofactors retained; target ATP route matched per enzyme.  —: unspecified polymer.', fontsize=8)
    for suffix in ['png', 'svg', 'pdf']:
        fig.savefig(out / ('similarity_heatmap.' + suffix), dpi=220, bbox_inches='tight')
    plt.close(fig)
    matched = [dict(enzyme=name, target_query=['amp', 'adp'][p], embedding=methods[c][0],
                    metric=methods[c][1], similarity=None if np.isnan(matrix[i, j]) else float(matrix[i, j]))
               for i, (name, p, _) in enumerate(selections) for j, c in enumerate(columns)]
    with (out / 'matched_similarities.tsv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(matched[0]), delimiter='\t')
        writer.writeheader(); writer.writerows(matched)


def rank_heatmap(out, core, methods):
    """Compare within-column rankings without mixing embedding score scales."""
    indices = [1, 5, 6]  # Three complete, finite native amide transformations.
    values = np.round(core[indices], 3)
    ranks = np.full_like(values, np.nan)
    for j in range(values.shape[1]):
        finite = np.unique(values[:, j][np.isfinite(values[:, j])])
        for i in range(values.shape[0]):
            if np.isfinite(values[i, j]):
                ranks[i, j] = 1 + np.sum(finite > values[i, j])
    fig, ax = plt.subplots(figsize=(13.5, 2.8), layout='constrained')
    cmap = plt.get_cmap('YlGnBu_r').copy(); cmap.set_bad('#eeeeee')
    img = ax.imshow(np.ma.masked_invalid(ranks), cmap=cmap, vmin=1, vmax=3, aspect='auto')
    ax.set_yticks(range(3), [ROW_LABELS[i] for i in indices])
    ax.set_xticks(range(len(methods)), [m[2] for m in methods], fontsize=9)
    ax.tick_params(length=0)
    for spine in ax.spines.values(): spine.set_visible(False)
    for i in range(3):
        for j in range(len(methods)):
            value = ranks[i, j]
            ax.text(j, i, '—' if np.isnan(value) else str(int(value)), ha='center', va='center',
                    color='white' if value == 1 else '#183348', weight='bold' if value == 1 else 'normal')
    fig.colorbar(img, ax=ax, ticks=[1, 2, 3], fraction=.018, pad=.015, label='Rank (1 = closest)')
    ax.set_title('Organic reaction ranking within each method', loc='left', weight='bold', fontsize=13)
    fig.supxlabel('Ranks use scores rounded to three decimals; ties share rank. SfaB activation and variable-length CphA1 polymers are excluded.', fontsize=9)
    for suffix in ['png', 'svg', 'pdf']:
        fig.savefig(out / ('core_rank_heatmap.' + suffix), dpi=220, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=PROJECT / 'results/native_nylon_similarity')
    parser.add_argument('--rxnfp-python', type=Path, default=Path('/nfs/lambda_stor_01/homes/rzhu/miniforge3/envs/rxnfp/bin/python'))
    parser.add_argument('--skip-rxnfp', action='store_true')
    args = parser.parse_args()
    now = datetime.now(timezone.utc)
    out = args.output_root.resolve()
    out.mkdir(parents=True, exist_ok=True)
    # Flat reruns must not present obsolete learned-model files as current.
    for name in ['rxnfp_embeddings.npz', 'rxnfp_embeddings.npz.json', 'rxnfp_worker.log']:
        (out / name).unlink(missing_ok=True)
    records, acid, amine, product = build_inputs()
    index = {r['name']: i for i, r in enumerate(records)}
    (out / 'reaction_inputs.json').write_text(json.dumps(records, indent=2) + '\n')
    print('Encoding classical fingerprints for', len(records), 'balanced reactions', flush=True)
    arrays = fingerprint_arrays(records)
    errors = []
    if not args.skip_rxnfp:
        print('Encoding two local RXNFP models on CPU', flush=True)
        try:
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', TOKENIZERS_PARALLELISM='false')
            with (out / 'rxnfp_worker.log').open('w') as log:
                subprocess.run([str(args.rxnfp_python), str(PROJECT / 'native_similarity_rxnfp.py'),
                                str(out / 'reaction_inputs.json'), str(out / 'rxnfp_embeddings.npz')],
                               env=env, stdout=log, stderr=subprocess.STDOUT, timeout=180, check=True)
            info = json.loads((out / 'rxnfp_embeddings.npz.json').read_text())
            assert info['ordered_names'] == [r['name'] for r in records]
            with np.load(out / 'rxnfp_embeddings.npz') as learned:
                arrays.update({k: learned[k] for k in learned.files})
        except Exception as exc:
            errors.append('RXNFP unavailable: ' + str(exc))
    # Save all independently computed vectors, not corpus row lookups.
    np.savez(out / 'embeddings.npz', **arrays)
    panels, rows = [], []
    for query in ['amp', 'adp']:
        matrix = np.full((len(ROW_LABELS), len(METHODS)), np.nan)
        target = index['target_' + query]
        for row_idx, key in enumerate(['target'] + ROW_KEYS):
            name = 'target_' + query if key == 'target' else key + '_full'
            reason = 'Unspecified polymer has no fixed molecular graph' if key.startswith('CphA1') else ''
            for col, (embedding, metric, label) in enumerate(METHODS):
                value = np.nan
                missing = reason
                if not missing and embedding not in arrays:
                    missing = 'Model unavailable or deliberately skipped; see manifest/log'
                if not missing and name in index:
                    value = similarity(arrays[embedding][index[name]], arrays[embedding][target], metric)
                    assert np.isfinite(value), (name, embedding)
                matrix[row_idx, col] = value
                rows.append(dict(native_reaction=key, target_query=query, embedding=embedding,
                                 metric=metric, similarity=None if np.isnan(value) else value,
                                 missing_reason=missing,
                                 native_input_name=name if name in index else '',
                                 target_input_name='target_' + query))
        assert np.allclose(matrix[0, :7], 1)
        panels.append(matrix)
    with (out / 'similarities.tsv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter='\t')
        writer.writeheader(); writer.writerows(rows)
    (out / 'similarities.json').write_text(json.dumps(rows, indent=2, allow_nan=False) + '\n')
    heatmap(out, panels, METHODS)
    notes = [
        'Each column names its representation and metric. Scores across different columns have different meanings; compare native rankings within a column.',
        'ECFP4 sides is a binary reactant/product concatenation. Diff methods sum product count fingerprints minus reactant count fingerprints. The hybrid concatenates ECFP4 count difference and binary DRFP without reweighting, as in the existing search.',
        'DRFP: 2048 bits, radius 3. Morgan/ECFP4: radius 2, 2048 features per side. AtomPair: 2048 counts. MACCS: 167 keys. Tanimoto is applied only to binary vectors; signed count and learned embeddings use cosine.',
        'One convention throughout: canonical neutral structures, conventional amides, sorted species, no atom maps, no explicit H+. Every encoded equation is checked for atom and charge balance. These newly computed scores are not the historical pH-7 corpus retrieval scores.',
        'All reaction inputs retain their cofactors. The compact heatmap selects the AMP target for DdaG, SfaB and McbA, and the ADP target for PylC and both CphA1 activities. AMP labels denote the adenylation route; SfaB retains AMP in its acyl-AMP product rather than releasing free AMP. These are route-matched analogies, not comparisons against one identical target equation.',
        'Only DRFP/Tanimoto, the existing hybrid/cosine method and rxnfp FT/cosine are displayed. Complete scores against both target equations remain in similarities.tsv for audit, without duplicating enzymes in the primary visualization.',
        'CphA1 cyanophycin has unspecified repeat length; the R-group drawings are explanatory fragments, not complete molecules. Neither polymer step receives a fabricated fingerprint score.',
        'Binary DRFP uses symmetric fragment differences and is direction-invariant. Cosine similarities can be negative. Shared ATP/nucleotide changes can dominate full-reaction scores.',
        'The hand-weighted substrate heuristic is not included in this embedding heatmap. Similarity is not a probability of activity, a yield prediction, or experimental validation.',
    ]
    diagrams = sorted((PROJECT / 'figures/native_reactions_2d').glob('*/native_reactions.svg'))
    native_image = ''
    if diagrams:
        relative = os.path.relpath(diagrams[-1], out)
        native_image = '<h2>Native reactions</h2><img src="' + html.escape(relative) + '">'
    page = ('<!doctype html><meta charset="utf-8"><title>Native versus nylon reactions</title>'
            '<style>body{font:16px/1.5 system-ui;max-width:1500px;margin:25px auto;padding:0 15px}img{width:100%}li{margin:8px 0}</style>'
            '<img src="similarity_heatmap.svg">'
            '<p><a href="similarity_heatmap.png">PNG</a> · <a href="similarity_heatmap.pdf">PDF</a> · '
            '<a href="matched_similarities.tsv">Displayed scores TSV</a></p>'
            '<details><summary>Methods and interpretation</summary><ul>' +
            ''.join('<li>' + html.escape(n) + '</li>' for n in notes) + '</ul></details>')
    polymer = PROJECT / 'figures/nylon66_polymerization/nylon66_polymerization.svg'
    if polymer.exists():
        page += '<p><a href="' + html.escape(os.path.relpath(polymer, out)) + '">Polymerization target diagram</a></p>'
    (out / 'report.html').write_text(page)
    reference = PROJECT / 'data/native_reaction_references.json'
    inputs = [Path(__file__), PROJECT / 'native_similarity_rxnfp.py', PROJECT / 'draw_native_reactions.py',
              reference, PROJECT / 'amide_synthesis/reaction_search/encode_and_search.py',
              PROJECT / 'amide_synthesis/reaction_search/normalize.py']
    if polymer.exists(): inputs.append(polymer)
    # Pin source compound files used by load_species and the formula checks.
    from amide_synthesis.infrastructure.paths import DB
    needed_ids = {'cpd03642', 'cpd25003', 'cpd00002', 'cpd00008', 'cpd00018', 'cpd00012', 'cpd00009'}
    for file in sorted(DB.glob('compound_[0-9][0-9].tsv')):
        if any(r['id'] in needed_ids for r in csv.DictReader(file.open(), delimiter='\t')):
            inputs.append(file)
    manifest = dict(created_utc=now.isoformat(), methods=METHODS, notes=notes, errors=errors,
                    query_conventions=['amp', 'adp'], displayed_target_selection='ATP route matched per enzyme',
                    output_policy='current artifacts directly in output directory; no run subfolders', rdkit_version=rdBase.rdkitVersion,
                    drfp_version=version('drfp'), rxnfp_requested=not args.skip_rxnfp,
                    rxnfp_available='bert_ft' in arrays and 'bert_pretrained' in arrays,
                    input_sha256={str(p.resolve()): digest(p) for p in inputs},
                    output_sha256={p.name: digest(p) for p in out.iterdir() if p.is_file() and p.name != 'manifest.json'})
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('Saved:', out.resolve(), flush=True)
    for error in errors:
        print(error, flush=True)
    return 1 if errors else 0


if __name__ == '__main__':
    raise SystemExit(main())
