#!/usr/bin/env python3
"""Rank saved ModelSEED reactions against nylon-6,6, retaining cofactors.

Primary metric: DRFP Tanimoto. Rank the full representable corpus, then report
the top 100 within the existing balanced free-amine amidation screen. Highlight
five chemistry-led candidates, with supplemental hybrid and RXNFP cosines.
DRFP retrieves the top 100; explicit functional-group review selects the five.
No protein-count or gene-mapping requirement affects selection.
Run with palm Python, using two CPU threads and the installed RXNFP worker.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, rdBase

from amide_synthesis.infrastructure.paths import DB
from amide_synthesis.infrastructure.review import artifact
from amide_synthesis.infrastructure.freshness import state, status
from amide_synthesis.reaction_search import corpus
from amide_synthesis.reaction_search.encode_and_search import build_queries, featurise, load_species
from amide_synthesis.reaction_search.acylation import count_free_amines
from amide_synthesis.reaction_search.substrate_distance import CARBOXYLIC_ACID
from nylon_chemical_review import review as chemical_review, render as render_chemical_review
from draw_top5_native_reactions import generate as draw_shortlisted_native_reactions

PROJECT = Path(__file__).resolve().parent
QUERIES = ['atp_amp_ligase', 'atp_adp_ligase', 'coa_aminolysis', 'direct_condensation']
DESCRIPTIONS = {
    'rxn21314': dict(title='Pre-bisucaberin formation',
        why='Joins two aliphatic amino-acid-type building blocks through a new amide; retains a terminal free amine and carboxylate. Uses the ATP/AMP route.',
        difference='AB-monomer self-condensation, with pre-existing N-hydroxyamides and shorter succinyl/C5-diamine segments; not adipate plus HMDA.'),
    'rxn21296': dict(title='Desferrioxamine G formation',
        why='Extends three flexible aliphatic amino/carboxyl building blocks into a linear polyamide with free acid and amine ends.',
        difference='Forms two new amides in trimer assembly, versus one in the target diad; N-hydroxylated, shorter monomer segments. ATP coupling stoichiometry is reproduced from ModelSEED, not experimentally revalidated.'),
    'rxn21308': dict(title='Pre-putrebactin formation',
        why='Amide coupling of two bifunctional aliphatic monomers leaves both an acid and an amine available for further coupling.',
        difference='Uses N-hydroxy-N-succinylputrescine AB monomers with C4 diamine segments, rather than separate C6 diacid and diamine.'),
    'rxn21038': dict(title='Citrate–spermidine ligation',
        why='Couples a polycarboxylate to a free aliphatic polyamine, forming an amide while retaining additional acid and amine groups.',
        difference='Citrate is branched, hydroxylated and tricarboxylic; spermidine contains three amines and unequal spacers. The target uses straight-chain difunctional monomers.'),
    'rxn21301': dict(title='Desferrioxamine B formation',
        why='Builds a flexible aliphatic amide-linked chain from amino/carboxyl precursors via ATP-dependent condensation.',
        difference='Multi-building-block assembly with N-hydroxyamides and an acetyl-capped end; it does not retain the target diad’s free acid/amine pair.'),
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def canonical_reaction(smiles):
    return '>>'.join('.'.join(sorted(Chem.MolToSmiles(Chem.MolFromSmiles(s))
                                    for s in side.split('.'))) for side in smiles.split('>>'))


def target_route(smiles, species):
    left, right = [set(side.split('.')) for side in smiles.split('>>')]
    if species['ATP'] in left and species['AMP'] in right and species['PPI'] in right: return 0
    if species['ATP'] in left and species['ADP'] in right and species['PI'] in right: return 1
    if species['COA'] in right and species['COA'] not in left: return 2
    return 3


def organic_group(smiles, species):
    """Deduplicate ATP bookkeeping only; never used to calculate a score."""
    bookkeeping = {species[k] for k in ['ATP', 'AMP', 'ADP', 'PI', 'PPI']}
    key = '>>'.join('.'.join(sorted(s for s in side.split('.') if s not in bookkeeping))
                    for side in smiles.split('>>'))
    return hashlib.sha256(key.encode()).hexdigest()


def metadata():
    rows, hashes = {}, {}
    for path in sorted(DB.glob('reaction_[0-9][0-9].tsv')):
        hashes[str(path)] = digest(path)
        with path.open() as stream:
            for line, row in enumerate(csv.DictReader(stream, delimiter='\t'), 2):
                row['source_file'] = str(path)
                row['source_line'] = line
                rows[row['id']] = row
    return rows, hashes


def plot_scores(frame, path, overview=False, curated=False):
    columns = ['drfp_tanimoto', 'hybrid_cosine', 'rxnfp_cosine']
    values = frame[columns].to_numpy(float)
    plt.rcParams.update({'svg.fonttype': 'none', 'pdf.fonttype': 42})
    fig, ax = plt.subplots(figsize=(6.7, 3.8 if not overview else 19), layout='constrained')
    palette = plt.get_cmap('RdBu_r').copy(); palette.set_bad('#eeeeee')
    im = ax.imshow(np.ma.masked_invalid(values), vmin=0, vmax=1, cmap=palette, aspect='auto')
    labels = frame.rxn_id.tolist() if curated else [f'{i+1}. {rid}' for i, rid in enumerate(frame.rxn_id)]
    ax.set_yticks(range(len(frame)), labels, fontsize=10 if not overview else 7)
    ax.set_xticks(range(3), ['DRFP\nTanimoto', 'ECFP4 diff + DRFP\ncosine', 'rxnfp FT\ncosine'], fontsize=10)
    ax.tick_params(length=0)
    for spine in ax.spines.values(): spine.set_visible(False)
    if not overview:
        for i in range(len(frame)):
            for j in range(3):
                value = values[i, j]
                ax.text(j, i, '—' if not np.isfinite(value) else f'{value:.3f}', ha='center', va='center',
                        color='white' if np.isfinite(value) and (value < .23 or value > .77) else '#183348', fontsize=12)
    ax.set_title('Chemistry-led shortlist' if curated else 'Top 100 amide-forming reactions' if overview else 'DRFP retrieval shortlist',
                 fontsize=12, weight='bold', pad=10)
    fig.colorbar(im, ax=ax, fraction=.035, pad=.025, label='Similarity')
    fig.supxlabel('Chemistry-led selection; embedding scores are supporting evidence. Cofactors retained.' if curated else
                  'Ranked by DRFP Tanimoto; cofactors retained. Compare rankings within each method.', fontsize=8)
    for extension in ['png', 'svg', 'pdf']:
        fig.savefig(path.with_suffix('.' + extension), dpi=200, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=PROJECT / 'results/modelseed_nylon_top100')
    parser.add_argument('--rxnfp-python', type=Path, default=Path('/nfs/lambda_stor_01/homes/rzhu/miniforge3/envs/rxnfp/bin/python'))
    args = parser.parse_args()
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    freshness = {k: status(k) for k in state()['stages']}
    if freshness.get('table') != 'current': raise RuntimeError('Refresh candidate-screen inputs before ranking')
    print('Loading and hash-checking the saved ModelSEED corpus', flush=True)
    reactions, unit = corpus.load()
    source, db_hashes = metadata()
    query_table = build_queries().set_index('query').loc[QUERIES]
    saved_queries = pd.read_csv(PROJECT / 'results/queries.tsv', sep='\t').set_index('query')
    assert query_table.rxn_smiles.to_dict() == saved_queries.loc[QUERIES].rxn_smiles.to_dict(), 'Target structures drifted from saved corpus queries'
    vectors = featurise(query_table.rxn_smiles.tolist())
    bits = unit[:, 2048:] > 0
    tanimoto = np.stack([np.logical_and(bits, q[2048:] > 0).sum(1) /
                         np.logical_or(bits, q[2048:] > 0).sum(1) for q in vectors], axis=1)
    cosine = unit @ (vectors / np.linalg.norm(vectors, axis=1, keepdims=True)).T
    del bits, unit
    species = load_species()
    route = np.array([target_route(s, species) for s in reactions.rxn_smiles])
    reactions['target_query'] = [QUERIES[i] for i in route]
    reactions['drfp_tanimoto'] = tanimoto[np.arange(len(reactions)), route]
    reactions['hybrid_cosine'] = cosine[np.arange(len(reactions)), route]
    for i, query in enumerate(QUERIES):
        reactions['drfp_' + query] = tanimoto[:, i]
        reactions['hybrid_' + query] = cosine[:, i]
    screen_path = PROJECT / 'results/amide_forming_neighbours.tsv'
    screen = pd.read_csv(screen_path, sep='\t').drop_duplicates('rxn_id')
    eligible_ids = set(screen.rxn_id)
    assert len(eligible_ids) == 480
    reactions['amide_screen_eligible'] = reactions.rxn_id.isin(eligible_ids)
    for field in ['name', 'definition', 'aliases', 'source_file', 'source_line']:
        reactions[field] = [source[r].get(field, '') for r in reactions.rxn_id]
    reactions['name'] = reactions['name'].fillna('')
    reactions['curation_note'] = reactions.rxn_id.map({'rxn46045': 'Raw ModelSEED product label says fumaramoyl; the repaired molecular graph and aliases identify fumaroyl-DAP.'}).fillna('')
    reactions = reactions.sort_values(['drfp_tanimoto', 'hybrid_cosine', 'rxn_id'], ascending=[False, False, True]).reset_index(drop=True)
    reactions.insert(0, 'database_rank', np.arange(1, len(reactions) + 1))
    eligible = reactions[reactions.amide_screen_eligible].copy().reset_index(drop=True)
    assert len(eligible) == 480 and eligible.heavy_balanced.all() and eligible.proton_fixable.all()
    eligible.insert(0, 'amide_rank', np.arange(1, len(eligible) + 1))
    eligible['organic_transformation_group'] = [organic_group(s, species) for s in eligible.rxn_smiles]
    top100 = eligible.head(100).copy()
    top5 = eligible.drop_duplicates('organic_transformation_group').head(5).copy()
    print('Ranking complete: 33,246 representable reactions; 480 eligible amidations; encoding supplemental RXNFP for the top 100', flush=True)
    selected_ids = set(top100.rxn_id) | set(top5.rxn_id) | {'rxn46045', 'rxn23366', 'rxn60314'}
    input_rows = [dict(name='target:' + q, rxn_smiles=canonical_reaction(query_table.loc[q, 'rxn_smiles'])) for q in QUERIES]
    selected = reactions[reactions.rxn_id.isin(selected_ids)].copy()
    input_rows += [dict(name=r.rxn_id, rxn_smiles=canonical_reaction(r.rxn_smiles)) for r in selected.itertuples()]
    (out / 'rxnfp_inputs.json').write_text(json.dumps(input_rows, indent=2) + '\n')
    worker_output = out / 'rxnfp_embeddings.npz'
    worker_output.unlink(missing_ok=True)
    worker_output.with_name(worker_output.name + '.json').unlink(missing_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', TOKENIZERS_PARALLELISM='false')
    with (out / 'rxnfp_worker.log').open('w') as stream:
        subprocess.run([str(args.rxnfp_python), str(PROJECT / 'native_similarity_rxnfp.py'),
                        str(out / 'rxnfp_inputs.json'), str(worker_output), '--models', 'bert_ft', '--allow-missing'],
                       env=env, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=600)
    info = json.loads((out / 'rxnfp_embeddings.npz.json').read_text())
    assert info['ordered_names'] == [r['name'] for r in input_rows]
    names = {r['name']: i for i, r in enumerate(input_rows)}
    with np.load(worker_output) as data: learned = data['bert_ft']
    learned_scores = {}
    for r in selected.itertuples():
        a, b = learned[names[r.rxn_id]], learned[names['target:' + r.target_query]]
        learned_scores[r.rxn_id] = float(a @ b / np.linalg.norm(a) / np.linalg.norm(b)) if np.isfinite(a).all() and np.isfinite(b).all() else np.nan
    for frame in [reactions, eligible, top100, top5]:
        frame['rxnfp_cosine'] = frame.rxn_id.map(learned_scores)
    for key in ['display_title', 'why_similar', 'key_difference']:
        top5[key] = ''
    for i, r in top5.iterrows():
        annotation = DESCRIPTIONS.get(r.rxn_id)
        if not annotation: raise RuntimeError('Unreviewed top-five result: ' + r.rxn_id)
        top5.loc[i, 'display_title'] = annotation['title']
        top5.loc[i, 'why_similar'] = annotation['why']
        top5.loc[i, 'key_difference'] = annotation['difference']
        left, right = [[Chem.MolFromSmiles(s) for s in side.split('.')] for side in r.rxn_smiles.split('>>')]
        amide = Chem.MolFromSmarts('C(=O)N')
        top5.loc[i, 'new_amide_bonds'] = sum(len(m.GetSubstructMatches(amide)) for m in right) - sum(len(m.GetSubstructMatches(amide)) for m in left)
        organic_products = [m for m in right if m.HasSubstructMatch(amide)]
        top5.loc[i, 'product_free_amines'] = sum(count_free_amines(m) for m in organic_products)
        top5.loc[i, 'product_free_carboxyls'] = sum(len(m.GetSubstructMatches(CARBOXYLIC_ACID)) for m in organic_products)
    top5.insert(0, 'selection_rank', np.arange(1, 6))
    # Preserve the full equations and species; ATP-only grouping affects selection, not scoring.
    reactions.to_csv(out / 'all_modelseed_scores.tsv', sep='\t', index=False)
    reactions.head(100).to_csv(out / 'database_top100_unfiltered.tsv', sep='\t', index=False)
    eligible.to_csv(out / 'all_amide_scores.tsv', sep='\t', index=False)
    top100.to_csv(out / 'top100_reactions.tsv', sep='\t', index=False)
    top5.to_csv(out / 'drfp_top5_reactions.tsv', sep='\t', index=False)
    references = reactions[reactions.rxn_id.isin({'rxn46045', 'rxn23366', 'rxn60314'})].copy()
    references['amide_rank'] = references.rxn_id.map(eligible.set_index('rxn_id').amide_rank)
    references.to_csv(out / 'literature_reference_scores.tsv', sep='\t', index=False)
    target_export = query_table.reset_index()
    target_export.to_csv(out / 'target_queries.tsv', sep='\t', index=False)
    plot_scores(top5, out / 'drfp_top5_similarity_heatmap')
    plot_scores(top100, out / 'top100_similarity_heatmap', overview=True)
    esc = lambda x: html.escape(str(x))
    summary = '<tr><th>Rank</th><th>ModelSEED</th><th>Reaction</th><th>DRFP Tanimoto</th><th>Why similar</th><th>Main difference</th></tr>'
    for r in top5.itertuples():
        summary += '<tr>' + ''.join('<td>' + esc(x) + '</td>' for x in [r.selection_rank, r.rxn_id,
            r.display_title, f'{r.drfp_tanimoto:.3f}', r.why_similar, r.key_difference]) + '</tr>'
    table = '<tr><th>Rank</th><th>ModelSEED</th><th>DRFP</th><th>Hybrid cosine</th><th>rxnfp cosine</th><th>Target</th><th>Native reaction</th></tr>'
    for r in top100.itertuples():
        values = [r.amide_rank, r.rxn_id, f'{r.drfp_tanimoto:.3f}', f'{r.hybrid_cosine:.3f}',
                  '—' if not np.isfinite(r.rxnfp_cosine) else f'{r.rxnfp_cosine:.3f}', r.target_query, r.definition]
        table += '<tr>' + ''.join('<td>' + esc(v) + '</td>' for v in values) + '</tr>'
    dda = eligible[eligible.rxn_id == 'rxn46045'].iloc[0]
    notes = [
        'Primary ranking: DRFP Tanimoto, descending; ties use hybrid cosine and then ModelSEED ID. RXNFP is supplemental within the retrieved list, not a globally evaluated RXNFP ranking.',
        'Scope: 33,359 saved fully representable ModelSEED reactions, with 113 zero hybrid vectors excluded; 33,246 searchable reactions. The existing balanced free-amine acylation screen provides 480 eligible amidations. Missing/symbolic reactions are outside the numerical ranking.',
        'Full reactants, products, stoichiometric copies and cofactors are retained. ATP/AMP, ATP/ADP, acyl-CoA or direct-condensation targets are assigned from explicit native cofactor signatures, not whichever target maximizes a score. The remaining literal nylon monomers are C6 adipate/HMDA in every query.',
        'This ranking uses the saved repaired ModelSEED pH-7 convention with H+ stripped, not the neutral depiction convention used by the earlier five-enzyme standalone chart. Comparisons within this report use one consistent convention.',
        'The unfiltered database top 100 is retained separately. DRFP is direction-invariant: rxn00044 (nylon-6 dimer hydrolysis) scores 1.0 but is not an amide-forming candidate. No fingerprint score alone proves suitable synthetic chemistry.',
        'The DRFP-only five are retained as a retrieval baseline. The primary five-entry shortlist instead uses explicit functional-group categories reviewed within the retrieved 100. No weighted composite score or strict within-category ranking is claimed. Alternative ATP representations are grouped only for selection; every scored equation retains its cofactors.',
        'DRFP values are Tanimoto similarities, not percentages of molecular identity. Cosine values use different embeddings and are not probabilities, yields or activity estimates. A high score can be driven by shared nucleotide chemistry.',
        'The rxnfp FT model is the authors’ released bert_ft checkpoint. Inputs are canonicalized and species sorted. Reactions exceeding 512 tokens or containing unknown tokens are marked unscored rather than silently truncated.',
        'Names and definitions are reproduced from local ModelSEED. Structural explanations of the five selected entries are a chemical interpretation, not experimental validation. ModelSEED ATP-coupling stoichiometry and alternate mechanistic annotations may require curation.',
        'DdaG rxn46045 is ranked ' + str(int(dda.amide_rank)) + ' of 480 eligible amidations by DRFP Tanimoto (' + f'{dda.drfp_tanimoto:.3f}' + '); it is not forced into the top five.',
    ]
    page = ('<!doctype html><meta charset="utf-8"><title>ModelSEED reactions similar to nylon-6,6</title>'
        '<style>body{font:16px/1.5 system-ui;max-width:1350px;margin:25px auto;padding:0 15px}img{max-width:850px;width:100%}'
        'table{border-collapse:collapse;width:100%}td,th{padding:9px;border-bottom:1px solid #ddd;text-align:left;vertical-align:top}'
        'th{background:#eee}input{padding:9px;font:inherit;margin:15px 0;width:95%}</style>'
        '<h1>ModelSEED reactions similar to nylon-6,6 diad formation</h1>'
        '<p>Top 100 of 480 screened amide-forming reactions. Ranked by <b>DRFP Tanimoto</b>; cofactors retained.</p>'
        '<img src="drfp_top5_similarity_heatmap.svg"><table>' + summary + '</table>'
        '<p><a href="top100_reactions.tsv">Top 100 TSV</a> · <a href="drfp_top5_reactions.tsv">DRFP-only five reactions TSV</a> · '
        '<a href="drfp_top5_similarity_heatmap.pdf">Figure PDF</a> · <a href="database_top100_unfiltered.tsv">Unfiltered database top 100</a></p>'
        '<details><summary>Methods and limitations</summary><ul>' + ''.join('<li>' + esc(n) + '</li>' for n in notes) + '</ul></details>'
        '<details><summary>Browse the full top 100</summary><input id="search" placeholder="Filter by reaction ID or chemistry">'
        '<table id="ranking">' + table + '</table><p><a href="top100_similarity_heatmap.svg">All 100 rows as a heatmap</a></p></details>'
        '<script>document.querySelector("#search").oninput=e=>{let q=e.target.value.toLowerCase();document.querySelectorAll("#ranking tr").forEach((r,i)=>{if(i)r.hidden=!r.textContent.toLowerCase().includes(q)})}</script>')
    (out / 'drfp_retrieval_report.html').write_text(page)
    reviewed, curated = chemical_review(top100)
    reviewed.to_csv(out / 'chemical_review.tsv', sep='\t', index=False)
    curated.to_csv(out / 'top5_reactions.tsv', sep='\t', index=False)
    plot_scores(curated, out / 'top5_similarity_heatmap', curated=True)
    draw_shortlisted_native_reactions(out / 'top5_reactions.tsv', out)
    render_chemical_review(out, reviewed, curated)
    resources = Path('/tmp/modelseed_top100_resources.json')
    if resources.exists(): shutil.copy2(resources, out / 'resources.json')
    input_paths = [Path(__file__), PROJECT / 'native_similarity_rxnfp.py', corpus.CORPUS,
        corpus.EMBEDDING, PROJECT / 'results/corpus_manifest.json', screen_path,
        PROJECT / 'results/queries.tsv', PROJECT / 'mapping_policy.json',
        PROJECT / 'amide_synthesis/reaction_search/corpus.py',
        PROJECT / 'amide_synthesis/reaction_search/encode_and_search.py',
        PROJECT / 'amide_synthesis/reaction_search/amide_enzymes.py',
        PROJECT / 'nylon_chemical_review.py',
        PROJECT / 'draw_top5_native_reactions.py',
        PROJECT / 'draw_native_reactions.py',
        PROJECT / 'amide_synthesis/reaction_search/acylation.py',
        PROJECT / 'amide_synthesis/reaction_search/substrate_distance.py']
    manifest = dict(created_utc=now, database_scope='saved local ModelSEED snapshot, not a new live database retrieval',
        corpus=corpus.manifest(), searchable_reactions=len(reactions), eligible_amidations=len(eligible),
        primary_metric='DRFP Tanimoto', tie_breakers=['hybrid cosine descending', 'ModelSEED ID ascending'],
        retrieval_metric_role='candidate retrieval only',
        selection_method='Explicit qualitative reacting-group categories within the DRFP top 100; no composite score',
        unique_top5=curated.rxn_id.tolist(), top5_raw_ranks=curated.amide_rank.tolist(),
        drfp_retrieval_top5=top5.rxn_id.tolist(), chemical_review_categories=reviewed.chemical_tier.value_counts().to_dict(),
        derived_artifacts={'top5_native_reactions': 'top5_native_reactions_manifest.json'},
        notes=notes, target_queries=QUERIES, cofactors_retained=True, rdkit_version=rdBase.rdkitVersion,
        rxnfp_skipped_count=len(info['models']['bert_ft']['skipped_indices']),
        pipeline_freshness_before=freshness,
        input_sha256={**{str(p.resolve()): digest(p) for p in input_paths}, **db_hashes},
        output_sha256={p.name: digest(p) for p in out.iterdir() if p.is_file() and p.name != 'manifest.json'})
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('Saved:', out, flush=True)
    print(curated[['chemical_tier', 'amide_rank', 'rxn_id', 'drfp_tanimoto', 'hybrid_cosine', 'rxnfp_cosine']].to_string(index=False), flush=True)
    print('RXNFP skipped inputs:', manifest['rxnfp_skipped_count'], flush=True)


if __name__ == '__main__': main()
