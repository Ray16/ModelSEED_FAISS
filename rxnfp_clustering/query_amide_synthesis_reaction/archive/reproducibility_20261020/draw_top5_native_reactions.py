#!/usr/bin/env python3
"""Draw the current chemistry-led five ModelSEED reactions and nylon target.

Read the actual shortlisted reaction SMILES; depict neutral conventional
structures while retaining ATP turnover as explicit arrow labels. Save one
compact six-panel PNG/SVG/PDF directly beside the top-five table.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import textwrap

import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem.MolStandardize import rdMolStandardize

from draw_native_reactions import Canvas, composition, molecule, TEAL, MUTED
from amide_synthesis.reaction_search.acylation import neutral_key
from amide_synthesis.reaction_search.encode_and_search import load_species, balance_with_protons

PROJECT = Path(__file__).resolve().parent
WIDTH, HEIGHT = 1400, 330
LABELS = {
    'rxn05123': ('Glutamate–putrescine', 'L-Glutamate', 'Putrescine', 'γ-Glutamylputrescine'),
    'rxn46045': ('DdaG', 'Fumarate', 'L-DAP', 'N³-Fumaroyl-L-DAP'),
    'rxn21038': ('Citrate–spermidine', 'Citrate', 'Spermidine', 'Citrylspermidine'),
    'rxn42510': ('Citrate–D-ornithine', 'Citrate', 'D-Ornithine', 'Citryl-D-ornithine'),
    'rxn47618': ('Citrate–L-DAP', 'Citrate', 'L-DAP', 'N³-Citryl-L-DAP'),
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def neutral(smiles):
    return Chem.MolToSmiles(rdMolStandardize.Uncharger().uncharge(Chem.MolFromSmiles(smiles)))


def prepare(row, species):
    left, right = [side.split('.') for side in row['rxn_smiles'].split('>>')]
    # Validate the full native equation under the corpus proton convention.
    balanced_left, balanced_right, _ = balance_with_protons(row['rxn_id'], left, right)
    cofactor_names = ['ATP', 'AMP', 'ADP', 'PPI', 'PI', 'WATER']
    bookkeeping = {species[name] for name in cofactor_names}
    organic_left = [s for s in left if s not in bookkeeping]
    organic_right = [s for s in right if s not in bookkeeping]
    if len(organic_left) != 2 or len(organic_right) != 1:
        raise ValueError('Expected a two-substrate, single-amide shortlist entry: ' + row['rxn_id'])
    by_key = {neutral_key(Chem.MolFromSmiles(s)): s for s in organic_left}
    acid_key = neutral_key(Chem.MolFromSmiles(row['reacting_acid_smiles']))
    amine_key = neutral_key(Chem.MolFromSmiles(row['reacting_acceptor_smiles']))
    acid, amine, product = neutral(by_key[acid_key]), neutral(by_key[amine_key]), neutral(organic_right[0])
    if composition(molecule(acid)) + composition(molecule(amine)) != composition(molecule(product)) + composition(molecule('O')):
        raise ValueError('Organic condensation is not balanced: ' + row['rxn_id'])
    if row['target_query'] == 'atp_adp_ligase':
        expected = ['ADP', 'PI']; turnover = ['ADP + Pi']
    elif row['target_query'] == 'atp_amp_ligase':
        expected = ['AMP', 'PPI']; turnover = ['AMP + PPi']
    else:
        raise ValueError('Unexpected native cofactor route: ' + row['target_query'])
    assert Counter(s for s in left if s in bookkeeping) == Counter([species['ATP']])
    assert Counter(s for s in right if s in bookkeeping) == Counter(species[s] for s in expected)
    matches = molecule(product).GetSubstructMatches(Chem.MolFromSmarts('C(=O)N'))
    assert len(matches) == 1
    return dict(rxn_id=row['rxn_id'], title=LABELS[row['rxn_id']][0],
                reactants=[(LABELS[row['rxn_id']][1], acid), (LABELS[row['rxn_id']][2], amine)],
                product=(LABELS[row['rxn_id']][3], product), turnover=turnover,
                native_reaction_smiles=row['rxn_smiles'],
                proton_balanced_native_reaction_smiles='.'.join(balanced_left) + '>>' + '.'.join(balanced_right))


def draw_panel(scheme):
    canvas = Canvas(WIDTH, HEIGHT)
    canvas.text(20, 10, scheme['title'], 28, bold=True)
    canvas.text(1060, 16, scheme['rxn_id'], 20, MUTED, anchor='middle')
    for (label, smiles), x in zip(scheme['reactants'], [15, 340]):
        canvas.mol(molecule(smiles), x, 65, 295, 190)
        for n, line in enumerate(textwrap.wrap(label, 26)):
            canvas.text(x + 147, 266 + n * 23, line, 19, anchor='middle')
    canvas.text(325, 147, '+', 29, anchor='middle')
    canvas.line(665, 164, 775, 164)
    canvas.line(761, 155, 775, 164); canvas.line(761, 173, 775, 164)
    canvas.text(720, 126, 'ATP', 20, TEAL, anchor='middle')
    for i, line in enumerate(scheme['turnover']):
        canvas.text(720, 181 + i * 23, line, 17 if len(scheme['turnover']) > 1 else 18, TEAL, anchor='middle')
    product = molecule(scheme['product'][1])
    carbon, oxygen, nitrogen = product.GetSubstructMatch(Chem.MolFromSmarts('C(=O)N'))
    bond = product.GetBondBetweenAtoms(carbon, nitrogen).GetIdx()
    canvas.mol(product, 800, 65, 580, 190, bond)
    canvas.text(1090, 266, scheme['product'][0], 19, anchor='middle')
    return canvas


def generate(table, out):
    table, out = Path(table), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(table, sep='\t')
    if len(frame) != 5 or set(frame.rxn_id) != set(LABELS):
        raise ValueError('Review labels and chemistry before drawing a changed five-reaction shortlist')
    species = load_species()
    schemes = [prepare(row, species) for row in frame.to_dict('records')]
    target = dict(title='Nylon-6,6 diad target', rxn_id='',
        reactants=[('Adipate', 'O=C(O)CCCCC(=O)O'), ('HMDA', 'NCCCCCCN')],
        product=('Nylon-6,6 diad', 'O=C(O)CCCCC(=O)NCCCCCCN'),
        turnover=['AMP + PPi', 'or ADP + Pi'])
    composite = Canvas(WIDTH * 2 + 40, HEIGHT * 3 + 130)
    composite.text(20, 15, 'Nylon target and shortlisted native reactions', 34, bold=True)
    for i, scheme in enumerate([target] + schemes):
        composite.paste(draw_panel(scheme), (i % 2) * (WIDTH + 40), 80 + (i // 2) * HEIGHT)
    composite.text(20, 1080, 'Turquoise: bond formed. Cofactor turnover shown at each arrow. Neutral forms drawn.', 22, MUTED)
    composite.save(out / 'top5_native_reactions', pdf=True)
    data = dict(target=target, native_reactions=schemes,
                depiction_convention='Conventional neutral forms; raw full and proton-balanced native equations retained. ATP and its products are displayed as text, not deleted.',
                target_scope='Two hypothetical ATP-coupled framings of the diad target; each native panel shows its actual route.')
    (out / 'top5_native_reactions.json').write_text(json.dumps(data, indent=2) + '\n')
    files = ['top5_native_reactions.' + ext for ext in ['png', 'svg', 'pdf', 'json']]
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), rdkit_version=rdBase.rdkitVersion,
        rxn_ids=frame.rxn_id.tolist(), full_native_equations_validated=True,
        neutral_organic_condensations_validated=True,
        input_sha256={str(p.resolve()): digest(p) for p in [table, Path(__file__), PROJECT / 'draw_native_reactions.py']},
        output_sha256={name: digest(out / name) for name in files})
    (out / 'top5_native_reactions_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return files + ['top5_native_reactions_manifest.json']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--table', type=Path, default=PROJECT / 'results/modelseed_nylon_top100/top5_reactions.tsv')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    out = args.output if args.output else args.table.parent
    files = generate(args.table, out)
    # Keep the containing current report's artifact registry consistent on a standalone rerun.
    parent = out / 'manifest.json'
    if parent.exists():
        m = json.loads(parent.read_text())
        m.setdefault('derived_artifacts', {})['top5_native_reactions'] = 'top5_native_reactions_manifest.json'
        m['output_sha256'].update({name: digest(out / name) for name in files})
        parent.write_text(json.dumps(m, indent=2) + '\n')
    print((out / 'top5_native_reactions.png').resolve())


if __name__ == '__main__': main()
