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
    'rxn22343': ('D-Lysine addition (MurE)', 'UDP-MurNAc-L-Ala-D-Glu', 'D-Lysine', 'UDP-MurNAc-L-Ala-D-Glu-D-Lys'),
    'rxn42395': ('Staphyloferrin B (SbnC)', '2-Oxoglutarate', 'Diamino citryl intermediate', 'Staphyloferrin B'),
    'rxn09897': ('Glycylglycine formation', 'Glycine', 'Glycine', 'Glycylglycine'),
    'rxn45007': ('Citryl-DAP ligation (SbnF)', 'Citryl-aminoethylamide intermediate', 'L-DAP', 'Diamino citryl intermediate'),
    'rxn21036': ('Petrobactin intermediate (AsbB)', 'Dihydroxybenzoyl-citryl-spermidine', 'Spermidine', 'Bis-spermidine citryl intermediate'),
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
    if len(organic_right) != 1:
        raise ValueError('Expected a single organic amide product: ' + row['rxn_id'])
    parts = Counter(neutral(s) for s in organic_left)
    if len(parts) == 2:
        acid = neutral(row['reacting_acid_smiles'])
        amine = neutral(row['reacting_acceptor_smiles'])
        if set(parts) != {acid, amine}:
            raise ValueError('Reacting-partner labels cannot be matched to input structures')
        parts = {acid:parts[acid], amine:parts[amine]}
    if len(parts) > 2:
        raise ValueError('More than two distinct organic partners; use the full saved equation')
    product = neutral(organic_right[0])
    # Reconstruct neutral mass balance for the whole organic transformation.
    from nylon_chemical_review import AMIDE
    count_amides = lambda smi: len(molecule(smi).GetSubstructMatches(AMIDE))
    delta = count_amides(product) - sum(count_amides(reactant) for reactant in map(neutral, organic_left))
    total = Counter()
    for reactant, count in parts.items():
        for element, value in composition(molecule(reactant)).items():
            total[element] += count*value
    if total != composition(molecule(product)) + Counter({k: delta*v for k,v in composition(molecule('O')).items()}):
        raise ValueError('Organic condensation is not balanced: ' + row['rxn_id'])
    turnover = []
    for key, label in [('AMP','AMP'),('ADP','ADP'),('PPI','PPi'),('PI','Pi')]:
        count = right.count(species[key])
        if count: turnover.append((str(count)+' ' if count != 1 else '') + label)
    labels = LABELS.get(row['rxn_id'])
    reactants = []
    for i,(smiles,count) in enumerate(parts.items()):
        label = labels[i+1] if labels else 'Substrate ' + str(i+1)
        reactants.append(((str(count)+' × ' if count != 1 else '') + label, smiles))
    return dict(rxn_id=row['rxn_id'], title=labels[0] if labels else row.get('display_title',row['rxn_id']),
                reactants=reactants, product=(labels[3] if labels else 'Amide product',product),
                turnover=[' + '.join(turnover)],
                atp_label=(str(left.count(species['ATP']))+' ' if left.count(species['ATP']) != 1 else '') + 'ATP',
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
    if len(scheme['reactants']) > 1: canvas.text(325, 147, '+', 29, anchor='middle')
    canvas.line(665, 164, 775, 164)
    canvas.line(761, 155, 775, 164); canvas.line(761, 173, 775, 164)
    canvas.text(720, 126, scheme.get('atp_label','ATP'), 20, TEAL, anchor='middle')
    for i, line in enumerate(scheme['turnover']):
        canvas.text(720, 181 + i * 23, line, 17 if len(scheme['turnover']) > 1 else 18, TEAL, anchor='middle')
    product = molecule(scheme['product'][1])
    canvas.mol(product, 800, 65, 580, 190)
    canvas.text(1090, 266, scheme['product'][0], 19, anchor='middle')
    return canvas


def generate(table, out):
    table, out = Path(table), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(table, sep='\t')
    if not len(frame) or frame.rxn_id.duplicated().any():
        raise ValueError('Expected distinct shortlisted reactions')
    species = load_species()
    schemes = [prepare(row, species) for row in frame.to_dict('records')]
    target = dict(title='Nylon-6,6 diad target', rxn_id='',
        reactants=[('Adipate', 'O=C(O)CCCCC(=O)O'), ('HMDA', 'NCCCCCCN')],
        product=('Nylon-6,6 diad', 'O=C(O)CCCCC(=O)NCCCCCCN'),
        turnover=['AMP + PPi', 'or ADP + Pi'])
    rows = (len(frame) + 2) // 2
    composite = Canvas(WIDTH * 2 + 40, HEIGHT * rows + 130)
    composite.text(20, 15, 'Nylon target and shortlisted native reactions', 34, bold=True)
    for i, scheme in enumerate([target] + schemes):
        composite.paste(draw_panel(scheme), (i % 2) * (WIDTH + 40), 80 + (i // 2) * HEIGHT)
    composite.text(20, HEIGHT * rows + 90, 'Cofactor turnover and stoichiometric copies shown explicitly. Neutral structures; no atom-mapped bond assignment.', 22, MUTED)
    stem = 'top' + str(len(frame)) + '_native_reactions'
    composite.save(out / stem, pdf=True)
    data = dict(target=target, native_reactions=schemes,
                depiction_convention='Conventional neutral forms; raw full and proton-balanced native equations retained. ATP and its products are displayed as text, not deleted.',
                target_scope='Two hypothetical ATP-coupled framings of the diad target; each native panel shows its actual route.')
    (out / (stem + '.json')).write_text(json.dumps(data, indent=2) + '\n')
    files = [stem + '.' + ext for ext in ['png', 'svg', 'pdf', 'json']]
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), rdkit_version=rdBase.rdkitVersion,
        rxn_ids=frame.rxn_id.tolist(), full_native_equations_validated=True,
        neutral_organic_condensations_validated=True,
        input_sha256={str(p.resolve()): digest(p) for p in [table, Path(__file__), PROJECT / 'draw_native_reactions.py']},
        output_sha256={name: digest(out / name) for name in files})
    (out / (stem + '_manifest.json')).write_text(json.dumps(manifest, indent=2) + '\n')
    return files + [stem + '_manifest.json']


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
