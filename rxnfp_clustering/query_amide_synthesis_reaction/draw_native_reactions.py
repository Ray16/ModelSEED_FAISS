#!/usr/bin/env python3
"""Draw literature-native reactions with RDKit; run with the palm Python.

Outputs a new timestamped directory containing a combined PNG/SVG/PDF,
individual PNG/SVG panels, exact depicted SMILES and a provenance manifest.
Neutral conventional structures are used for legibility, not as a pH model.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import html
import io
import json
from pathlib import Path
import textwrap
import xml.etree.ElementTree as ET

from PIL import Image, ImageDraw, ImageFont
from rdkit import Chem, rdBase
from rdkit.Chem import rdDepictor, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D

PROJECT = Path(__file__).resolve().parent
W, PANEL_H = 1400, 330
INK, MUTED, TEAL = '#183348', '#526574', '#007f86'
HIGHLIGHT = (0.15, 0.78, 0.70)
NS = 'http://www.w3.org/2000/svg'
ET.register_namespace('', NS)

# Conventional neutral amide tautomers of the ModelSEED structures. These
# explicit depictions avoid drawing ModelSEED's charged imidate tautomers.
SCHEMES = [
    dict(key='DdaG', enzyme='DdaG', title='DdaG — fumaroyl-DAP formation', ids='rxn46045',
         reactants=[('Fumarate', 'O=C(O)/C=C/C(=O)O'),
                    ('L-2,3-Diaminopropionate', 'N[C@@H](CN)C(=O)O')],
         product=('N³-Fumaroyl-L-2,3-diaminopropionate',
                  'N[C@@H](CNC(=O)/C=C/C(=O)O)C(=O)O'),
         cofactor='ATP → AMP + PPi', note='Dapdiamide biosynthesis; acylation of the β-amino group.',
         source_compounds=['cpd00106', 'cpd02150', 'cpd26036']),
    dict(key='SfaB', enzyme='SfaB', title='SfaB — native acid adenylation', ids='rxn60314',
         reactants=[('(3R)-3-Isocyanobutanoate', '[C-]#[N+][C@H](C)CC(=O)O')],
         product=('3-Isocyanobutanoyl-AMP (acyl adenylate)',
                  '[C-]#[N+][C@H](C)CC(=O)OP(=O)(O)OC[C@H]1O[C@@H](n2cnc3c(N)ncnc32)[C@H](O)[C@@H]1O'),
         cofactor='ATP → PPi', note='AMP remains in the product. Native activation forms an O–P bond.',
         source_compounds=['cpd46599', 'cpd48231']),
    dict(key='CphA1_Asp', enzyme='SuCphA1', title='CphA1 — step 1: Asp addition to the backbone',
         ids='rxn48822 · RHEA:13277',
         reactants=[('Cyanophycin chain terminus', '*C(=O)O'),
                    ('L-Aspartate', 'N[C@@H](CC(=O)O)C(=O)O')],
         product=('Cyanophycin–Asp intermediate', '*C(=O)N[C@@H](CC(=O)O)C(=O)O'),
         cofactor='ATP → ADP + Pi', note='R denotes the existing cyanophycin chain, up to its terminal carbonyl.',
         source_compounds=['cpd00041']),
    dict(key='CphA1_Arg', enzyme='SuCphA1', title='CphA1 — step 2: Arg attachment to the Asp side chain',
         ids='rxn49577 · RHEA:23888',
         reactants=[('Cyanophycin–Asp intermediate', '*C(=O)N[C@@H](CC(=O)O)C(=O)O'),
                    ('L-Arginine', 'N[C@@H](CCCNC(=N)N)C(=O)O')],
         product=('Cyanophycin extended by one Asp–Arg unit',
                  '*C(=O)N[C@@H](CC(=O)N[C@@H](CCCNC(=N)N)C(=O)O)C(=O)O'),
         cofactor='ATP → ADP + Pi', note='The new bond joins the Asp side-chain carboxyl to the Arg α-amino group.',
         source_compounds=['cpd00051']),
    dict(key='McbA', enzyme='McbA', title='McbA — marinacarboline amide formation', ids='Exact ModelSEED entry not located',
         reactants=[('1-Acetyl-β-carboline-3-carboxylate', 'CC(=O)c1nc(C(=O)O)cc2c1[nH]c1ccccc12'),
                    ('2-Phenylethylamine', 'NCCc1ccccc1')],
         product=('Corresponding marinacarboline amide',
                  'CC(=O)c1nc(C(=O)NCCc2ccccc2)cc2c1[nH]c1ccccc12'),
         cofactor='ATP → AMP + PPi', note='Representative native pairing; related amines yield other marinacarbolines.',
         source_compounds=[], structure_source='https://www.rcsb.org/ligand/EQ2',
         reference_acid_smiles='CC(=O)c1nc(cc2c1[nH]c3ccccc23)C(O)=O'),
    dict(key='PylC', enzyme='PylC', title='PylC — methylornithine–lysine ligation', ids='rxn23366',
         reactants=[('(3R)-3-Methyl-D-ornithine', 'C[C@H](CCN)[C@@H](N)C(=O)O'),
                    ('L-Lysine', 'NCCCC[C@H](N)C(=O)O')],
         product=('(3R)-3-Methyl-D-ornithyl-N⁶-L-lysine',
                  'C[C@H](CCN)[C@@H](N)C(=O)NCCCC[C@H](N)C(=O)O'),
         cofactor='ATP → ADP + Pi', note='Pyrrolysine biosynthesis; acylation of the lysine ε-amino group.',
         source_compounds=['cpd24167', 'cpd00039', 'cpd24174']),
]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def molecule(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f'Invalid depiction SMILES: {smiles}')
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            atom.SetProp('atomLabel', 'R')
    rdDepictor.Compute2DCoords(mol)
    return mol


def composition(mol):
    return Counter(a.GetAtomicNum() for a in Chem.AddHs(mol).GetAtoms())


def validate(scheme):
    """Check organic condensation balance, stereo and new-bond selection."""
    reactants = [molecule(s) for _, s in scheme['reactants']]
    product = molecule(scheme['product'][1])
    if scheme['key'] == 'SfaB':
        # Acid + AMP -> acyl-AMP + water; ATP -> PPi is shown at the arrow.
        reactants.append(molecule('Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)(O)O)[C@@H](O)[C@H]1O'))
        match = product.GetSubstructMatch(Chem.MolFromSmarts('C(=O)OP'))
        bond = product.GetBondBetweenAtoms(match[2], match[3])
    else:
        matches = product.GetSubstructMatches(Chem.MolFromSmarts('C(=O)N'))
        match = matches[-1]
        bond = product.GetBondBetweenAtoms(match[0], match[2])
        query = Chem.MolFromSmarts('C(=O)N')
        assert len(matches) == sum(len(m.GetSubstructMatches(query)) for m in reactants) + 1
    left = sum((composition(m) for m in reactants), Counter())
    right = composition(product) + composition(molecule('O'))
    assert left == right, f'Organic condensation imbalance: {scheme["key"]}'
    if scheme['key'] == 'McbA':
        assert Chem.MolToSmiles(reactants[0]) == Chem.MolToSmiles(molecule(scheme['reference_acid_smiles']))
    if scheme['key'] == 'PylC':
        assert [c for _, c in Chem.FindMolChiralCenters(reactants[0])] == ['R', 'R']
        assert [c for _, c in Chem.FindMolChiralCenters(reactants[1])] == ['S']
    if scheme['key'] == 'SfaB':
        assert Chem.FindMolChiralCenters(reactants[0])[0][1] == 'R'
    return product, bond.GetIdx()


class Canvas:
    def __init__(self, width, height):
        self.width, self.height = width, height
        self.image = Image.new('RGB', (width, height), 'white')
        self.draw = ImageDraw.Draw(self.image)
        self.svg = ET.Element(f'{{{NS}}}svg', width=str(width), height=str(height),
                              viewBox=f'0 0 {width} {height}')
        ET.SubElement(self.svg, f'{{{NS}}}rect', width='100%', height='100%', fill='white')

    def text(self, x, y, value, size=24, color=INK, bold=False, anchor='start'):
        font = ImageFont.truetype('DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf', size)
        self.draw.text((x, y), value, font=font, fill=color, anchor='mt' if anchor == 'middle' else 'lt')
        node = ET.SubElement(self.svg, f'{{{NS}}}text', x=str(x), y=str(y + size),
                             fill=color, **{'font-family': 'DejaVu Sans,sans-serif',
                             'font-size': str(size), 'text-anchor': anchor,
                             'font-weight': 'bold' if bold else 'normal'})
        node.text = value

    def line(self, x1, y1, x2, y2, color=INK, width=3):
        self.draw.line((x1, y1, x2, y2), fill=color, width=width)
        ET.SubElement(self.svg, f'{{{NS}}}line', x1=str(x1), y1=str(y1), x2=str(x2), y2=str(y2),
                      stroke=color, **{'stroke-width': str(width)})

    def mol(self, mol, x, y, width, height, highlight=None):
        kwargs = {}
        if highlight is not None:
            kwargs = dict(highlightAtoms=[], highlightBonds=[highlight],
                          highlightBondColors={highlight: HIGHLIGHT})
        for kind in ('png', 'svg'):
            drawer = (rdMolDraw2D.MolDraw2DCairo if kind == 'png' else rdMolDraw2D.MolDraw2DSVG)(width, height)
            opts = drawer.drawOptions()
            opts.bondLineWidth = 2.3
            opts.minFontSize = 18
            opts.maxFontSize = 25
            opts.padding = 0.06
            opts.highlightBondWidthMultiplier = 5
            drawer.DrawMolecule(mol, **kwargs)
            drawer.FinishDrawing()
            if kind == 'png':
                tile = Image.open(io.BytesIO(drawer.GetDrawingText())).convert('RGB')
                self.image.paste(tile, (x, y))
            else:
                tile = ET.fromstring(drawer.GetDrawingText())
                tile.set('x', str(x)); tile.set('y', str(y))
                self.svg.append(tile)

    def save(self, path, pdf=False):
        self.image.save(path.with_suffix('.png'), dpi=(180, 180))
        ET.ElementTree(self.svg).write(path.with_suffix('.svg'), encoding='utf-8', xml_declaration=True)
        if pdf:
            self.image.save(path.with_suffix('.pdf'), 'PDF', resolution=180)

    def paste(self, other, x, y):
        self.image.paste(other.image, (x, y))
        other.svg.set('x', str(x)); other.svg.set('y', str(y))
        self.svg.append(other.svg)


def panel(canvas, scheme, reference, top):
    titles = {'CphA1_Asp': 'CphA1 · Asp addition', 'CphA1_Arg': 'CphA1 · Arg addition'}
    title = titles.get(scheme['key'], scheme['enzyme'])
    canvas.text(20, top + 10, title, 28, bold=True)
    identifier = scheme['ids'].split(' · ')[0] if scheme['key'] != 'McbA' else 'ModelSEED: not located'
    canvas.text(1060, top + 16, identifier, 20, MUTED, anchor='middle')
    product, highlighted_bond = validate(scheme)
    if len(scheme['reactants']) == 1:
        positions = [(130, 430)]
    else:
        positions = [(15, 295), (340, 295)]
        canvas.text(325, top + 147, '+', 29, anchor='middle')
    short_labels = {
        'DdaG': ['Fumarate', 'L-DAP', 'N³-Fumaroyl-L-DAP'],
        'SfaB': ['(3R)-3-Isocyanobutanoate', 'Acyl-AMP'],
        'CphA1_Asp': ['Cyanophycin terminus', 'L-Asp', 'Cyanophycin–Asp'],
        'CphA1_Arg': ['Cyanophycin–Asp', 'L-Arg', 'Extended cyanophycin'],
        'McbA': ['β-Carboline acid', '2-Phenylethylamine', 'Marinacarboline amide'],
        'PylC': ['(3R)-3-Methyl-D-ornithine', 'L-Lys', 'Methylornithyl-N⁶-L-Lys'],
    }[scheme['key']]
    for i, ((label, smiles), (x, width)) in enumerate(zip(scheme['reactants'], positions)):
        canvas.mol(molecule(smiles), x, top + 65, width, 190)
        for n, line in enumerate(textwrap.wrap(short_labels[i], 26)):
            canvas.text(x + width // 2, top + 266 + n * 23, line, 19, anchor='middle')
    canvas.line(665, top + 164, 775, top + 164)
    canvas.line(761, top + 155, 775, top + 164)
    canvas.line(761, top + 173, 775, top + 164)
    canvas.text(720, top + 126, 'ATP', 20, TEAL, anchor='middle')
    canvas.text(720, top + 181, scheme['cofactor'].split(' → ')[1], 18, TEAL, anchor='middle')
    canvas.mol(product, 800, top + 65, 580, 190, highlighted_bond)
    canvas.text(1090, top + 266, short_labels[-1], 19, anchor='middle')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=PROJECT / 'figures/native_reactions_2d')
    args = parser.parse_args()
    now = datetime.now(timezone.utc)
    out = args.output_root / now.strftime('%Y%m%dT%H%M%S%fZ')
    out.mkdir(parents=True, exist_ok=False)
    refs_path = PROJECT / 'data/native_reaction_references.json'
    reference = json.loads(refs_path.read_text())
    refs = {r['enzyme']: r for r in reference['enzymes']}
    height = 80 + PANEL_H * 3 + 50
    canvas = Canvas(W * 2 + 40, height)
    canvas.text(20, 15, 'Native reactions', 34, bold=True)
    data = []
    for i, scheme in enumerate(SCHEMES):
        ref = refs[scheme['enzyme']]
        single = Canvas(W, PANEL_H)
        panel(single, scheme, ref, 0)
        single.save(out / scheme['key'])
        canvas.paste(single, (i % 2) * (W + 40), 80 + (i // 2) * PANEL_H)
        data.append({**scheme, 'reference': ref,
                     'canonical_product_smiles': Chem.MolToSmiles(molecule(scheme['product'][1])),
                     'product_chiral_centers': Chem.FindMolChiralCenters(molecule(scheme['product'][1])),
                     'validation': 'SMILES sanitized; organic condensation balanced; one new amide or acyl-AMP bond selected'})
    y = 80 + PANEL_H * 3 + 5
    notes = [
        'Drawing convention: neutral acid/amine forms and conventional amide tautomers; protonation-dependent H⁺ terms omitted.',
        'R is a symbolic cyanophycin-chain continuation. SfaB makes an acyl adenylate; the other panels show amide formation.',
        'CphA1 is BAA17890.1 / P73833 (PCC6803/Kazusa), called “SuCphA1” in the nylon paper; UTEX2470 SuCphA1 is a different reference.',
        'ModelSEED IDs identify native reaction chemistry; they do not by themselves establish an exact protein-to-reaction mapping.',
    ]
    canvas.text(20, y, 'Turquoise: bond formed.  R: cyanophycin chain.  Neutral forms shown.', 22, MUTED)
    canvas.save(out / 'native_reactions', pdf=True)
    (out / 'depictions.json').write_text(json.dumps(data, indent=2) + '\n')
    # Retain the actual source records used to anchor these small-molecule graphs.
    import csv
    db = PROJECT.parents[1] / 'ModelSEEDDatabase/Biochemistry'
    compound_ids = {c for s in SCHEMES for c in s['source_compounds']}
    raw, input_hashes = {}, {str(Path(__file__).resolve()): digest(__file__), str(refs_path): digest(refs_path)}
    for file in sorted(db.glob('compound_[0-9][0-9].tsv')):
        for row in csv.DictReader(file.open(), delimiter='\t'):
            if row['id'] in compound_ids:
                raw[row['id']] = row
                input_hashes[str(file)] = digest(file)
    assert set(raw) == compound_ids
    (out / 'modelseed_compound_sources.json').write_text(json.dumps(raw, indent=2) + '\n')
    links = ''.join(f'<li><a href="{s["key"]}.svg">{html.escape(s["title"])}</a> '
                    f'— <a href="{refs[s["enzyme"]]["source"]}">primary source</a></li>' for s in SCHEMES)
    (out / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Native reaction diagrams</title>'
        '<style>body{font:17px/1.5 system-ui;max-width:1500px;margin:30px auto}img{width:100%}</style>'
        '<h1>Native reaction diagrams</h1><p><a href="native_reactions.svg">Vector SVG</a> · '
        '<a href="native_reactions.png">PNG</a> · <a href="native_reactions.pdf">PDF</a> · '
        '<a href="depictions.json">Structures and provenance</a></p><img src="native_reactions.svg">'
        '<ul>' + links + '</ul><p>' + '</p><p>'.join(map(html.escape, notes)) + '</p>')
    manifest = {'created_utc': now.isoformat(), 'rdkit_version': rdBase.rdkitVersion,
                'enzymes': 5, 'reaction_panels': 6, 'validation_passed': True,
                'layout': 'compact two-column, three-row grid; details in accompanying files',
                'modelseed_scope': 'local compound snapshot; not a live retrieval',
                'mcbA_acid_structure_reference': 'https://www.rcsb.org/ligand/EQ2',
                'input_sha256': input_hashes,
                'output_sha256': {p.name: digest(p) for p in out.iterdir() if p.is_file()}}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(out.resolve())


if __name__ == '__main__':
    main()
