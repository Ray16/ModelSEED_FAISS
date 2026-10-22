#!/usr/bin/env python3
"""Draw the formal linear-chain nylon-6,6 polymerization target with RDKit.

Shows explicit H/OH chain ends and (2n-1) waters for n diacids + n diamines.
This is a separate polymerization illustration, not a new enzyme activity claim.
Run in the palm environment; PNG, SVG, PDF and the exact source graphs are saved.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
from rdkit import Chem, rdBase
from rdkit.Chem import rdDepictor, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D

from draw_native_reactions import Canvas, molecule

PROJECT = Path(__file__).resolve().parent
NS = 'http://www.w3.org/2000/svg'
ACID = 'O=C(O)CCCCC(=O)O'
AMINE = 'NCCCCCCN'
UNIT = 'NCCCCCCNC(=O)CCCCC(=O)'
REPEAT = '*' + UNIT + '*'


def counts(smiles):
    return Counter(a.GetAtomicNum() for a in Chem.AddHs(Chem.MolFromSmiles(smiles)).GetAtoms())


def repeat_drawing(canvas, x, y, width, height):
    mol = Chem.MolFromSmiles(REPEAT)
    ends = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
    mol.GetAtomWithIdx(ends[0]).SetProp('atomLabel', 'H')
    mol.GetAtomWithIdx(ends[1]).SetProp('atomLabel', 'OH')
    rdDepictor.Compute2DCoords(mol)
    conformer = mol.GetConformer()
    coords = conformer.GetPositions()
    delta = coords[ends[1]] - coords[ends[0]]
    angle = -np.arctan2(delta[1], delta[0])
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    for i, position in enumerate(coords):
        rotated = rotation @ position[:2]
        conformer.SetAtomPosition(i, (float(rotated[0]), float(rotated[1]), 0.))
    positions = None
    for kind in ['png', 'svg']:
        drawer = (rdMolDraw2D.MolDraw2DCairo if kind == 'png' else rdMolDraw2D.MolDraw2DSVG)(width, height)
        opts = drawer.drawOptions()
        opts.bondLineWidth = 2.6
        opts.minFontSize = 23
        opts.maxFontSize = 29
        opts.padding = .07
        drawer.DrawMolecule(mol)
        positions = [(drawer.GetDrawCoords(i).x, drawer.GetDrawCoords(i).y) for i in range(mol.GetNumAtoms())]
        drawer.FinishDrawing()
        if kind == 'png':
            canvas.image.paste(Image.open(io.BytesIO(drawer.GetDrawingText())).convert('RGB'), (x, y))
        else:
            tile = ET.fromstring(drawer.GetDrawingText())
            tile.set('x', str(x)); tile.set('y', str(y)); canvas.svg.append(tile)
    # Brackets cross the two terminal bonds; the H and OH end groups remain outside.
    neighbors = [mol.GetAtomWithIdx(e).GetNeighbors()[0].GetIdx() for e in ends]
    left = x + (positions[ends[0]][0] + positions[neighbors[0]][0]) / 2
    right = x + (positions[ends[1]][0] + positions[neighbors[1]][0]) / 2
    internal = [p for i, p in enumerate(positions) if i not in ends]
    top = y + min(p[1] for p in internal) - 23
    bottom = y + max(p[1] for p in internal) + 23
    for bx, sign in [(left, 1), (right, -1)]:
        canvas.line(bx, top, bx, bottom, width=2)
        canvas.line(bx, top, bx + sign * 12, top, width=2)
        canvas.line(bx, bottom, bx + sign * 12, bottom, width=2)
    canvas.text(right + 8, bottom - 5, 'n', 25)


def main():
    out = PROJECT / 'figures/nylon66_polymerization'
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'manifest.json').exists():
        raise RuntimeError('Output already exists; preserve or explicitly relocate it before regenerating.')
    for n in [1, 2, 3, 10]:
        left = Counter({k: v * n for k, v in (counts(ACID) + counts(AMINE)).items()})
        waters = Counter({k: v * (2 * n - 1) for k, v in counts('O').items()})
        assert left == counts(UNIT * n + 'O') + waters, f'Balance failed for n={n}'
    repeat = Chem.MolFromSmiles(REPEAT)
    composition = counts(REPEAT)
    assert {k: v for k, v in composition.items() if k != 0} == {6: 12, 1: 22, 7: 2, 8: 2}
    canvas = Canvas(2100, 390)
    canvas.text(25, 12, 'Nylon-6,6 polymerization', 33, bold=True)
    for smiles, x, label in [(ACID, 15, 'n Adipic acid'), (AMINE, 425, 'n Hexamethylenediamine')]:
        canvas.mol(molecule(smiles), x, 70, 360, 215)
        canvas.text(x + 180, 315, label, 23, anchor='middle')
    canvas.text(401, 158, '+', 35, anchor='middle')
    canvas.line(825, 185, 1005, 185)
    canvas.line(987, 174, 1005, 185); canvas.line(987, 196, 1005, 185)
    canvas.text(915, 137, 'Condensation', 22, anchor='middle')
    repeat_drawing(canvas, 1040, 75, 1030, 215)
    canvas.text(1555, 315, 'Nylon-6,6  +  (2n − 1) H₂O', 25, anchor='middle')
    canvas.save(out / 'nylon66_polymerization', pdf=True)
    description = dict(acid_smiles=ACID, diamine_smiles=AMINE, repeat_smiles=REPEAT,
                       finite_chain_construction='UNIT repeated n times, followed by terminal O; leading N has an implicit second H',
                       unit_smiles=UNIT, repeat_formula='C12H22N2O2',
                       equation='n adipic acid + n HMDA -> H[–NH(CH2)6NHCO(CH2)4CO–]nOH + (2n-1) H2O',
                       scope='Formal linear-chain condensation with specified H/OH end groups; not experimentally demonstrated enzymatic polymerization',
                       validated_chain_lengths=[1, 2, 3, 10])
    (out / 'structures.json').write_text(json.dumps(description, indent=2) + '\n')
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), rdkit_version=rdBase.rdkitVersion,
                    source_sha256={str(Path(__file__).resolve()): sha(Path(__file__)),
                                   str(PROJECT / 'draw_native_reactions.py'): sha(PROJECT / 'draw_native_reactions.py')},
                    output_sha256={p.name: sha(p) for p in out.iterdir() if p.is_file()}, validation_passed=True)
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(out.resolve())


if __name__ == '__main__': main()
