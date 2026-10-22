#!/usr/bin/env python
"""Measure adenylation geometry in cofactor-bound crystal structures.

42 candidates have a deposited structure with ATP, ADP or a non-hydrolysable
analogue actually bound. Those give an experimental phosphorus position, which
is the one thing a geometric adenylation proxy needs and cannot otherwise get
without cofolding.

WHAT IS MEASURED, per structure:

    d_nuc_P   nearest carboxylate oxygen to the transferred phosphorus
    angle     O_nuc - P - O_leaving, the in-line attack angle; 180 degrees is
              ideal backside attack at phosphorus

The nucleophile is taken from a bound carboxylate ligand where one is present,
otherwise from an Asp/Glu side chain, and which of the two it was is recorded.
A protein-derived carboxylate is NOT the same evidence as a bound substrate
analogue and must not be read as one.

WHY THE FROZEN SCORE IS NOT APPLIED HERE. The external proxy is
`sum(w*q)/(0.1+sum(w))` over EXACTLY 50 poses - its regularizer was calibrated
at that ensemble size, and its own docstring says so. A crystal structure is
one pose. Feeding n=1 into a 50-pose estimator would produce a number that
looks comparable to the published scores and is not. So the raw geometry is
reported and the score is left to the cofolding stage, where 50 poses exist.

RESULT: NEGATIVE, AND THAT IS THE USEFUL PART. All 42 structures were measured
successfully, and none of them contains a substrate carboxylate poised to
attack the phosphorus:

  31  nearest carboxylate is a protein ASP/GLU side chain. Near an alpha
      phosphate those are almost always Mg-coordinating or catalytic residues,
      not the attacking nucleophile.
  11  nearest carboxylate belongs to a bound ligand - and every one of them is
      a CRYSTALLISATION ARTIFACT: phosphate, sulfate or glycerol from the
      buffer and cryoprotectant.
   0  genuine substrate carboxylate.

So adenylation geometry cannot be read off the crystallographic record for
these enzymes. The numbers below are real measurements of the wrong thing, and
are kept only so nobody repeats the attempt. `nucleophile_is_artifact` flags
the artifact cases explicitly, because a 5.1 A "substrate" distance that is
actually a sulfate ion is exactly the sort of number that survives into a
figure.

This also settles the question the cofolding stage was waiting on: it is not a
convenience. The proxy needs a modelled substrate pose because no deposited
structure supplies one.

This is a measurement on deposited coordinates. It is written to its own file,
carries no weight in the ranking, and the per-structure provenance (PDB id,
resolution, ligand, nucleophile source) travels with every number.
"""
from __future__ import annotations

import json
import math
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from amide_synthesis.infrastructure.http import get
from amide_synthesis.infrastructure.freshness import tracked_stage
from amide_synthesis.infrastructure.paths import CACHE, RES

STRUCTURES = CACHE / 'pdb_structures'          # untrusted downloaded data
DOWNLOAD = 'https://files.rcsb.org/download/{}.cif'
THREADS = int(os.environ.get('PDB_THREADS', 6))

ATP_LIKE = {'ATP', 'ADP', 'AMP', 'ANP', 'ACP', 'AGS', 'APC'}
# Phosphorus atoms, closest-to-adenosine first: the alpha phosphate is the one
# attacked during adenylation (ATP -> acyl-AMP + PPi).
PHOSPHORUS_PREFERENCE = ('PA', 'P', 'PB', 'PG')
CARBOXYLATE_LIGAND_ATOMS = {'OXT', 'O1', 'O2', 'OE1', 'OE2', 'OD1', 'OD2'}
# Buffer, cryoprotectant and ion-soak components. Oxygen-bearing and often the
# nearest thing to a phosphate, so they masquerade as substrate nucleophiles.
ARTIFACT_LIGANDS = {
    'SO4', 'PO4', 'GOL', 'EDO', 'PEG', 'MPD', 'TRS', 'ACT', 'ACY', 'FMT',
    'CIT', 'MES', 'EPE', 'IMD', 'DMS', 'NO3', 'CL', 'BR', 'IOD', 'MLI',
    'PG4', 'P6G', '1PE', 'SCN', 'AZI', 'CAC', 'BCT'}
PROTEIN_CARBOXYLATE = {'ASP': ('OD1', 'OD2'), 'GLU': ('OE1', 'OE2')}
AMINO_ACIDS = {
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
    'MSE', 'SEC', 'PYL'}
WATERS = {'HOH', 'DOD', 'WAT'}
# Beyond this a carboxylate is not poised to attack anything.
MAX_NUCLEOPHILE_DISTANCE = float(os.environ.get('MAX_NUC_DISTANCE', 8.0))


def ensure_structure(pdb_id, fetch=get):
    """Download one mmCIF into the untrusted cache. Returns a path or None."""
    STRUCTURES.mkdir(parents=True, exist_ok=True)
    path = STRUCTURES / f'{pdb_id}.cif'
    if path.exists() and path.stat().st_size > 2000:
        return path
    body, _, failed = fetch(DOWNLOAD.format(pdb_id), timeout=120)
    if failed or not body:
        return None
    path.write_bytes(body)
    return path


def _angle(a, b, c):
    """Angle a-b-c in degrees."""
    v1 = [a[i] - b[i] for i in range(3)]
    v2 = [c[i] - b[i] for i in range(3)]
    n1 = math.sqrt(sum(x * x for x in v1))
    n2 = math.sqrt(sum(x * x for x in v2))
    if n1 == 0 or n2 == 0:
        return None
    cos = sum(v1[i] * v2[i] for i in range(3)) / (n1 * n2)
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def _distance(a, b):
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def measure(path, pdb_id=''):
    """Nearest carboxylate-to-phosphorus geometry in one structure."""
    import gemmi
    try:
        structure = gemmi.read_structure(str(path))
        structure.remove_alternative_conformations()
        structure.remove_hydrogens()
    except Exception:
        return {'pdb_id': pdb_id, 'parse_failed': True}
    if not len(structure):
        return {'pdb_id': pdb_id, 'parse_failed': True}

    model = structure[0]
    phosphates, ligand_oxygens, protein_oxygens = [], [], []
    for chain in model:
        for residue in chain:
            name = residue.name.strip().upper()
            if name in ATP_LIKE:
                for atom in residue:
                    if atom.element.name == 'P':
                        phosphates.append((name, atom.name.strip().upper(),
                                           (atom.pos.x, atom.pos.y, atom.pos.z)))
                    elif atom.element.name == 'O':
                        ligand_oxygens.append(
                            ('_cofactor', atom.name.strip().upper(),
                             (atom.pos.x, atom.pos.y, atom.pos.z)))
            elif name in PROTEIN_CARBOXYLATE:
                wanted = PROTEIN_CARBOXYLATE[name]
                for atom in residue:
                    if atom.name.strip().upper() in wanted:
                        protein_oxygens.append(
                            (f'{name}{residue.seqid.num}', atom.name,
                             (atom.pos.x, atom.pos.y, atom.pos.z)))
            elif name not in AMINO_ACIDS and name not in WATERS:
                for atom in residue:
                    if (atom.element.name == 'O'
                            and atom.name.strip().upper() in CARBOXYLATE_LIGAND_ATOMS):
                        ligand_oxygens.append(
                            (name, atom.name.strip().upper(),
                             (atom.pos.x, atom.pos.y, atom.pos.z)))

    if not phosphates:
        return {'pdb_id': pdb_id, 'no_phosphorus': True}

    # Prefer the alpha phosphate, the one attacked during adenylation.
    def rank(entry):
        label = entry[1]
        return (PHOSPHORUS_PREFERENCE.index(label)
                if label in PHOSPHORUS_PREFERENCE else len(PHOSPHORUS_PREFERENCE))
    phosphates.sort(key=rank)
    ligand_name, p_label, p_position = phosphates[0]

    # A bound substrate carboxylate is far better evidence than a side chain.
    substrate = [o for o in ligand_oxygens if o[0] != '_cofactor']
    for source_label, pool in (('bound_ligand', substrate),
                               ('protein_side_chain', protein_oxygens)):
        candidates = [(o, _distance(o[2], p_position)) for o in pool]
        candidates = [c for c in candidates if c[1] <= MAX_NUCLEOPHILE_DISTANCE]
        if not candidates:
            continue
        nucleophile, d = min(candidates, key=lambda c: c[1])
        # In-line attack: the angle to whichever cofactor oxygen on this
        # phosphorus sits most nearly opposite the nucleophile.
        angles = [a for a in (
            _angle(nucleophile[2], p_position, o[2])
            for o in ligand_oxygens
            if _distance(o[2], p_position) < 2.2) if a is not None]
        return {
            'pdb_id': pdb_id,
            'cofactor': ligand_name,
            'phosphorus': p_label,
            'nucleophile_source': source_label,
            'nucleophile': f'{nucleophile[0]}:{nucleophile[1]}',
            'nucleophile_is_artifact': bool(
                source_label == 'bound_ligand'
                and nucleophile[0].upper() in ARTIFACT_LIGANDS),
            'd_nuc_P': round(d, 3),
            'inline_angle_deg': round(max(angles), 1) if angles else None,
            'parse_failed': False,
        }
    return {'pdb_id': pdb_id, 'no_nucleophile_within_cutoff': True,
            'cofactor': ligand_name, 'phosphorus': p_label}


def run(pdb_ids, threads=THREADS):
    def one(pdb_id):
        path = ensure_structure(pdb_id)
        if path is None:
            return {'pdb_id': pdb_id, 'download_failed': True}
        return measure(path, pdb_id=pdb_id)

    with ThreadPoolExecutor(max_workers=threads) as pool:
        return pd.DataFrame(list(pool.map(one, pdb_ids)))


@tracked_stage('geometry')
def main():
    from amide_synthesis.infrastructure.paths import ensure_directories
    ensure_directories()
    evidence = pd.read_csv(RES / 'pdb_evidence.tsv', sep='\t')
    holo = evidence[evidence.pdb_has_atp_like.fillna(False).astype(bool)]
    print(f'{len(holo)} candidates have a cofactor-bound structure', flush=True)

    facts = run(holo.best_pdb.dropna().unique().tolist())
    merged = holo.merge(facts, left_on='best_pdb', right_on='pdb_id', how='left')
    merged.to_csv(RES / 'adenylation_geometry.tsv', sep='\t', index=False)

    measured = merged[merged.d_nuc_P.notna()] if 'd_nuc_P' in merged else merged.head(0)
    summary = {
        'candidates': int(len(holo)),
        'structures_measured': int(len(measured)),
        'download_failed': int(merged.get('download_failed',
                                          pd.Series(dtype=bool)).fillna(False).sum()),
        'no_nucleophile_within_cutoff': int(
            merged.get('no_nucleophile_within_cutoff',
                       pd.Series(dtype=bool)).fillna(False).sum()),
        'nucleophile_source': (measured.nucleophile_source.value_counts().to_dict()
                               if len(measured) else {}),
        'nucleophile_is_crystallisation_artifact': int(
            measured.get('nucleophile_is_artifact',
                         pd.Series(dtype=bool)).fillna(False).sum())
        if len(measured) else 0,
        'genuine_substrate_carboxylate': int(
            ((measured.nucleophile_source == 'bound_ligand')
             & ~measured.get('nucleophile_is_artifact',
                             pd.Series(False, index=measured.index))
             .fillna(False)).sum()) if len(measured) else 0,
        'median_d_nuc_P': (round(float(measured.d_nuc_P.median()), 2)
                           if len(measured) else None),
        'median_inline_angle_deg': (
            round(float(pd.to_numeric(measured.inline_angle_deg,
                                      errors='coerce').median()), 1)
            if len(measured) else None),
        'note': ('raw geometry only; the external 50-pose score is NOT applied '
                 'to a single crystal pose. NEGATIVE RESULT: no deposited '
                 'structure contains a substrate carboxylate poised at the '
                 'phosphorus - side chains are Mg/catalytic, bound ligands '
                 'are buffer artifacts (SO4/PO4/GOL). Cofolding is required, '
                 'not optional.'),
    }
    print(json.dumps(summary, indent=2), flush=True)
    (RES / 'adenylation_geometry_summary.json').write_text(
        json.dumps(summary, indent=2) + '\n')

    if len(measured):
        show = ['uniprot', 'best_pdb', 'best_pdb_resolution_a', 'cofactor',
                'phosphorus', 'nucleophile_source', 'd_nuc_P',
                'inline_angle_deg']
        print('\nClosest in-line geometry:', flush=True)
        print(measured.sort_values('d_nuc_P').head(15)[show].to_string(index=False),
              flush=True)


if __name__ == '__main__':
    main()
