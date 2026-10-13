#!/usr/bin/env python
"""What has crystallography already seen in each candidate's pocket?

ANOTHER DATASET USED FOR ONE QUESTION. The `pdb` stage retrieves 1,316 entries
and records, per candidate, whether a structure exists, whether it holds an
ATP-like cofactor and whether a metal is present. The `bound_components` column
- the identity of everything else in the pocket - was carried and never read.

That column is experimental substrate evidence, and this project has very
little of that: outside the 49-member assayed panel, every substrate claim here
is inferred from a reaction equation. A crystal structure with a diamine or a
dicarboxylate bound is a direct observation that the pocket accommodates the
motif nylon-6,6 needs.

So the distinct ligands are fetched once from RCSB, classified with this
project's OWN chemistry functions - the same `_acid_carbons` and
`_amine_nitrogens` the substrate screen uses, so a ligand is judged by exactly
the criteria a substrate is - and joined back per candidate.

Measured 2026-10-13: 147 distinct bound components, 18 of them a diamine or a
dicarboxylate, reaching 17 candidates. The two strongest corroborate
independently-derived conclusions rather than adding new ones, which is the
most useful thing a new evidence source can do:

    P22188  MurE, bound to API = 2,6-DIAMINOPIMELIC ACID - a C7 diacid AND
            diamine, 13 heavy atoms against adipate's 10. MurE was already one
            of the three staged cofold targets, chosen on reaction chemistry;
            this is crystallographic proof its pocket takes the motif.
    P0AES0  Glutathionylspermidine synthetase, bound to GGA and GSH, three free
            amines. The same family the CHAP-label investigation found to be
            genuine bifunctional amide synthetases.

NOT FOLDED INTO THE RANKING, and that is a judgement worth stating rather than
burying. `has_pdb` already contributes +1.0 to `master_priority`, so a bonus
for a bound bifunctional ligand would be consistent and defensible - it is an
experimental observation, not a prediction. It is left out because changing the
ranking is the user's call, not a side effect of adding an evidence source. The
column is reported and used as a tie-break; see the README.
"""
from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from .http import get_text
from .paths import CACHE, RES

API = 'https://data.rcsb.org/rest/v1/core/chemcomp/{component}'
LIGAND_CACHE = CACHE / 'pdb_ligand_chemistry.tsv'
THREADS = int(os.environ.get('LIGAND_THREADS', 6))

# What the nylon-6,6 bond needs on each side. Two of either makes a monomer
# that can extend a chain rather than cap it.
BIFUNCTIONAL_MIN = 2

# CURATED binding sites, as well as crystallographic ones.
#
# Crystallography reaches 17 candidates, because almost none has a structure
# with a ligand in it. UniProt's `binding_site` annotation names the ligand
# with a ChEBI id for 23,502 candidates (43%) - the same kind of claim about
# what the pocket holds, from sequence-level curation instead of a crystal.
# Only 90 distinct ligands appear, so this costs one cached file.
#
# Merged into the SAME axis as the crystallographic evidence rather than added
# as a new one. The two are partially correlated - a curator often reads the
# ligand off the structure - and this project has twice overstated convergence
# by counting correlated observations separately.
CHEBI_SMILES = CACHE / 'rhea-chebi-smiles.tsv'
CHEBI_URL = 'https://ftp.expasy.org/databases/rhea/tsv/rhea-chebi-smiles.tsv'
CHEBI_IN_TEXT = re.compile(r'ChEBI:CHEBI:(\d+)')

# A bound ligand only counts if it is also COMPACT. meso-diaminopimelate (13
# heavy atoms against adipate's 10) is the motif we want;
# UDP-N-acetylmuramoyl-L-alanyl-D-glutamate, at 58, is a scaffold the pocket
# is built around rather than a nylon-like monomer. The tolerance is the same
# 3x adipate that `family_evaluation` uses for donor parsimony, so the two
# cannot drift apart.
ADIPATE_HEAVY_ATOMS = 10
SCAFFOLD_TOLERANCE = 3.0
MAX_LIGAND_HEAVY_ATOMS = int(ADIPATE_HEAVY_ATOMS * SCAFFOLD_TOLERANCE)


def split_components(value):
    return [x.strip() for x in str(value or '').replace(',', ';').split(';')
            if x.strip()]


def fetch_component(component, fetch=get_text):
    """Name and SMILES for one PDB chemical component."""
    text, _, failed = fetch(API.format(component=component), timeout=40)
    if failed:
        return None
    try:
        payload = json.loads(text)
    except Exception:
        return None
    name = (payload.get('chem_comp') or {}).get('name')
    smiles = None
    for descriptor in payload.get('pdbx_chem_comp_descriptor') or []:
        kind = descriptor.get('type')
        if kind == 'SMILES_CANONICAL':
            smiles = descriptor.get('descriptor')
            break
        if kind == 'SMILES' and not smiles:
            smiles = descriptor.get('descriptor')
    return {'component': component, 'ligand_name': name, 'smiles': smiles}


def classify(rows):
    """Count acid and amine groups with the SUBSTRATE screen's own functions.

    Using the same predicates as the substrate screen is the point: a ligand
    is judged by exactly the criteria a substrate is, so the two cannot drift
    apart. Metals and malformed entries fall through to zeros rather than
    raising - some PDB components (metal-coordinated cobalamins) are not valid
    SMILES at all.
    """
    from rdkit import Chem
    from rdkit import RDLogger
    from .substrate_distance import _acid_carbons, _amine_nitrogens
    RDLogger.DisableLog('rdApp.*')

    out = []
    for row in rows:
        if not row:
            continue
        acids = amines = heavy = 0
        molecule = (Chem.MolFromSmiles(row['smiles'])
                    if row.get('smiles') else None)
        if molecule is not None:
            acids = len(_acid_carbons(molecule))
            amines = len(_amine_nitrogens(molecule))
            heavy = molecule.GetNumHeavyAtoms()
        out.append({**row, 'ligand_acids': acids, 'ligand_amines': amines,
                    'ligand_heavy_atoms': heavy,
                    'ligand_bifunctional': (acids >= BIFUNCTIONAL_MIN
                                            or amines >= BIFUNCTIONAL_MIN)})
    return pd.DataFrame(out)


def ligand_chemistry(components, fetch=get_text, threads=THREADS):
    """Classified chemistry for every component, cached on disk."""
    known = (pd.read_csv(LIGAND_CACHE, sep='\t', low_memory=False)
             if LIGAND_CACHE.exists() else pd.DataFrame())
    have = set(known.component) if len(known) else set()
    missing = sorted(set(components) - have)
    if missing:
        with ThreadPoolExecutor(max_workers=threads) as pool:
            rows = list(pool.map(lambda c: fetch_component(c, fetch), missing))
        fresh = classify(rows)
        known = (pd.concat([known, fresh], ignore_index=True)
                 if len(known) else fresh)
        if len(known):
            LIGAND_CACHE.parent.mkdir(parents=True, exist_ok=True)
            known.to_csv(LIGAND_CACHE, sep='\t', index=False)
    return known


def _chebi_smiles():
    """ChEBI id -> SMILES, from the cached Rhea bulk file."""
    if not CHEBI_SMILES.exists():
        body, _, failed = get_text(CHEBI_URL, timeout=180)
        if failed or not body:
            return {}
        CHEBI_SMILES.parent.mkdir(parents=True, exist_ok=True)
        CHEBI_SMILES.write_text(body)
    frame = pd.read_csv(CHEBI_SMILES, sep='\t', header=None,
                        names=['chebi', 'smiles'])
    return {str(r.chebi).replace('CHEBI:', ''): r.smiles
            for r in frame.itertuples()}


def chebi_chemistry(accessions_needed):
    """ChEBI id -> (acids, amines, heavy atoms), from the Rhea bulk SMILES.

    Rhea publishes ChEBI structures, and this project already trusts and
    caches Rhea reaction SMILES, so no new source is introduced. Classified
    with the substrate screen's own predicates, as the PDB components are.
    """
    from rdkit import Chem
    from rdkit import RDLogger
    from .substrate_distance import _acid_carbons, _amine_nitrogens
    RDLogger.DisableLog('rdApp.*')

    if not CHEBI_SMILES.exists():
        body, _, failed = get_text(CHEBI_URL, timeout=180)
        if failed or not body:
            return {}
        CHEBI_SMILES.parent.mkdir(parents=True, exist_ok=True)
        CHEBI_SMILES.write_text(body)
    frame = pd.read_csv(CHEBI_SMILES, sep='\t', header=None,
                        names=['chebi', 'smiles'])
    wanted = set(accessions_needed)
    out = {}
    for row in frame.itertuples():
        identifier = str(row.chebi).replace('CHEBI:', '')
        if identifier not in wanted:
            continue
        molecule = Chem.MolFromSmiles(str(row.smiles))
        if molecule is None:
            continue
        out[identifier] = (len(_acid_carbons(molecule)),
                           len(_amine_nitrogens(molecule)),
                           molecule.GetNumHeavyAtoms())
    return out


def curated_bifunctional(binding_sites, chemistry):
    """Per-row: the compact bifunctional ligands a curator assigned."""
    qualifying = {k for k, (acids, amines, heavy) in chemistry.items()
                  if (acids >= BIFUNCTIONAL_MIN or amines >= BIFUNCTIONAL_MIN)
                  and heavy <= MAX_LIGAND_HEAVY_ATOMS}
    out = []
    for text in binding_sites:
        found = sorted({c for c in CHEBI_IN_TEXT.findall(str(text or ''))
                        if c in qualifying})
        out.append(';'.join('CHEBI:' + c for c in found) if found else None)
    return out, qualifying


# POCKET CAPACITY for every candidate, not only the few in the dossier.
#
# HMDA spans 6 atoms between its amines and adipate 4 between its acids, so a
# pocket demonstrably holding something at least that long has ROOM on that
# side. Spermidine spans 8 and meso-diaminopimelate 5/5, which is why the
# polyamine and DAP handlers clear it.
#
# This is the half of the blocked cofold that needs no docking. It is a
# necessary condition and nowhere near sufficient - room is not competence -
# and it is deliberately not a convergence axis, because it derives from the
# same ligand data as the ligand axis and would be counted twice.
#
# Measured over the whole table: 3,397 candidates have room for HMDA, 13,777
# for adipate, and 1,844 for BOTH - of which only 4 are among the 28 converged
# candidates, so this surfaces proteins the convergence count does not.
HMDA_AMINE_SPAN = 6
ADIPATE_ACID_SPAN = 4
CHEBI_ANY = re.compile(r'chebi:(\d+)', re.I)


def ligand_spans(chemistry_by_id):
    """ChEBI id -> (amine span, acid span), cached per id."""
    from rdkit import Chem
    from rdkit import RDLogger
    from .substrate_distance import (_acid_carbons, _amine_nitrogens,
                                     _widest_spacer)
    RDLogger.DisableLog('rdApp.*')

    out = {}
    for identifier, smiles in chemistry_by_id.items():
        molecule = Chem.MolFromSmiles(str(smiles))
        if molecule is None:
            continue
        out[identifier] = (
            _widest_spacer(molecule, _amine_nitrogens(molecule))[0] or 0,
            _widest_spacer(molecule, _acid_carbons(molecule))[0] or 0)
    return out


def capacity(texts, spans):
    """Widest amine and acid span among each row's named ligands."""
    rows = []
    for text in texts:
        amine = acid = 0
        widest = None
        for identifier in set(CHEBI_ANY.findall(str(text or ''))):
            found = spans.get(identifier)
            if not found:
                continue
            if found[0] > amine:
                amine, widest = found[0], identifier
            acid = max(acid, found[1])
        rows.append({'amine_span': amine, 'acid_span': acid,
                     'widest_amine_ligand': widest,
                     'room_for_hmda': amine >= HMDA_AMINE_SPAN,
                     'room_for_adipate': acid >= ADIPATE_ACID_SPAN})
    return pd.DataFrame(rows)


def attach(evidence, chemistry):
    """Per-candidate best bound bifunctional ligand."""
    if not len(chemistry):
        return pd.DataFrame(columns=['uniprot'])
    table = chemistry.set_index('component')
    bifunctional = set(table[table.ligand_bifunctional.fillna(False)
                             .astype(bool)].index)
    rows = []
    for row in evidence.itertuples():
        found = [c for c in split_components(
            getattr(row, 'pdb_bound_components', '')) if c in bifunctional]
        if not found:
            rows.append({'uniprot': row.uniprot,
                         'bound_bifunctional_ligand': None,
                         'bound_ligand_acids': 0, 'bound_ligand_amines': 0,
                         'bound_ligand_name': None})
            continue
        best = max(found, key=lambda c: (table.loc[c, 'ligand_amines'],
                                         table.loc[c, 'ligand_acids']))
        rows.append({
            'uniprot': row.uniprot,
            'bound_bifunctional_ligand': ';'.join(sorted(found)),
            'bound_ligand_acids': int(table.loc[best, 'ligand_acids']),
            'bound_ligand_amines': int(table.loc[best, 'ligand_amines']),
            'bound_ligand_name': table.loc[best, 'ligand_name'],
        })
    return pd.DataFrame(rows)


def main():
    from .paths import ensure_directories
    ensure_directories()

    facts = RES / 'pdb_entry_facts.tsv'
    evidence = RES / 'pdb_evidence.tsv'
    if not (facts.exists() and evidence.exists()):
        raise SystemExit('run the pdb stage first; nothing to read')

    entries = pd.read_csv(facts, sep='\t', low_memory=False)
    components = sorted({c for value in entries.bound_components.dropna()
                         for c in split_components(value)})
    print(f'{len(entries):,} PDB entries, {len(components)} distinct bound '
          'components', flush=True)

    chemistry = ligand_chemistry(components)
    bifunctional = chemistry[chemistry.ligand_bifunctional.fillna(False)
                             .astype(bool)]
    print(f'{len(bifunctional)} are a diamine or a dicarboxylate', flush=True)

    per_candidate = attach(pd.read_csv(evidence, sep='\t', low_memory=False),
                           chemistry)

    # Curated binding sites, for every candidate rather than only those with a
    # structure. Joined onto the same frame and merged into one axis.
    master = RES / 'master_candidates_enriched.tsv'
    if not master.exists():
        master = RES / 'master_candidates.tsv'
    curated_count = 0
    if master.exists():
        table = pd.read_csv(master, sep='\t', low_memory=False,
                            usecols=lambda c: c in ('uniprot', 'binding_site'))
        sites = (table.binding_site.fillna('').astype(str)
                 if 'binding_site' in table.columns
                 else pd.Series('', index=table.index))
        needed = {c for text in sites for c in CHEBI_IN_TEXT.findall(text)}
        chebi = chebi_chemistry(needed)
        found, qualifying = curated_bifunctional(sites, chebi)
        curated = pd.DataFrame({'uniprot': table.uniprot,
                                'curated_bifunctional_ligand': found})
        print(f'{len(needed)} binding-site ChEBI ligands, {len(qualifying)} '
              f'compact and bifunctional', flush=True)
        per_candidate = per_candidate.merge(curated.drop_duplicates('uniprot'),
                                            on='uniprot', how='outer')
        curated_count = int(curated.curated_bifunctional_ligand.notna().sum())

    # Pocket capacity for the whole table, from the same ChEBI structures.
    if master.exists():
        full = pd.read_csv(
            master, sep='\t', low_memory=False,
            usecols=lambda c: c in ('uniprot', 'binding_site',
                                    'catalytic_activity'))
        text = (full.get('binding_site', pd.Series('', index=full.index))
                .fillna('').astype(str) + ' '
                + full.get('catalytic_activity',
                           pd.Series('', index=full.index))
                .fillna('').astype(str))
        needed = {c for value in text for c in CHEBI_IN_TEXT.findall(value)}
        structures = {k: v for k, v in _chebi_smiles().items() if k in needed}
        spans = ligand_spans(structures)
        room = capacity(text, spans)
        room.insert(0, 'uniprot', full.uniprot.values)
        room.to_csv(RES / 'pocket_capacity.tsv', sep='\t', index=False)
        print(f'pocket capacity: {int(room.room_for_hmda.sum()):,} have room '
              f'for HMDA, {int(room.room_for_adipate.sum()):,} for adipate, '
              f'{int((room.room_for_hmda & room.room_for_adipate).sum()):,} '
              'for both', flush=True)

    if 'curated_bifunctional_ligand' not in per_candidate.columns:
        per_candidate['curated_bifunctional_ligand'] = None
    # ONE axis: crystallographic or curated. See the module docstring.
    per_candidate['any_bifunctional_ligand'] = (
        per_candidate.bound_bifunctional_ligand.notna()
        | per_candidate.curated_bifunctional_ligand.notna())
    per_candidate.to_csv(RES / 'ligand_evidence.tsv', sep='\t', index=False)

    hits = per_candidate[per_candidate.bound_bifunctional_ligand.notna()]
    summary = {
        'pdb_entries': int(len(entries)),
        'distinct_components': len(components),
        'components_resolved': int(chemistry.smiles.notna().sum())
        if 'smiles' in chemistry else 0,
        'bifunctional_components': int(len(bifunctional)),
        'candidates_with_bound_bifunctional_ligand': int(len(hits)),
        'candidates_with_curated_bifunctional_ligand': curated_count,
        'candidates_on_either': int(
            per_candidate.any_bifunctional_ligand.sum()),
        'max_ligand_heavy_atoms': MAX_LIGAND_HEAVY_ATOMS,
        'note': 'Experimental observation of what the pocket holds. Reported '
                'and usable as a tie-break; deliberately NOT added to '
                'master_priority - see the module docstring.',
    }
    print(json.dumps(summary, indent=2), flush=True)
    (RES / 'ligand_evidence_summary.json').write_text(
        json.dumps(summary, indent=2) + '\n')

    if len(hits):
        print('\nCandidates whose pocket has been seen holding a diamine or '
              'dicarboxylate:', flush=True)
        print(hits.sort_values(['bound_ligand_amines', 'bound_ligand_acids'],
                               ascending=False).to_string(index=False),
              flush=True)


if __name__ == '__main__':
    main()
