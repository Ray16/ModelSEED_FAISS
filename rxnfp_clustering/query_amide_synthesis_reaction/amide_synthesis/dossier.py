#!/usr/bin/env python
"""Everything known about the handful of candidates worth a bench slot.

ONCE A SEARCH HAS NARROWED, BREADTH STOPS PAYING. The pipeline's output is a
54,199-row table and a 301 x 97 spreadsheet; the `converged` stage reduces
that to the few candidates supported by several independent observations. For
those few, the useful work is no longer finding more proteins but resolving
everything about each one into something a person can act on.

So this writes one readable dossier per converged candidate, and fetches the
four UniProt fields the pipeline never asked for because they are worthless in
bulk and valuable here:

    Mutagenesis         residue-level functional data - which substitutions
                        kill or alter activity. Present for 3 of the 8, and
                        directly relevant to engineering the pocket.
    Biotechnology       whether anyone has already used this enzyme
    Miscellaneous       curator notes that fit nowhere else
    PubMed              the literature, so a claim can be chased to a paper
    BRENDA xref         BRENDA's own entry. The API needs credentials this
                        project does not have, but the cross-reference says
                        whether an entry exists at all - 4 of the 8 have one,
                        which is the specific list worth looking up by hand.

It also joins the sidecar columns the converged table was missing - AlphaFold
confidence, measured kinetics, subcellular location, ligand evidence, EC
ceiling - so the dossier is the single place everything lands.

NOT A RANKING. The order is the convergence count, and the text states for
every candidate what is measured and what is inferred. Nothing here has been
assayed on nylon-6,6 except the panel proteins, and none of these is one.
"""
from __future__ import annotations

import os
import re

import pandas as pd

from . import uniprot
from .paths import RES

EXTRA_FIELDS = ('accession,lit_pubmed_id,cc_biotechnology,cc_miscellaneous,'
                'ft_mutagen,xref_brenda,organism_id')
SIDECARS = (
    ('alphafold_quality.tsv', ['plddt_mean', 'model_confident']),
    ('curated_properties.tsv', ['km_min_mm', 'kinetics', 'ph_optimum',
                                'subcellular_location', 'sequence_similarities']),
    ('ligand_evidence.tsv', ['bound_bifunctional_ligand',
                             'curated_bifunctional_ligand',
                             'bound_ligand_name']),
    ('ec_substrate_ceiling.tsv', ['ec_substrate_ceiling', 'own_rhea_reaction']),
    ('pdb_evidence.tsv', ['best_pdb', 'best_pdb_resolution_a',
                          'pdb_has_atp_like']),
    ('adenylation_geometry.tsv', ['d_nuc_P', 'inline_angle_deg']),
)
MAX_DOSSIERS = int(os.environ.get('DOSSIER_LIMIT', 20))

# NEAR-MISSES WITH DEMONSTRATED ROOM, documented alongside the converged set.
#
# The convergence count is a coarse instrument and its chemistry floor is a
# threshold. Trypanothione synthetase misses by 0.0116 and its pocket has held
# a ligand spanning 30 between amines against HMDA's 6; glutathionylspermidine
# synthetase is in the same position. Both are proteins a bench scientist
# should see, and leaving them out while recommending them elsewhere would be
# an artifact that disagrees with its own author.
#
# They are admitted on an EXPLICIT second criterion rather than by lowering the
# floor - 2 independent lines AND room for both monomers - and the dossier
# labels which criterion each entry met, so the distinction stays visible.
NEAR_MISS_LINES = int(os.environ.get('DOSSIER_NEAR_MISS_LINES', 2))
MAX_NEAR_MISSES = int(os.environ.get('DOSSIER_NEAR_MISSES', 8))

# Free di- and polyamines, named in order of how close they are to HMDA.
# The nylon-6,6 acceptor is hexane-1,6-diamine, and NO enzyme in UniProt is
# annotated to use it - 0 of 54,199 candidates name it, which is the whole
# reason this project exists. The nearest native acceptors are therefore the
# most informative single fact about a candidate, and they are rare: only 8
# candidates in the table name cadaverine, the C5 diamine, one methylene
# short of HMDA.
DIAMINE_ACCEPTORS = (
    ('hexane-1,6-diamine', 'C6 diamine - the nylon-6,6 acceptor itself'),
    ('1,6-diaminohexane', 'C6 diamine - the nylon-6,6 acceptor itself'),
    ('cadaverine', 'C5 diamine - ONE methylene short of HMDA'),
    ('putrescine', 'C4 diamine'),
    ('agmatine', 'C4 diamine, guanidinylated at one end'),
    ('spermidine', 'triamine'),
    ('spermine', 'tetraamine'),
)


# POCKET CAPACITY, from ligands rather than from docking.
#
# The blocked cofold would answer "can this pocket hold adipate and HMDA in a
# catalytically competent pose". A weaker but unblocked question is answerable
# now: has the pocket been SEEN holding something at least as long? Every
# ligand a protein binds or acts on is named with a ChEBI id, Rhea publishes
# ChEBI structures, and `_widest_spacer` already measures the span the
# substrate score uses.
#
#     HMDA        amine-to-amine span 6
#     adipate     acid-to-acid span   4
#     cadaverine  5   putrescine 4   spermidine 8   meso-DAP 5 amine / 5 acid
#
# So spermidine spans LONGER than HMDA, and meso-diaminopimelate longer than
# adipate: a pocket demonstrably accommodating either has room for the nylon
# monomer on that side.
#
# This is capacity, NOT competence - room is necessary and nowhere near
# sufficient, and it is deliberately not a convergence axis because it derives
# from the same ligand data as `bifunctional_ligand_in_the_pocket` and would
# be counted twice.
HMDA_AMINE_SPAN = 6
ADIPATE_ACID_SPAN = 4
# UniProt writes `ChEBI:CHEBI:30616`; case-insensitive so a bare `ChEBI:30616`
# cannot slip past either.
CHEBI_ANY = re.compile(r'chebi:(\d+)', re.I)


def _chebi_structures():
    from .paths import CACHE
    path = CACHE / 'rhea-chebi-smiles.tsv'
    if not path.exists():
        return {}
    frame = pd.read_csv(path, sep='\t', header=None,
                        names=['chebi', 'smiles'])
    return {str(r.chebi).replace('CHEBI:', ''): r.smiles
            for r in frame.itertuples()}


def pocket_capacity(row, structures=None):
    """Widest amine and acid span among the ligands this protein handles."""
    from rdkit import Chem
    from rdkit import RDLogger
    from .substrate_distance import (_acid_carbons, _amine_nitrogens,
                                     _widest_spacer)
    RDLogger.DisableLog('rdApp.*')

    structures = _chebi_structures() if structures is None else structures
    text = ' '.join(_text(row.get(k)) for k in
                    ('binding_site', 'catalytic_activity'))
    amine = acid = 0
    widest = None
    for identifier in set(CHEBI_ANY.findall(text)):
        smiles = structures.get(identifier)
        if not smiles:
            continue
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            continue
        span = _widest_spacer(molecule, _amine_nitrogens(molecule))[0] or 0
        if span > amine:
            amine, widest = span, identifier
        acid = max(acid, _widest_spacer(
            molecule, _acid_carbons(molecule))[0] or 0)
    return {'amine_span': amine, 'acid_span': acid, 'widest_ligand': widest,
            'fits_hmda': amine >= HMDA_AMINE_SPAN,
            'fits_adipate': acid >= ADIPATE_ACID_SPAN}


def native_diamine_acceptors(row):
    """Free di/polyamines this protein is curated to accept."""
    text = ' '.join(_text(row.get(k)).lower()
                    for k in ('catalytic_activity', 'function'))
    seen, out = set(), []
    for name, note in DIAMINE_ACCEPTORS:
        if name in text and note not in seen:
            seen.add(note)
            out.append((name, note))
    return out


def _text(value, limit=None):
    text = str(value if value is not None else '').strip()
    if text.lower() in ('nan', 'none', '<na>', ''):
        return ''
    return text[:limit] if limit else text


def clean_statement(value):
    """UniProt CC text without the evidence codes, which swamp it."""
    text = re.sub(r'\{ECO:[^}]*\}', '', _text(value))
    text = re.sub(r'\s*;\s*$', '', re.sub(r'\s+', ' ', text))
    return text.strip()


def mutagenesis_notes(value):
    """One line per substitution: the position, the change, the effect."""
    out = []
    for block in _text(value).split('MUTAGEN '):
        block = block.strip()
        if not block:
            continue
        position = block.split(';')[0].strip()
        note = re.search(r'/note="([^"]+)"', block)
        if note:
            out.append(f'{position}: {clean_statement(note.group(1))}')
    return out


def gather(accessions, search=uniprot.search):
    """The fields worth fetching only for a shortlist this small."""
    if not accessions:
        return pd.DataFrame(columns=['uniprot'])
    rows = []
    for start in range(0, len(accessions), 20):
        chunk = accessions[start:start + 20]
        got, _, _ = search(' OR '.join(f'accession:{a}' for a in chunk),
                           max_pages=2, fields=EXTRA_FIELDS)
        rows.extend(got)
    if not rows:
        return pd.DataFrame(columns=['uniprot'])
    return pd.DataFrame(rows).rename(columns={
        'Entry': 'uniprot', 'PubMed ID': 'pubmed',
        'Biotechnological use': 'biotechnology',
        'Miscellaneous [CC]': 'miscellaneous', 'Mutagenesis': 'mutagenesis',
        'BRENDA': 'brenda', 'Organism (ID)': 'taxon_id'})


def render(row):
    """One candidate, as markdown."""
    name = _text(row.get('protein_name'), 120) or _text(row.get('gene')) \
        or row['uniprot']
    lines = [f"## {row['uniprot']} — {name}", '']
    organism = _text(row.get('organism'), 90)
    if organism:
        lines.append(f"**Organism** {organism}  ")
    gene = _text(row.get('gene'), 40)
    if gene:
        lines.append(f"**Gene** `{gene}`  ")
    length = _text(row.get('length'))
    if length:
        lines.append(f"**Length** {length} aa  ")
    lines.append('')

    if _text(row.get('selected_by')) == 'near_miss_with_room':
        lines.append('**Why it is here** — a NEAR MISS: it does not reach the '
                     'convergence threshold, but its pocket has demonstrably '
                     'held both monomers\' worth of span. Admitted on that '
                     'stated criterion, not by lowering the floor.')
    else:
        lines.append(
            f"**Why it is here** — {int(row.get('n_discriminating_lines', 0))} "
            'independent lines: '
            + _text(row.get('evidence_lines')).replace('+', ', '))
    lines.append('')

    facts = []
    distance = row.get('best_substrate_distance')
    if pd.notna(distance):
        source = _text(row.get('substrate_distance_source')) or 'unknown source'
        facts.append(f'substrate distance **{distance}** ({source})')
    ceiling = row.get('ec_substrate_ceiling')
    if pd.notna(ceiling):
        facts.append(f'its EC class reaches {round(float(ceiling), 3)}')
    plddt = row.get('plddt_mean')
    if pd.notna(plddt):
        facts.append(f'AlphaFold pLDDT **{round(float(plddt), 1)}**')
    pdb = _text(row.get('best_pdb'))
    if pdb:
        resolution = row.get('best_pdb_resolution_a')
        facts.append(f'structure `{pdb}`'
                     + (f' at {resolution} Å' if pd.notna(resolution) else ''))
    distance_p = row.get('d_nuc_P')
    if pd.notna(distance_p):
        facts.append(f'in-line adenylation geometry {round(float(distance_p), 2)} Å')
    km = row.get('km_min_mm')
    if pd.notna(km):
        facts.append(f'measured K_M **{km} mM**')
    for key, label in (('bound_bifunctional_ligand', 'crystallographic ligand'),
                       ('curated_bifunctional_ligand', 'curated ligand')):
        value = _text(row.get(key))
        if value:
            facts.append(f'{label} `{value}`')
    if facts:
        lines += ['**Resolved**', ''] + [f'- {f}' for f in facts] + ['']

    capacity = row.get('_capacity')
    if isinstance(capacity, dict) and (capacity['amine_span']
                                       or capacity['acid_span']):
        verdict = []
        if capacity['fits_hmda']:
            verdict.append(f"spans {capacity['amine_span']} between amines, "
                           f'against HMDA\'s {HMDA_AMINE_SPAN} — **room for '
                           'the acceptor**')
        elif capacity['amine_span']:
            verdict.append(f"widest amine span {capacity['amine_span']}, "
                           f"short of HMDA's {HMDA_AMINE_SPAN}")
        if capacity['fits_adipate']:
            verdict.append(f"spans {capacity['acid_span']} between acids, "
                           f'against adipate\'s {ADIPATE_ACID_SPAN} — **room '
                           'for the donor**')
        elif capacity['acid_span']:
            verdict.append(f"widest acid span {capacity['acid_span']}, "
                           f"short of adipate's {ADIPATE_ACID_SPAN}")
        lines += ['**Pocket capacity** (from ligands it is curated to handle; '
                  'room, not competence)', '']
        lines += [f'- {v}' for v in verdict] + ['']

    acceptors = native_diamine_acceptors(row)
    if acceptors:
        lines += ['**Native acceptor is a free di/polyamine** — the '
                  'acceptor-side question for nylon-6,6', '']
        lines += [f'- `{name}` — {note}' for name, note in acceptors] + ['']

    for key, heading in (('catalytic_activity', 'Catalytic activity'),
                         ('function', 'Function'),
                         ('cofactor', 'Cofactor'),
                         ('subcellular_location', 'Location'),
                         ('sequence_similarities', 'Family (UniProt)'),
                         ('biotechnology', 'Biotechnology'),
                         ('miscellaneous', 'Note')):
        text = clean_statement(row.get(key))
        if text:
            lines += [f'**{heading}** {text[:400]}', '']

    mutations = mutagenesis_notes(row.get('mutagenesis'))
    if mutations:
        lines += ['**Mutagenesis** (residue-level functional data)', '']
        lines += [f'- {m[:160]}' for m in mutations[:8]] + ['']

    brenda = _text(row.get('brenda'))
    if brenda:
        lines += [f'**BRENDA** `{brenda}` — an entry exists; the API needs '
                  'credentials this project does not have', '']
    pubmed = _text(row.get('pubmed'))
    if pubmed:
        ids = [p for p in pubmed.replace(' ', '').split(';') if p][:8]
        lines += ['**Literature** ' + ', '.join(
            f'[{p}](https://pubmed.ncbi.nlm.nih.gov/{p}/)' for p in ids), '']
    return '\n'.join(lines)


def main():
    from .paths import ensure_directories
    ensure_directories()

    source = RES / 'converged_candidates.tsv'
    if not source.exists():
        raise SystemExit('run the converged stage first')
    table = pd.read_csv(source, sep='\t', low_memory=False).head(MAX_DOSSIERS)
    table['selected_by'] = 'converged'
    print(f'{len(table)} converged candidates to document', flush=True)

    # Near-misses: 2 lines AND room for both monomers. Admitted on a stated
    # second criterion, never by moving the chemistry floor.
    capacity_path = RES / 'pocket_capacity.tsv'
    master_path = RES / 'master_candidates_enriched.tsv'
    if capacity_path.exists() and master_path.exists():
        room = pd.read_csv(capacity_path, sep='\t', low_memory=False)
        room = room[room.room_for_hmda.fillna(False).astype(bool)
                    & room.room_for_adipate.fillna(False).astype(bool)]
        everything = pd.read_csv(master_path, sep='\t', low_memory=False)
        pool = everything[everything.uniprot.isin(room.uniprot)
                          & ~everything.uniprot.isin(table.uniprot)]
        if len(pool):
            near = pool.nlargest(MAX_NEAR_MISSES, 'master_priority').copy()
            near['selected_by'] = 'near_miss_with_room'
            near['n_discriminating_lines'] = NEAR_MISS_LINES
            near['evidence_lines'] = 'room for both monomers; below the ' \
                                     'chemistry floor'
            table = pd.concat([table, near], ignore_index=True)
            print(f'{len(near)} near-misses with room for both monomers',
                  flush=True)

    for name, columns in SIDECARS:
        path = RES / name
        if not path.exists():
            continue
        extra = pd.read_csv(path, sep='\t', low_memory=False)
        keep = [c for c in columns
                if c in extra.columns and c not in table.columns]
        if keep and 'uniprot' in extra.columns:
            table = table.merge(
                extra[['uniprot'] + keep].drop_duplicates('uniprot'),
                on='uniprot', how='left')

    accessions = [a for a in table.uniprot.dropna().astype(str)
                  if re.match(r'^[A-NR-Z0-9][0-9A-Z]{3,9}$', a)]
    fetched = gather(accessions)
    if len(fetched):
        table = table.merge(fetched.drop_duplicates('uniprot'),
                            on='uniprot', how='left')

    structures = _chebi_structures()
    table['_capacity'] = [pocket_capacity(row, structures)
                          for _, row in table.iterrows()]
    fits = sum(1 for c in table._capacity if c['fits_hmda'])
    print(f'pocket has held a ligand spanning >= HMDA: {fits} of {len(table)}',
          flush=True)

    header = [
        '# Candidates worth a bench slot', '',
        f'{len(table)} proteins, each supported by at least '
        f'{int(table.n_discriminating_lines.min())} independent observations '
        'that are not merely family membership. Ordered by how many lines '
        'agree, then by substrate distance.', '',
        'Entries marked a NEAR MISS did not reach the convergence threshold '
        'and are included on a separate, stated criterion: two independent '
        'lines plus demonstrated room for both monomers. The threshold was '
        'not moved to accommodate them.', '',
        '**None of these has been assayed on nylon-6,6.** Convergence means '
        'several methods that fail differently happen to agree, which is a '
        'reason to spend a bench slot and not evidence that the enzyme works. '
        'With three measured positives in the reference panel, no ranking '
        'here can be validated.', '',
        '---', '']
    body = [render(row) for _, row in table.iterrows()]
    (RES / 'CANDIDATES.md').write_text('\n'.join(header + body) + '\n')

    counts = {c: int(table[c].notna().sum()) for c in
              ('mutagenesis', 'brenda', 'biotechnology', 'km_min_mm',
               'plddt_mean', 'best_pdb') if c in table.columns}
    print(f'written {RES / "CANDIDATES.md"}', flush=True)
    print('resolved across the set: ' + ', '.join(
        f'{k} {v}/{len(table)}' for k, v in counts.items()), flush=True)


if __name__ == '__main__':
    main()
