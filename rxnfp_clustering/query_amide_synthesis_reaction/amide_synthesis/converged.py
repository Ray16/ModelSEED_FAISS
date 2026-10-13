#!/usr/bin/env python
"""Which candidates are backed by SEVERAL independent lines of evidence?

THE QUESTION THE SHORTLIST DOES NOT ANSWER. `shortlist` builds a diverse
screening panel: 301 proteins over 50 families, with caps of 12 per family and
3 per clade, because the right next experiment covers as many distinct families
as there are bench slots. That is the correct shape for "assay broadly" and the
wrong shape for "which enzymes are most likely to work" - diversity caps
actively remove the second-best member of a strong family.

This asks the other question. It does not rank; it counts how many
INDEPENDENT observations support each candidate, because agreement between
methods that can fail differently is worth more than a high score from any one.

THE AXES, and why independence had to be enforced by hand:

    measured_nylon_activity               assayed titre on the diad itself
    own_reaction_chemistry                >= 0.75 on the protein's OWN
                                          reaction - ModelSEED or the one
                                          UniProt curates for it
    lead_family_verdict                   its family passed on its own seeds
    crystallographic_bifunctional_ligand  a diamine or dicarboxylate has been
                                          seen in the pocket
    adenylation_competent_structure       a cofactor-bound structure exists
    independent_geometric_score           scored by the separate project

A FIRST VERSION COUNTED SIX AXES AND WAS WRONG. It listed cofactor-bound
structure and in-line adenylation geometry separately - but the geometry is
MEASURED FROM those structures, so a protein with one necessarily had both and
scored two lines for one observation. Collapsing them took candidates with
>= 5 lines from 4 to 0 and those with >= 4 from 16 to 4. Correlated axes
inflate agreement, which is the entire thing this file exists to measure, so
any new axis must be checked against the others before it is added.

WHAT A COUNT OF 4 DOES NOT MEAN. None of these is validated against nylon-6,6
activity, and with 3 measured positives none can be. Convergence means several
methods that fail differently happen to agree - a reason to spend a bench slot,
not evidence the enzyme works.
"""
from __future__ import annotations

import json

import pandas as pd

from .paths import RES

CHEMISTRY_FLOOR = 0.75
MIN_LINES = 3

# Family membership is a WEAK axis and the data says so: 42,490 of 54,199
# candidates (78%) carry a lead-family signature, because the lead families
# include ATP-grasp (9,160 members) and acyl-CoA N-acyltransferase (13,500).
# Tightening the verdict does not rescue it - requiring the strongest reason,
# the diacid+diamine pairing, still covers 65%, and additionally demanding a
# compact donor still covers 45%.
#
# An axis satisfied by most of the table cannot discriminate, and counting it
# alongside rare observations overstates agreement. So it is excluded from the
# headline count rather than tuned with an invented size threshold:
# `n_discriminating_lines` is the number that matters, `n_independent_lines`
# is kept beside it so nothing is hidden. The four candidates at the top are
# the same either way, which is the useful check.
WEAK_AXES = ('lead_family_verdict',)

SIDECARS = (
    ('ligand_evidence.tsv', ['bound_bifunctional_ligand',
                             'any_bifunctional_ligand']),
    ('pdb_evidence.tsv', ['pdb_has_atp_like']),
    ('adenylation_geometry.tsv', ['d_nuc_P']),
)


def _series(frame, name, default=None):
    """Always a Series aligned to `frame`.

    `DataFrame.get(name)` returns None for an absent column. That has now
    been the bug five times in this project - master_table, enrich,
    shortlist, ec_substrate and here, written minutes after fixing the
    fourth. Every new module gets this helper, not a `.get`.
    """
    if name in frame.columns:
        return frame[name]
    return pd.Series([default] * len(frame), index=frame.index)


def _num(frame, name):
    return pd.to_numeric(_series(frame, name), errors='coerce').fillna(0.0)


def lead_signatures(evaluation):
    return {s.strip() for row in evaluation[evaluation.verdict.eq('lead')]
            .itertuples()
            for s in str(getattr(row, 'signatures', '')).split(';') if s.strip()}


def axes(table, lead):
    """One column per independent line of evidence."""
    signatures = (_series(table, 'pfam', '').fillna('').astype(str) + ';'
                  + _series(table, 'interpro', '').fillna('').astype(str))
    cofactor = (_series(table, 'pdb_has_atp_like', False)
                .fillna(False).astype(bool))
    geometry = _series(table, 'd_nuc_P').notna()
    return pd.DataFrame({
        'measured_nylon_activity':
            _series(table, 'evidence_strength', '').eq('measured_activity'),
        'own_reaction_chemistry':
            _num(table, 'substrate_distance_measured').combine(
                _num(table, 'substrate_distance_own_rhea'), max)
            >= CHEMISTRY_FLOOR,
        'lead_family_verdict':
            signatures.map(lambda v: any(a in v for a in lead)),
        # Crystallographic OR curated: one claim about what the pocket holds,
        # from two partially correlated sources. Falls back to the
        # crystallographic column alone if the merged flag is absent.
        'bifunctional_ligand_in_the_pocket':
            (_series(table, 'any_bifunctional_ligand', False)
             .fillna(False).astype(bool)
             | _series(table, 'bound_bifunctional_ligand').notna()),
        # ONE observation: the geometry is measured from these structures.
        'adenylation_competent_structure': cofactor | geometry,
        'independent_geometric_score': _series(table, 'struct_S').notna(),
    })


def main():
    from .paths import ensure_directories
    ensure_directories()

    table = pd.read_csv(RES / 'master_candidates_enriched.tsv', sep='\t',
                        low_memory=False) if (
        RES / 'master_candidates_enriched.tsv').exists() else pd.read_csv(
        RES / 'master_candidates.tsv', sep='\t', low_memory=False)
    for name, columns in SIDECARS:
        path = RES / name
        if not path.exists():
            continue
        extra = pd.read_csv(path, sep='\t', low_memory=False)
        keep = [c for c in columns if c in extra.columns and
                c not in table.columns]
        if keep and 'uniprot' in extra.columns:
            table = table.merge(extra[['uniprot'] + keep].drop_duplicates(
                'uniprot'), on='uniprot', how='left')

    evaluation = RES / 'family_evaluation.tsv'
    lead = lead_signatures(pd.read_csv(evaluation, sep='\t')) \
        if evaluation.exists() else set()

    found = axes(table, lead)
    table['n_independent_lines'] = found.sum(axis=1)
    discriminating = [c for c in found.columns if c not in WEAK_AXES]
    table['n_discriminating_lines'] = found[discriminating].sum(axis=1)
    table['evidence_lines'] = found.apply(
        lambda row: '+'.join(c for c in found.columns if row[c]), axis=1)
    table['best_substrate_distance'] = pd.concat(
        [_num(table, 'substrate_distance_measured'),
         _num(table, 'substrate_distance_inherited'),
         _num(table, 'substrate_distance_own_rhea')], axis=1).max(axis=1).round(3)

    converged = table[table.n_discriminating_lines >= MIN_LINES].sort_values(
        ['n_discriminating_lines', 'best_substrate_distance'],
        ascending=False)
    converged.to_csv(RES / 'converged_candidates.tsv', sep='\t', index=False)

    summary = {
        'candidates': int(len(table)),
        'by_line_count': {int(k): int(v) for k, v in
                          table.n_independent_lines.value_counts().items()},
        'by_discriminating_line_count': {
            int(k): int(v) for k, v in
            table.n_discriminating_lines.value_counts().items()},
        'converged_at_or_above_%d' % MIN_LINES: int(len(converged)),
        'weak_axes_excluded_from_the_headline': list(WEAK_AXES),
        'axis_totals': {c: int(found[c].sum()) for c in found.columns},
        'note': 'Convergence of methods that fail differently - a reason to '
                'spend a bench slot, not evidence the enzyme works. Nothing '
                'here is validated against nylon-6,6 activity.',
    }
    print(json.dumps(summary, indent=2), flush=True)
    (RES / 'converged_summary.json').write_text(
        json.dumps(summary, indent=2) + '\n')

    show = ['uniprot', 'gene', 'organism', 'n_discriminating_lines',
            'n_independent_lines', 'best_substrate_distance',
            'evidence_lines']
    show = [c for c in show if c in converged.columns]
    print(f'\n{len(converged)} candidates on >= {MIN_LINES} DISCRIMINATING '
          f'lines (family membership excluded; see WEAK_AXES):', flush=True)
    print(converged.head(20)[show].to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
