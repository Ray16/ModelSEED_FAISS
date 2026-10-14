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
import re

import pandas as pd

from .paths import RES

# 0.75 is a round number, so it was checked against the distribution rather
# than left as a guess. Substrate distance is DISCRETE here - each value is one
# reaction's score, shared by every protein annotated to it - and 0.75 falls in
# a real gap:
#
#     ... 0.7439 (8)  0.7465 (8)  0.7467 (1,154) | 0.7526 (1)  0.7640 (1,745) ...
#
# so it separates clusters instead of slicing one. Lowering it to 0.73 would
# admit 5,200 instead of 3,862.
#
# THE NEAR MISS WORTH KNOWING ABOUT. Trypanothione synthetase (O60993, Q9GT49,
# EC 6.3.1.9) scores 0.7384 - short by 0.0116 - and is otherwise a strong
# candidate: it acylates a free diamine, sits in a lead family, and its pocket
# has held a ligand spanning 30 between amines against HMDA's 6. It is
# deliberately NOT rescued by moving the floor, because tuning a threshold to
# admit a candidate one likes is how a ranking stops meaning anything. It sits
# at 2 discriminating lines and a reader who wants near-misses should read
# `n_discriminating_lines == 2`, which is 1,795 candidates.
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

# TESTED AND REJECTED: a curated ATP-binding site as structural evidence.
#
# `adenylation_competent_structure` reaches only 57 candidates, because few
# have a structure with a cofactor in it, and the obvious thickening is
# UniProt's own `binding_site` ATP annotation - 12,145 candidates, 22.4%,
# which is not weak by coverage. Coverage was the wrong test. Measured, it
# takes the >= 3-line list from 8 to 1,759: a 220-fold inflation.
#
# The reason is chemical, not statistical. Binding ATP is a PRECONDITION
# shared by essentially every ATP-dependent ligase, and this candidate set is
# selected for ligases - so the annotation is necessary rather than
# evidential. The crystallographic form earns its place because it shows the
# cofactor positioned in THAT pocket and comes with a measured nucleophile
# distance and attack angle; the sequence annotation carries neither.
#
# 1,759 "candidates on three independent lines" would have read as progress
# and meant nothing. Any new axis must be tested for what it does to the
# distribution, not just for how much it covers.
REJECTED_AXES = {
    'curated_atp_binding_site': (
        'a precondition of every ATP-dependent ligase, not evidence of amide '
        'formation; inflates the >= 3-line list from 8 to 1,759'),
}

SIDECARS = (
    # `curated_bifunctional_ligand` is needed to tell a curated ligand that
    # merely restates the curated reaction from a crystallographic one.
    ('ligand_evidence.tsv', ['bound_bifunctional_ligand',
                             'any_bifunctional_ligand',
                             'curated_bifunctional_ligand']),
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


CHEBI_ID = re.compile(r'CHEBI:\d+')


def ligand_restates_own_reaction(table):
    """Is the "ligand in the pocket" just the protein's own reaction again?

    MEASURED, and it is nearly total: of 1,754 candidates whose two
    discriminating lines were `own_reaction_chemistry` and
    `bifunctional_ligand_in_the_pocket`, **1,753** carry a curated
    binding-site ligand whose ChEBI id appears verbatim in their own
    catalytic-activity line. UniProt curates "binds UDP-MurNAc-L-Ala-D-Glu"
    and "catalyses a reaction consuming UDP-MurNAc-L-Ala-D-Glu" from the same
    knowledge; counting both is counting one fact twice. Collapsing it takes
    the headline from 28 candidates at >= 3 lines to 17, and the >= 2 set from
    1,823 to 84.

    Two cases are deliberately NOT collapsed:

    - a CRYSTALLOGRAPHIC ligand. Someone solved a structure and saw the
      molecule; that is an observation, not a restatement of the annotation,
      so `bound_bifunctional_ligand` always keeps its line. This is why MurE
      P9WJL3 is unaffected - it really does have UAG in the density.
    - a curated ligand the protein's own reaction does NOT involve. The
      pocket then holds something the annotated chemistry does not explain,
      which is a genuinely separate claim.

    And the collapse only applies where BOTH lines would otherwise fire,
    since the defect is double-counting rather than the ligand evidence
    itself being invalid.
    """
    curated = _series(table, 'curated_bifunctional_ligand').fillna('')
    crystallographic = _series(table, 'bound_bifunctional_ligand').notna()
    activity = _series(table, 'catalytic_activity').fillna('').astype(str)
    restates = pd.Series(
        [bool(ids) and any(i in text for i in ids)
         for ids, text in zip(curated.astype(str).map(CHEBI_ID.findall),
                              activity)],
        index=table.index)
    return restates & ~crystallographic


def axes(table, lead):
    """One column per independent line of evidence."""
    signatures = (_series(table, 'pfam', '').fillna('').astype(str) + ';'
                  + _series(table, 'interpro', '').fillna('').astype(str))
    cofactor = (_series(table, 'pdb_has_atp_like', False)
                .fillna(False).astype(bool))
    geometry = _series(table, 'd_nuc_P').notna()
    own_chemistry = (
        _num(table, 'substrate_distance_measured').combine(
            _num(table, 'substrate_distance_own_rhea'), max)
        >= CHEMISTRY_FLOOR)
    return pd.DataFrame({
        'measured_nylon_activity':
            _series(table, 'evidence_strength', '').eq('measured_activity'),
        # The screen proved the acceptor is a free DIAMINE - what HMDA is.
        # Rare (0.35% of the table) and near-orthogonal to the chemistry axis
        # (Jaccard 0.01; 138 of its 189 carry information no other axis has),
        # which is why it qualifies where curated ATP binding did not.
        'acylates_a_free_diamine':
            pd.to_numeric(_series(table, 'acylates_free_diamine', False),
                          errors='coerce').fillna(0).astype(bool),
        'own_reaction_chemistry': own_chemistry,
        'lead_family_verdict':
            signatures.map(lambda v: any(a in v for a in lead)),
        # Crystallographic OR curated: one claim about what the pocket holds,
        # from two partially correlated sources. Falls back to the
        # crystallographic column alone if the merged flag is absent.
        #
        # MINUS the case where the curated ligand simply restates the curated
        # reaction - see `ligand_restates_own_reaction`. That correlation was
        # missed when this axis was added: the check made at the time was
        # crystallographic-vs-curated, not ligand-vs-own-reaction.
        'bifunctional_ligand_in_the_pocket':
            (_series(table, 'any_bifunctional_ligand', False)
             .fillna(False).astype(bool)
             | _series(table, 'bound_bifunctional_ligand').notna())
            & ~(ligand_restates_own_reaction(table) & own_chemistry),
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

    # Publish the count for EVERY candidate, not only those clearing the
    # threshold. The near-miss tier needs to know which candidates hold
    # exactly two lines, and that information existed nowhere outside this
    # function - converged_candidates.tsv holds only the >= 3 subset, so a
    # downstream stage could not ask the question at all.
    table[['uniprot', 'n_independent_lines', 'n_discriminating_lines',
           'evidence_lines']].to_csv(RES / 'evidence_lines.tsv', sep='\t',
                                     index=False)

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
