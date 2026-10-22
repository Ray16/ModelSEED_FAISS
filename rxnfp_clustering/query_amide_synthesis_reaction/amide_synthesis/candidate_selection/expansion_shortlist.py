"""Prioritise expanded homolog candidates for an E. coli-testable cofold screen.

The 13 reference enzymes are too few, and the best step-2 hit (GlnA3, Q9KZC7) is from
Streptomyces coelicolor -- a high-GC actinomycete that is hard to express in E. coli.
This ranks the 95%-identity representative homolog set (results/top10_mapping/
representatives95.tsv) so the next cofold screen can pick candidates that are both
promising and expressible.

Expression tier is a TRANSPARENT ORGANISM-NAME HEURISTIC, not a measurement:
  hard     actinobacteria (Streptomyces/Mycobacterium/...), archaea/halophiles,
           thermophiles -- high GC and/or folding/host problems in E. coli;
  easy     well-known mesophilic E. coli-friendly genera (enterobacteria,
           Pseudomonas, Vibrio, Bacillus subtilis, ...);
  moderate everything else (mostly mesophilic proteobacteria -- usually fine,
           but not explicitly whitelisted).
Confirm host suitability before ordering; codon optimisation handles GC at the DNA
level, so 'hard' mainly flags folding/maturation risk.
"""
import csv
import re
from pathlib import Path

from amide_synthesis.infrastructure.paths import RES

REPS = RES / 'top10_mapping/representatives95.tsv'
OUT = RES / 'top10_mapping'

HARD = re.compile(r'Streptomyces|Mycobacter|Corynebact|Nocardia|Rhodococc|Actinomyc|'
                  r'Frankia|Bifidobact|Kitasatospora|Amycolatopsis|Saccharopolyspora|'
                  r'Halo|Natrono|Haloferax|Archae|Thermo|Pyro|Sulfolob|Methano|'
                  r'Geobacillus|Deinococ|hyperthermo', re.I)
EASY = re.compile(r'Escherichia|Salmonella|Klebsiella|Enterobacter|Citrobacter|Shigella|'
                  r'Pseudomonas|Ectopseudomonas|Vibrio|Serratia|Pantoea|Erwinia|Yersinia|'
                  r'Acinetobacter|Aeromonas|Pseudocitrobacter|Bacillus subtilis', re.I)
TIER_RANK = {'easy': 0, 'moderate': 1, 'hard': 2}

GENE = {'P78061': 'PuuA', 'P39641': 'BacD', 'Q9KZC7': 'GlnA3', 'Q9RDS6': 'GlnA2sc',
        'I3R584': 'GlnA2hf', 'Q9WY79': 'MurE', 'E2JA32': 'DdaG', 'Q2FW70': 'SfnaD',
        'Q2G1M9': 'SbnE', 'A0A0F7RJ52': 'AsbA', 'Q2G1N1': 'SbnC', 'Q2G1M8': 'SbnF',
        'Q81RQ8': 'AsbB'}


def tier(organism):
    if HARD.search(organism or ''):
        return 'hard'
    if EASY.search(organism or ''):
        return 'easy'
    return 'moderate'


def load():
    rows = []
    for r in csv.DictReader(REPS.open(), delimiter='\t'):
        ident = float(r['identity']) if r['identity'] else 0.0
        rows.append({
            'rxn_id': r['rxn_id'], 'accession': r['accession'],
            'organism': r['organism'], 'expression_tier': tier(r['organism']),
            'identity_to_ref': round(ident, 3),
            'reference': f"{GENE.get(r['reference_accession'], r['reference_accession'])} "
                         f"({r['reference_accession']})",
            'reviewed': r['reviewed'], 'fragment': r['fragment'],
            'length': r['length'], 'mapping_route': r['mapping_route'],
            'sequence': r['sequence'],
        })
    return rows


def rank(rows):
    return sorted(rows, key=lambda r: (
        r['rxn_id'] != 'rxn05123',                       # GS-family (the deep pool) first
        TIER_RANK[r['expression_tier']],                 # easy host first
        r['fragment'] == 'True',                         # whole proteins first
        -r['identity_to_ref'],                           # closest to a working reference
    ))


def main(top_n=40):
    rows = rank(load())
    cols = ['rxn_id', 'accession', 'organism', 'expression_tier', 'identity_to_ref',
            'reference', 'reviewed', 'fragment', 'length', 'mapping_route']
    with (OUT / 'expansion_shortlist.tsv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=cols, delimiter='\t')
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in cols})

    # A recommended screen set: non-fragment, easy/moderate host, diverse across reactions.
    screen = [r for r in rows if r['fragment'] != 'True' and r['expression_tier'] != 'hard']
    pick, per_rxn = [], {}
    for r in screen:                                     # cap per reaction for diversity
        if per_rxn.get(r['rxn_id'], 0) < (top_n - 10 if r['rxn_id'] == 'rxn05123' else 3):
            pick.append(r); per_rxn[r['rxn_id']] = per_rxn.get(r['rxn_id'], 0) + 1
    pick = pick[:top_n]
    with (OUT / 'expansion_screen_set.fasta').open('w') as handle:
        for r in pick:
            handle.write(f">{r['accession']} {r['rxn_id']} {r['expression_tier']} "
                         f"id={r['identity_to_ref']} {r['organism']}\n{r['sequence']}\n")

    import collections
    by_rxn = collections.Counter((r['rxn_id'], r['expression_tier']) for r in rows
                                 if r['fragment'] != 'True')
    print(f"{len(rows)} representative homologs; recommended screen set = {len(pick)}")
    print('Non-fragment reps by reaction and expression tier:')
    rxns = sorted({r['rxn_id'] for r in rows}, key=lambda x: (x != 'rxn05123', x))
    for rxn in rxns:
        e, m, h = (by_rxn[(rxn, t)] for t in ('easy', 'moderate', 'hard'))
        print(f'  {rxn}  easy={e:>3}  moderate={m:>3}  hard={h:>2}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
