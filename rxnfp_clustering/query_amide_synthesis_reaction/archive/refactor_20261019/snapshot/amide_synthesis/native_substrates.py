#!/usr/bin/env python
"""Extract each candidate's native reaction from UniProt, and group by family.

Reaction-level similarity says a candidate does the right KIND of chemistry. It
says nothing about what the enzyme actually accepts, nor whether a dozen hits
are really one family counted twelve times. Both have to be read off curated
annotation:

* the native reaction comes from the catalytic-activity statement, with its
  evidence code and PubMed support, so an experimentally demonstrated substrate
  is distinguishable from one inferred by similarity;
* the family comes from Pfam, with names resolved through the InterPro API, so
  the shortlist can be reported as families rather than as a redundant protein
  list.

This matters concretely: BibC's and PubC's native substrate turns out to be a
diamine-diacid HALF-AMIDE - free carboxylate at one end, free primary amine at
the other - which is structurally the same class as the nylon-6,6 diad this
project encodes as a PRODUCT. That reframes the experiment from "make the diad"
to "feed the diad to an enzyme that natively dimerises exactly this monomer".

Outputs (results/):
  native_substrates.tsv   per enzyme x reaction statement, with evidence
  enzyme_families.tsv     candidates collapsed to Pfam families
"""
from __future__ import annotations

import re
import sys
from collections import Counter, defaultdict

import pandas as pd

from .freshness import tracked_stage
from .paths import RES
from .families import pfam_names, split_accessions
TOP = 60            # candidates to annotate, by cosine

REACTION = re.compile(r'Reaction=(.+?);\s*Xref=')
RHEA = re.compile(r'Rhea:(RHEA:\d+)')
EC = re.compile(r'EC=([\d.\-]+)')
EVIDENCE = re.compile(r'Evidence=\{([^}]*)\}')
PUBMED = re.compile(r'PubMed:(\d+)')

# Species in nearly every ATP-dependent ligase equation; they say nothing about
# substrate scope and would swamp the substrate column.
COFACTORS = {'atp', 'adp', 'amp', 'diphosphate', 'phosphate', 'h(+)', 'h2o',
             'co2', 'coa', 'nad(+)', 'nadh', 'nadp(+)', 'nadph', 'o2', 'nh4(+)'}


def statements(blob: str):
    """Split a UniProt catalytic-activity blob into individual reactions."""
    if not isinstance(blob, str):
        return []
    out = []
    for chunk in blob.split('CATALYTIC ACTIVITY:'):
        chunk = chunk.strip()
        if not chunk:
            continue
        reaction = REACTION.search(chunk)
        if not reaction:
            continue
        ec = EC.search(chunk)
        evidence = EVIDENCE.search(chunk)
        out.append({
            'native_reaction': reaction.group(1).strip(),
            'rhea': ';'.join(list(dict.fromkeys(RHEA.findall(chunk)))[:3]),
            'ec': ec.group(1) if ec else '',
            'evidence': evidence.group(1).split(',')[0].strip() if evidence else '',
            'pubmed': ';'.join(dict.fromkeys(PUBMED.findall(chunk))),
        })
    return out


def substrates(equation: str):
    """Non-cofactor species on the left-hand side of a native equation."""
    left = re.split(r'\s=\s', equation)[0]
    found = []
    for term in re.split(r'\s\+\s', left):
        name = re.sub(r'^\d+\s+', '', term).strip()
        if name.lower() in COFACTORS or not name:
            continue
        found.append(name)
    return found


@tracked_stage('native')
def main():
    from .paths import ensure_directories
    ensure_directories()
    candidates = pd.read_csv(RES / 'candidates_ranked.tsv', sep='\t')
    annotations = pd.read_csv(RES / 'enzyme_annotations.tsv', sep='\t')
    per_protein = annotations.drop_duplicates('uniprot').set_index('uniprot')
    activity = per_protein['catalytic_activity'].to_dict()
    pfam_of = per_protein['pfam'].to_dict() if 'pfam' in per_protein else {}

    top = (candidates.sort_values('cosine_similarity', ascending=False)
                     .drop_duplicates('uniprot').head(TOP))

    # ---- resolve family names once for every Pfam seen
    seen = set()
    for uniprot in top.uniprot:
        for accession in split_accessions(pfam_of.get(uniprot)):
            if accession.strip():
                seen.add(accession.strip())
    print(f'resolving {len(seen)} Pfam families', flush=True)
    names = pfam_names(sorted(seen))

    rows = []
    for row in top.itertuples():
        accessions = split_accessions(pfam_of.get(row.uniprot))
        family_ids = ';'.join(accessions)
        family_names = ';'.join(names.get(a, '') or a for a in accessions)
        found = statements(activity.get(row.uniprot, ''))
        base = {'uniprot': row.uniprot, 'gene': row.gene, 'organism': row.organism,
                'cosine_similarity': row.cosine_similarity,
                'pfam': family_ids, 'family': family_names}
        if not found:
            rows.append(base | {'native_reaction': '', 'native_substrates': '',
                                'ec': '', 'rhea': '',
                                'evidence': 'no curated catalytic activity',
                                'pubmed': '', 'experimental': False})
            continue
        for statement in found:
            rows.append(base | {
                'native_reaction': statement['native_reaction'],
                'native_substrates': ' + '.join(substrates(statement['native_reaction'])),
                'ec': statement['ec'], 'rhea': statement['rhea'],
                'evidence': statement['evidence'], 'pubmed': statement['pubmed'],
                'experimental': 'ECO:0000269' in statement['evidence']})

    table = pd.DataFrame(rows, columns=[
        'uniprot', 'gene', 'organism', 'cosine_similarity', 'pfam', 'family',
        'native_reaction', 'native_substrates', 'ec', 'rhea', 'evidence', 'pubmed', 'experimental'])
    table.to_csv(RES / 'native_substrates.tsv', sep='\t', index=False)

    # ---- collapse to families
    groups = defaultdict(lambda: {'members': set(), 'genes': [], 'best': -1.0,
                                  'substrates': Counter(), 'experimental': 0,
                                  'ecs': set()})
    for row in table.itertuples():
        for accession, name in zip(str(row.pfam).split(';'), str(row.family).split(';')):
            if not accession.strip():
                continue
            entry = groups[(accession.strip(), name.strip())]
            entry['members'].add(row.uniprot)
            if row.gene and str(row.gene) != 'nan':
                entry['genes'].append(str(row.gene).split()[0])
            entry['best'] = max(entry['best'], row.cosine_similarity)
            entry['experimental'] += bool(row.experimental)
            if row.native_substrates:
                entry['substrates'][row.native_substrates] += 1
            if row.ec:
                entry['ecs'].add(row.ec)

    families = pd.DataFrame([
        {'pfam': accession, 'family': name or accession,
         'n_enzymes': len(v['members']),
         'best_cosine': v['best'],
         'n_experimental_statements': v['experimental'],
         'ec_numbers': ';'.join(sorted(v['ecs'])[:6]),
         'example_genes': ';'.join(list(dict.fromkeys(v['genes']))[:6]),
         'representative_native_substrates':
             v['substrates'].most_common(1)[0][0] if v['substrates'] else '',
         'members': ';'.join(sorted(v['members'])[:10])}
        for (accession, name), v in groups.items()], columns=[
            'pfam', 'family', 'n_enzymes', 'best_cosine', 'n_experimental_statements',
            'ec_numbers', 'example_genes', 'representative_native_substrates', 'members'])
    families = families.sort_values(['best_cosine', 'n_enzymes'],
                                    ascending=[False, False]).reset_index(drop=True)
    families.to_csv(RES / 'enzyme_families.tsv', sep='\t', index=False)

    with_activity = table[table.native_reaction != '']
    print(f'\n{table.uniprot.nunique()} candidates | '
          f'{with_activity.uniprot.nunique()} with a curated native reaction | '
          f'{table[table.experimental.astype(bool)].uniprot.nunique()} with EXPERIMENTAL evidence '
          f'(ECO:0000269) | {len(families)} families')

    pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 60)
    print('\nFamilies:')
    print(families.head(12)[['best_cosine', 'n_enzymes', 'family', 'example_genes',
                             'representative_native_substrates']].to_string(index=False))
    print('\nNative substrates of the top candidates:')
    print(with_activity.drop_duplicates('uniprot')
          .head(10)[['cosine_similarity', 'gene', 'native_substrates', 'pubmed']]
          .to_string(index=False))


if __name__ == '__main__':
    sys.exit(main())
