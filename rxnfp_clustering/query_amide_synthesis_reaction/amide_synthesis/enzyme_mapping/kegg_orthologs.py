#!/usr/bin/env python
"""Resolve reactions to enzymes through KEGG orthology, where UniProt cannot.

THE GAP THIS CLOSES. ModelSEED carries no reaction->enzyme mapping, so every
route here is a pivot through some other database. All of them went through
UniProt: a reaction's Rhea id, its EC, or its name. A reaction with none of
those is recorded as having no enzyme at all, and 153 of 480 were (146
after this route). But KEGG
assigns ORTHOLOGY GROUPS to reactions directly, and a KO carries both an EC
and the genes that realise it - a bridge that does not pass through UniProt's
reaction annotation and so does not fail where it fails.

Measured on the 175 unresolved reaction records: 24 carry a KEGG `R` alias and
10 of those have a KO. Those 10 are not unresolvable after all; three of them
are carnosine synthase (EC 6.3.2.11), which ligates beta-alanine onto the free
alpha-amino group of histidine - ATP-dependent acylation of a free amine, the
target chemistry - and the pipeline simply never connected the reaction to the
enzyme because ModelSEED left the EC field empty.

WHAT THIS IS NOT, MEASURED RATHER THAN ASSUMED. It is tempting to read the
above as a new retrieval channel. It is not. Across all 91 screened reactions
that have a KO, the route reaches 89 distinct ECs, and exactly 3 of them are
absent from the 55,020 candidates already held: a peptidase, alanine racemase,
and an acid--thiol ligase. None forms an amide. The pipeline already holds the
chemistry; what it lacked was the LINK between a reaction and the enzymes that
perform it. So this is an annotation fix that corrects a reported recall
number and adds the tail of two relevant EC classes - not a way to find
chemistry that was missing.

Accordingly the candidates it admits are `activity_ec`: their evidence really
is a broader EC match, and only the route that found the EC is new. They are
NOT exempt from the family verdict, unlike `active_homolog` - that exemption
exists because its evidence is measured activity that contradicts annotation,
whereas this route's evidence IS annotation.
"""
from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request

import pandas as pd

from amide_synthesis.infrastructure.paths import RES

KEGG = 'https://rest.kegg.jp/'
UNIPROT = 'https://rest.uniprot.org/uniprotkb/search'
# ModelSEED ships the alias table; no network needed for the reaction pivot.
ALIASES = ('../../ModelSEEDDatabase/Biochemistry/Aliases/'
           'Unique_ModelSEED_Reaction_Aliases.txt')
PAUSE = 0.35
RETRIES = 3
# A KEGG reaction id, to keep malformed alias rows out of the query.
KEGG_REACTION = re.compile(r'^R\d{5}$')
EC_IN_NAME = re.compile(r'EC:([0-9.\- ]+)')
# A partial EC (`2.3.1.-`) names a sub-subclass, not an activity, and
# expanding one would pull in every member of it.
COMPLETE_EC = re.compile(r'^\d+\.\d+\.\d+\.\d+$')


def _get(path, timeout=90):
    """A KEGG REST call, retried; '' when the endpoint has nothing."""
    for attempt in range(RETRIES):
        try:
            with urllib.request.urlopen(KEGG + path, timeout=timeout) as handle:
                return handle.read().decode('utf-8', 'replace').strip()
        except Exception:
            if attempt == RETRIES - 1:
                return ''
            time.sleep(1)
    return ''


def kegg_aliases(path=None):
    """ModelSEED reaction id -> KEGG reaction id, from the shipped table."""
    source = RES / '..' / ALIASES if path is None else path
    frame = pd.read_csv(source, sep='\t', low_memory=False)
    frame.columns = ['rxn_id', 'external', 'source']
    kegg = frame[frame.source.eq('KEGG')
                 & frame.external.astype(str).str.match(KEGG_REACTION)]
    return kegg[['rxn_id', 'external']].drop_duplicates('rxn_id')


def orthologs(kegg_ids):
    """KEGG reaction id -> the KO groups assigned to it."""
    rows = []
    for kegg_id in kegg_ids:
        payload = _get(f'link/ko/rn:{kegg_id}')
        for line in payload.splitlines() if payload else []:
            if '\t' in line:
                rows.append({'kegg_rn': kegg_id,
                             'ko': line.split('\t')[1].replace('ko:', '')})
        time.sleep(PAUSE)
    return pd.DataFrame(rows, columns=['kegg_rn', 'ko'])


def ortholog_ecs(kos):
    """KO -> its name and the COMPLETE ECs it declares."""
    rows = []
    for ko in kos:
        info = _get(f'get/ko:{ko}')
        name = next((line[12:].strip() for line in info.splitlines()
                     if line.startswith('NAME')), '')
        found = EC_IN_NAME.search(name)
        ecs = [e for e in (found.group(1).split() if found else [])
               if COMPLETE_EC.match(e)]
        rows.append({'ko': ko, 'ko_name': name.split('[')[0].strip(),
                     'ecs': ';'.join(ecs)})
        time.sleep(PAUSE)
    return pd.DataFrame(rows, columns=['ko', 'ko_name', 'ecs'])


def proteins_for_ec(ec, size=500):
    """Every UniProt accession curated to an EC, up to `size`."""
    query = urllib.parse.quote(f'(ec:{ec})')
    url = f'{UNIPROT}?query={query}&size={size}&fields=accession'
    try:
        with urllib.request.urlopen(url, timeout=120) as handle:
            body = json.load(handle)
    except Exception:
        return []
    return [row['primaryAccession'] for row in body.get('results', [])]


def unresolved_reactions():
    """The screened reactions that no UniProt pivot could resolve."""
    path = RES / 'reactions_without_enzymes.tsv'
    if not path.exists():
        return pd.DataFrame(columns=['rxn_id'])
    return pd.read_csv(path, sep='\t', low_memory=False)


def main():
    from amide_synthesis.infrastructure.paths import ensure_directories
    from amide_synthesis.enzyme_mapping.enrich import fetch
    from amide_synthesis.candidate_selection.master_table import RENAME
    ensure_directories()

    unresolved = unresolved_reactions()
    aliases = kegg_aliases()
    pivot = unresolved.merge(aliases, on='rxn_id', how='inner')
    print(f'{unresolved.rxn_id.nunique()} unresolved reactions, '
          f'{pivot.rxn_id.nunique()} with a KEGG alias', flush=True)

    # Captured BEFORE the join below, which is an inner join and would
    # otherwise make this count the reactions that have an ORTHOLOG rather
    # than the ones that have an alias - two different numbers (24 and 10),
    # reported under the alias name.
    with_alias = int(pivot.rxn_id.nunique())

    links = orthologs(sorted(pivot.external.unique()))
    pivot = pivot.merge(links, left_on='external', right_on='kegg_rn',
                        how='inner')
    if pivot.empty:
        print('no KEGG ortholog for any unresolved reaction', flush=True)
        return

    info = ortholog_ecs(sorted(pivot.ko.unique()))
    pivot = pivot.merge(info, on='ko', how='left')
    resolved = pivot.rxn_id.nunique()
    print(f'{resolved} of them have a KEGG ortholog '
          f'({pivot.ko.nunique()} distinct KOs)', flush=True)
    pivot.to_csv(RES / 'kegg_ortholog_links.tsv', sep='\t', index=False)

    # One protein set per EC the orthologs declare. A reaction keeps the EC
    # that reached it, so a reader can see WHY each protein is here.
    wanted = {}
    for _, row in pivot.iterrows():
        for ec in str(row.ecs or '').split(';'):
            if ec:
                wanted.setdefault(ec, set()).add(row.rxn_id)
    accessions, provenance = set(), {}
    for ec, reactions in sorted(wanted.items()):
        found = proteins_for_ec(ec)
        print(f'  EC {ec:12s} {len(found):5d} UniProt entries '
              f'({len(reactions)} reaction(s))', flush=True)
        for accession in found:
            accessions.add(accession)
            provenance.setdefault(accession, (ec, sorted(reactions)[0]))
        time.sleep(PAUSE)

    rows, failed = fetch(sorted(accessions))
    frame = pd.DataFrame(rows).rename(columns=RENAME)
    if frame.empty:
        print('no records retrieved', flush=True)
        return
    frame['kegg_ortholog_ec'] = frame.uniprot.map(
        lambda a: provenance.get(a, ('', ''))[0])
    frame['kegg_source_rxn'] = frame.uniprot.map(
        lambda a: provenance.get(a, ('', ''))[1])
    frame['discovery_route'] = 'kegg_ortholog'
    # The evidence really is a broader EC match; only the route that found
    # the EC is new. Reusing the tier keeps the existing weight and quota.
    frame['evidence_strength'] = 'activity_ec'
    frame['discovery_tier'] = 'kegg_ortholog_of_unresolved_reaction'
    frame.to_csv(RES / 'kegg_ortholog_candidates.tsv', sep='\t', index=False)

    summary = {
        'unresolved_reactions': int(unresolved.rxn_id.nunique()),
        'with_a_kegg_alias': with_alias,
        'resolved_by_an_ortholog': int(resolved),
        'distinct_kos': int(pivot.ko.nunique()),
        'ecs_expanded': sorted(wanted),
        'candidates': int(len(frame)),
        'failed_batches': failed,
        'note': 'An annotation fix, not a new retrieval channel. Across all '
                'screened reactions with a KO this route reaches 89 ECs, of '
                'which 3 are new to the candidate set and none forms an '
                'amide. What was missing was the LINK from a reaction to the '
                'enzymes that perform it, not the enzymes.',
    }
    print(json.dumps(summary, indent=2), flush=True)
    (RES / 'kegg_ortholog_summary.json').write_text(
        json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    main()
