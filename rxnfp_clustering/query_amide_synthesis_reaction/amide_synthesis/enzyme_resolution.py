"""Reaction-to-enzyme evidence routes and deterministic protein selection.

The search callable is supplied by the caller, so retrieval decisions can be
verified against offline fixtures without UniProt or a ModelSEED checkout.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# UniProt's protein-existence ladder, best evidence first. Used as a ranking
# key so an experimentally observed protein outranks a genome prediction.
EXISTENCE = {'Evidence at protein level': 0, 'Evidence at transcript level': 1,
             'Inferred from homology': 2, 'Predicted': 3, 'Uncertain': 4}

PER_REACTION = 8          # enzymes kept per shortlisted reaction
MAX_EUKARYOTE = 3         # of those slots, at most this many non-prokaryotes

# Cofactors and generic species carry no information about which enzyme acts.
GENERIC = {'atp', 'adp', 'amp', 'ppi', 'phosphate', 'h2o', 'h+', 'coa', 'nad',
           'nadh', 'nadp', 'nadph', 'co2', 'o2', 'nh3', 'pi', 'diphosphate',
           'water', 'proton', 'glutamate', 'glutamine', 'citrate'}
PRODUCT_TERM = re.compile(r'\(\d+(?:\.\d+)?\)\s+([^\[]+)\[')


def product_names(definition: str):
    """Distinctive product names from a ModelSEED reaction definition."""
    if not isinstance(definition, str) or '=>' not in definition:
        return []
    right = re.split(r'<?=>', definition)[-1]
    names = []
    for raw in PRODUCT_TERM.findall(right):
        name = raw.strip()
        if name.lower() in GENERIC or len(name) < 8:
            continue
        names.append(name)
        # "pre-bisucaberin" -> "bisucaberin"; the mature product is what UniProt names
        if name.lower().startswith('pre-'):
            names.append(name[4:])
    return list(dict.fromkeys(names))

def is_prokaryote(entry):
    lineage = entry.get('Taxonomic lineage', '')
    return 'Bacteria' in lineage or 'Archaea' in lineage


def evidence_strength(route):
    if route.startswith('Rhea'):
        return 'exact_reaction'
    if route.startswith('EC'):
        return 'activity_ec'
    return 'name_match'


def rank_entries(entries, limit=PER_REACTION, max_eukaryote=MAX_EUKARYOTE):
    """Keep distinct proteins, strongest evidence first, with taxonomy slot fill.

    The eukaryote cap applies only while prokaryotes can fill the remaining
    slots. Duplicate accessions retain their strongest evidence route.
    """
    if limit < 0 or max_eukaryote < 0:
        raise ValueError('protein limits must be non-negative')
    ranked = sorted(entries, key=lambda entry: (
        not entry.get('_route', '').startswith('Rhea'),
        entry.get('_route', '').startswith('Name'),
        entry.get('Reviewed', '') != 'reviewed',
        not is_prokaryote(entry),
        EXISTENCE.get(entry.get('Protein existence', ''), 9),
        entry.get('Entry', ''),
        entry.get('_route', ''),
    ))
    unique, seen = [], set()
    for entry in ranked:
        if entry['Entry'] not in seen:
            seen.add(entry['Entry'])
            unique.append(entry)
    kept, taken, eukaryotes = [], set(), 0
    for entry in unique:
        if len(kept) == limit:
            break
        if not is_prokaryote(entry):
            if eukaryotes == max_eukaryote:
                continue
            eukaryotes += 1
        kept.append(entry)
        taken.add(entry['Entry'])
    for entry in unique:
        if len(kept) == limit:
            break
        if entry['Entry'] not in taken:
            kept.append(entry)
            taken.add(entry['Entry'])
    return kept


def select_reactions(hits, floor):
    """Best framing per reaction, admitted only at or above the OOD floor."""
    useful = (hits.sort_values('cosine_similarity', ascending=False, kind='stable')
                  .drop_duplicates('rxn_id'))
    return useful[useful.cosine_similarity >= floor].copy()


@dataclass
class Resolution:
    entries: list[dict]
    route: str
    pool_size: int
    failed: bool
    truncated: tuple[str, ...]
    has_identifier: bool

    @property
    def status(self):
        if self.entries:
            return 'partial_lookup' if self.failed else 'resolved'
        if self.failed:
            return 'lookup_failed'
        return 'no_hits' if self.has_identifier else 'no_identifier'


def resolve_reaction(hit, aliases, metanetx, *, search_client):
    """Collect Rhea AND EC evidence, then try product names if no hits arrived.

    Search failures and saturated pools are distinct from successful empty
    lookups. Copies carry route metadata without mutating client-owned rows.
    """
    from_modelseed = set(aliases.get(hit.rxn_id, {}).get('Rhea', ()))
    from_metanetx = set(metanetx.get(hit.rxn_id, ()))
    rhea = sorted(from_modelseed | from_metanetx, key=int)[:6]
    source = ('ModelSEED+MetaNetX' if from_modelseed and from_metanetx
              else 'MetaNetX' if from_metanetx else 'ModelSEED')
    ec = [e for e in str(hit.ec).split(';')
          if e and e != 'nan' and '-' not in e and not e.split('.')[-1].isalpha()]
    found, routes, cuts = [], [], []
    failed = False
    queried = False

    def collect(query, label, route, **kwargs):
        nonlocal failed, queried
        queried = True
        rows, errored, saturated = search_client(query, **kwargs)
        failed |= errored
        if saturated:
            cuts.append(label)
        found.extend(dict(row, _route=route) for row in rows)
        return bool(rows)

    if rhea and collect(' OR '.join(f'rhea:{r}' for r in rhea), 'Rhea',
                        f"Rhea[{source}]:{','.join(rhea)}"):
        routes.append(f'Rhea/{source}')
    if ec and collect(' OR '.join(f'ec:{e}' for e in ec), 'EC', f"EC:{','.join(ec)}"):
        routes.append('EC')
    if not found:
        for name in product_names(getattr(hit, 'definition', '')):
            if collect(f'protein_name:"{name}" AND (synthase OR synthetase OR ligase)',
                       'Name', f'Name:{name}', size=25):
                routes.append('Name')
                break
    result = Resolution(rank_entries(found), '+'.join(routes), len(found), failed,
                        tuple(dict.fromkeys(cuts)), queried)
    if not result.route:
        result.route = result.status.replace('_', ' ')
    return result
