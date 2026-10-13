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


# A reaction the screen has PROVEN acylates a free amine, and which scores this
# highly on nylon-like substrate chemistry, is not out-of-distribution in any
# useful sense - whatever the cofactor-dominated cosine says.
SUBSTRATE_DISTANCE_RESCUE = 0.40


def select_reactions(hits, floor, rescue=SUBSTRATE_DISTANCE_RESCUE):
    """Best framing per reaction, admitted on cosine OR on substrate chemistry.

    The OOD floor is a cosine threshold, and cosine here is dominated by the
    cofactor rather than the amide bond. Applying it alone was the third and
    last place the embedding silently gated retrieval: it refused 96 of the
    205 screened reactions, and they were the HIGHEST-scoring ones - every
    folylpolyglutamate synthetase (a processive amide polymerase, substrate
    distance 0.883) and the meso-diaminopimelate Mur ligases (0.877), all of
    which carry a usable EC and would have resolved immediately.

    The floor still does its job for reactions with no independent evidence;
    it just no longer overrules the screen.
    """
    useful = (hits.sort_values('cosine_similarity', ascending=False, kind='stable')
                  .drop_duplicates('rxn_id'))
    admitted = useful.cosine_similarity >= floor
    if 'substrate_distance' in useful.columns:
        admitted |= useful.substrate_distance.fillna(0) >= rescue
    # Stage 03 already decided admission, having applied the mechanism screens.
    # Re-deciding it here against a DIFFERENT threshold meant a reaction could
    # be admitted to the candidate table and then never resolved - which is
    # simply incoherent, and it cost real enzymes: rxn14123 (EC 2.3.2.2) and
    # rxn19001 (EC 6.3.1.10) sit at cosine 0.499 against a floor of 0.5041 and
    # were refused, though their ECs return 100 and 26 UniProt entries.
    # Everything that reached `hits` is resolved; the floor is kept as a LABEL
    # so the OOD information survives without gating anything.
    useful = useful.copy()
    useful['below_ood_floor'] = ~admitted
    return useful


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


def resolve_reaction(hit, aliases, metanetx, *, search_client, foreign=None):
    """Collect Rhea AND EC evidence, then try product names if no hits arrived.

    Search failures and saturated pools are distinct from successful empty
    lookups. Copies carry route metadata without mutating client-owned rows.
    """
    from_modelseed = set(aliases.get(hit.rxn_id, {}).get('Rhea', ()))
    from_metanetx = set(metanetx.get(hit.rxn_id, ()))
    # Pivot from sibling identifiers too. MetaNetX may not know the reaction
    # under its seed.reaction: id while knowing it as MetaCyc or KEGG.
    from_foreign = set()
    for name, table in (foreign or {}).items():
        for key in aliases.get(hit.rxn_id, {}).get(name, ()):
            from_foreign.update(table.get(key, ()))
    rhea = sorted(from_modelseed | from_metanetx | from_foreign, key=int)[:6]
    labels = [n for n, present in (('ModelSEED', from_modelseed),
                                   ('MetaNetX', from_metanetx),
                                   ('MetaNetX-xref', from_foreign)) if present]
    source = '+'.join(labels) or 'ModelSEED'
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
        # Two fields, and every name, unioned rather than first-hit-wins.
        #
        # A reaction known only to MetaCyc has no identifier UniProt indexes -
        # 79 of the 109 unresolved reactions are in exactly that position - so
        # the product name is the only remaining handle. Searching
        # `protein_name` alone is the wrong field for a PRODUCT: an enzyme that
        # makes putrebactin may simply be called "PubC", while the product
        # appears in its CATALYTIC ACTIVITY line. Measured on putrebactin, the
        # two fields return overlapping but different sets (5 and 2), so both
        # are queried. Stopping at the first name that hit also discarded the
        # later names for no reason.
        hit_any = False
        for name in product_names(getattr(hit, 'definition', '')):
            guard = '(synthase OR synthetase OR ligase)'
            for field, tag in (('protein_name', 'Name'),
                               ('cc_catalytic_activity', 'Activity')):
                if collect(f'{field}:"{name}" AND {guard}',
                           tag, f'{tag}:{name}', size=25):
                    hit_any = True
        if hit_any:
            routes.append('Name')
    result = Resolution(rank_entries(found), '+'.join(routes), len(found), failed,
                        tuple(dict.fromkeys(cuts)), queried)
    if not result.route:
        result.route = result.status.replace('_', ' ')
    return result
