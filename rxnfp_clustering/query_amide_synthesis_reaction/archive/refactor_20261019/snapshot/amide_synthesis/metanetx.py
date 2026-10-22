"""Recover Rhea reaction IDs for ModelSEED reactions that lack them, via MetaNetX.

ModelSEED's own alias table carries a Rhea ID for only ~61% of the reactions this
analysis ranks, and no EC for a further 23%. Those reactions are unreachable by
any structured enzyme lookup, so they silently yield no candidate at all - a
recall failure indistinguishable, in the output, from "no enzyme exists". It
nearly cost the whole result: rxn21314 (BibC, the top-scoring hit) carries only
MetaCyc and MetaNetX identifiers.

MetaNetX reconciles reaction identifiers across databases. Its reac_xref table
maps every source ID - including `seed.reaction:rxnNNNNN` - onto a shared MNXR
identifier, so pivoting through MNXR recovers Rhea IDs that ModelSEED does not
record. For rxn21314 this yields rhea:79295-79298, which is exactly the Rhea
reaction cited in BibC's own UniProt catalytic-activity annotation.

The 80 MB table is downloaded once into its own directory and parsed by explicit
path; nothing from it is executed. The distilled seed->Rhea map is cached as
JSON so later runs skip the parse.
"""
from __future__ import annotations

import json
from collections import defaultdict

from .paths import CACHE
from .http import get

DOWNLOAD = CACHE / 'metanetx_download'               # untrusted downloaded data
XREF = DOWNLOAD / 'reac_xref.tsv'
MAP_CACHE = CACHE / 'metanetx_seed_to_rhea.json'
FOREIGN_CACHE = CACHE / 'metanetx_foreign_to_rhea.json'

# ModelSEED's alias table names these; MetaNetX indexes each under its own
# prefix. Pivoting through them reaches reactions whose SEED id is absent from
# MetaNetX even though a sibling identifier is present - measured: 14 of the 77
# unresolved MetaCyc-bearing reactions gain a Rhea this way, including
# rxn19370, which looked undeposited under the SEED pivot alone.
FOREIGN_PREFIXES = {
    'MetaCyc': 'metacyc.reaction:',
    'KEGG': 'kegg.reaction:',
    'BiGG': 'bigg.reaction:',
}
URL = 'https://www.metanetx.org/ftp/latest/reac_xref.tsv'


def ensure_download(timeout=600) -> bool:
    """Fetch reac_xref.tsv if absent. Returns False if unavailable (offline)."""
    if XREF.exists() and XREF.stat().st_size > 1_000_000:
        return True
    DOWNLOAD.mkdir(parents=True, exist_ok=True)
    print(f'downloading MetaNetX reac_xref.tsv (~80 MB)', flush=True)
    body, _, failed = get(URL, timeout=timeout)
    if failed or body is None:
        print('  ! MetaNetX unavailable; Rhea IDs limited to ModelSEED aliases',
              flush=True)
        return False
    XREF.write_bytes(body)
    return True


def seed_to_rhea(refresh: bool = False) -> dict[str, list[str]]:
    """ModelSEED reaction id -> Rhea numeric ids, pivoted through MNXR."""
    if MAP_CACHE.exists() and not refresh:
        return json.loads(MAP_CACHE.read_text())
    if not ensure_download():
        return {}

    seed_by_mnx: dict[str, list[str]] = defaultdict(list)
    rhea_by_mnx: dict[str, list[str]] = defaultdict(list)
    with XREF.open() as handle:
        for line in handle:
            if line.startswith('#'):
                continue
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 2:
                continue
            source, mnx = parts[0], parts[1]
            if mnx in ('EMPTY', ''):
                continue
            if source.startswith('seed.reaction:'):
                seed_by_mnx[mnx].append(source.split(':', 1)[1])
            elif source.startswith('rhea:'):
                rhea_by_mnx[mnx].append(source.split(':', 1)[1])

    mapping: dict[str, list[str]] = {}
    for mnx, seeds in seed_by_mnx.items():
        rheas = rhea_by_mnx.get(mnx)
        if not rheas:
            continue
        for seed in seeds:
            mapping.setdefault(seed, [])
            for rhea in rheas:
                if rhea not in mapping[seed]:
                    mapping[seed].append(rhea)
    for seed in mapping:
        mapping[seed] = sorted(mapping[seed], key=int)

    MAP_CACHE.write_text(json.dumps(mapping))
    print(f'MetaNetX: {len(mapping):,} ModelSEED reactions carry a Rhea ID',
          flush=True)
    return mapping


def foreign_to_rhea(refresh: bool = False) -> dict[str, dict[str, list[str]]]:
    """{namespace: {foreign reaction id -> Rhea ids}} pivoted through MNXR.

    The SEED pivot only helps when MetaNetX knows the reaction under its
    `seed.reaction:` identifier. Many ModelSEED reactions are cross-referenced
    to MetaCyc or KEGG but not registered under their own SEED id, so pivoting
    from the sibling identifier reaches reactions the SEED pivot cannot.
    """
    if FOREIGN_CACHE.exists() and not refresh:
        return json.loads(FOREIGN_CACHE.read_text())
    if not ensure_download():
        return {}

    rhea_by_mnx: dict[str, list[str]] = defaultdict(list)
    foreign_by_id: dict[str, dict[str, list[str]]] = {
        name: defaultdict(list) for name in FOREIGN_PREFIXES}
    with XREF.open() as handle:
        for line in handle:
            if line.startswith('#'):
                continue
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 2:
                continue
            source, mnx = parts[0], parts[1]
            if mnx in ('EMPTY', ''):
                continue
            if source.startswith('rhea:'):
                rhea_by_mnx[mnx].append(source.split(':', 1)[1])
                continue
            for name, prefix in FOREIGN_PREFIXES.items():
                if source.startswith(prefix):
                    foreign_by_id[name][source.split(':', 1)[1]].append(mnx)
                    break

    out: dict[str, dict[str, list[str]]] = {}
    for name, by_id in foreign_by_id.items():
        resolved = {}
        for foreign, mnxs in by_id.items():
            rheas = sorted({r for mnx in mnxs for r in rhea_by_mnx.get(mnx, ())},
                           key=int)
            if rheas:
                resolved[foreign] = rheas
        out[name] = resolved
    FOREIGN_CACHE.write_text(json.dumps(out))
    print('MetaNetX foreign pivot: '
          + ', '.join(f'{k} {len(v):,}' for k, v in out.items()), flush=True)
    return out


def rhea_for_reaction(rxn_id, aliases, seed_map, foreign_map):
    """Every Rhea id reachable for one ModelSEED reaction, by any pivot."""
    found = set(seed_map.get(rxn_id, ()))
    entry = aliases.get(rxn_id, {})
    for name, table in (foreign_map or {}).items():
        for foreign in entry.get(name, ()):
            found.update(table.get(foreign, ()))
    return sorted(found, key=int)
