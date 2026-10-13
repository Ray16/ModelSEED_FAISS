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
