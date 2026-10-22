#!/usr/bin/env python
"""Exact-sequence identity lookup through UniParc.

The 50-member assayed panel is the most valuable set of rows in the whole
table - the only proteins here with measured titre on the nylon-6,6 diad - and
the least annotated, because they arrive as NCBI accessions from a separate
project. Accession-based resolution reached only 10 of 48.

UniParc is a sequence-identity archive: every protein sequence ever deposited
in any member database, keyed by its CRC64 checksum. Looking a sequence up
there answers "has this exact protein been seen anywhere else, under any
accession" without any alignment or inference - it is identity, not homology.

CRC64 CORRECTNESS IS LOAD-BEARING. A wrong checksum returns zero hits, which
is indistinguishable from "this sequence is not in UniParc" - a silent false
negative across the whole panel. The first implementation here was wrong in
exactly that way and reported 0/48, which looked like a finding. It is now the
canonical Swiss-Prot split-32-bit algorithm and is validated in the tests
against a checksum UniProt publishes for a known entry.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote

from .http import get_text

API = 'https://rest.uniprot.org/uniparc/search'
ENTRY = 'https://rest.uniprot.org/uniparc/{}.json'
THREADS = 6
POLY64REVH = 0xd8000000


def _tables():
    high, low = [0] * 256, [0] * 256
    for i in range(256):
        partl, parth = i, 0
        for _ in range(8):
            rflag = partl & 1
            partl >>= 1
            if parth & 1:
                partl |= (1 << 31)
            parth >>= 1
            if rflag:
                parth ^= POLY64REVH
        high[i], low[i] = parth, partl
    return high, low


_HIGH, _LOW = _tables()


def crc64(sequence: str) -> str:
    """Swiss-Prot CRC64 of a protein sequence, as UniProt publishes it."""
    crch = crcl = 0
    for char in sequence:
        shr = (crch & 0xFF) << 24
        t1h = crch >> 8
        t1l = (crcl >> 8) | shr
        index = (crcl ^ ord(char)) & 0xFF
        crch = t1h ^ _HIGH[index]
        crcl = t1l ^ _LOW[index]
    return '%08X%08X' % (crch, crcl)


def clean_sequence(value):
    """A usable protein sequence, or None.

    A missing sequence arrives as a float NaN from pandas, which is not
    iterable - the kind of input that raises deep inside the checksum rather
    than being rejected at the boundary.
    """
    if value is None or isinstance(value, float):
        return None
    text = str(value).strip().upper()
    if not text or text == 'NAN':
        return None
    return ''.join(c for c in text if c.isalpha())


def lookup(sequence, fetch=get_text):
    """UniParc record for an exact sequence, or None.

    Two steps on purpose. The search endpoint resolves a checksum to a UPI but
    returns no cross-references, which is the entire point of the lookup - so
    the entry is then fetched by UPI. Doing only the search silently yields a
    record with every annotation field blank, which looks like "found but
    unannotated" rather than "not fetched".
    """
    sequence = clean_sequence(sequence)
    if not sequence:
        return None
    checksum = crc64(sequence)
    url = (f'{API}?query={quote(f"checksum:{checksum}")}'
           '&format=json&size=1')
    text, _, failed = fetch(url, timeout=60)
    if failed or not text:
        return None
    try:
        results = json.loads(text).get('results') or []
    except ValueError:
        return None
    if not results:
        return None
    upi = results[0].get('uniParcId')
    entry = results[0]
    if upi:
        detail, _, detail_failed = fetch(ENTRY.format(upi), timeout=60)
        if not detail_failed and detail:
            try:
                entry = json.loads(detail)
            except ValueError:
                pass
    refs = entry.get('uniParcCrossReferences') or []
    uniprot = [r.get('id') for r in refs
               if r.get('database', '').startswith('UniProt')
               and r.get('active')]
    organisms = [r.get('organism', {}).get('scientificName') for r in refs
                 if r.get('organism', {}).get('scientificName')]
    genes = [r.get('geneName') for r in refs if r.get('geneName')]
    proteins = [r.get('proteinName') for r in refs if r.get('proteinName')]
    return {
        # From the search result: the detail document does not repeat it.
        'uniparc_id': upi or entry.get('uniParcId'),
        'uniparc_crc64': checksum,
        'uniparc_uniprot': ';'.join(dict.fromkeys(uniprot))[:200],
        'uniparc_organism': ';'.join(dict.fromkeys(organisms))[:200],
        'uniparc_gene': ';'.join(dict.fromkeys(genes))[:120],
        'uniparc_protein_name': ';'.join(dict.fromkeys(proteins))[:200],
        'uniparc_n_crossrefs': len(refs),
    }


def lookup_many(sequences, fetch=get_text, threads=THREADS):
    """{key: record} for a mapping of key -> sequence."""
    items = [(k, v) for k, v in sequences.items() if clean_sequence(v)]
    if not items:
        return {}

    def one(item):
        key, sequence = item
        return key, lookup(sequence, fetch=fetch)

    out = {}
    with ThreadPoolExecutor(max_workers=threads) as pool:
        for key, record in pool.map(one, items):
            if record:
                out[key] = record
    return out
