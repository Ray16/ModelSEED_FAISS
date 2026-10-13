"""Resolve Pfam accessions to family names, cached on disk.

Family is the right unit for reading this shortlist: sixteen proteins from one
family is a convergence, not sixteen independent hits. So every table that lists
candidates carries its family, rather than leaving it to a separate file.

UniProt gives Pfam accessions; the human-readable name comes from the InterPro
API, looked up once per accession and cached.
"""
from __future__ import annotations

import json
import os
import time

import pandas as pd

from .paths import CACHE
from .http import get_text

FAMILY_CACHE = CACHE / 'pfam_names.json'

# Second domains of proteins already counted under their primary family; listing
# them separately would double-count the same enzymes.
SECONDARY_DOMAINS = {
    'Ferric iron reductase FhuF-like transporter',
    'Glutamine synthetase N-terminal domain',
    'Glutamine synthetase, beta-Grasp domain',
    'Mur ligase middle domain',
    'Mur ligase family, catalytic domain',
    'Acyl-CoA thioester hydrolase/BAAT N-terminal region',
}


def pfam_names(accessions):
    """Accession -> family name; offline mode uses cached names and raw IDs."""
    known = json.loads(FAMILY_CACHE.read_text()) if FAMILY_CACHE.exists() else {}
    missing = [a for a in dict.fromkeys(accessions) if a and a not in known]
    if os.environ.get('AMIDE_SYNTHESIS_OFFLINE') == '1':
        return known
    changed = False
    for accession in missing:
        url = f'https://www.ebi.ac.uk/interpro/api/entry/pfam/{accession}/'
        text, _, failed = get_text(url, timeout=30)
        try:
            if failed:
                continue  # a transient outage must not become a permanent cache entry
            known[accession] = json.loads(text)['metadata']['name']['name']
            changed = True
        except Exception as exc:
            print(f'  ! Pfam {accession}: {exc}', flush=True)
        time.sleep(0.2)
    if changed:
        FAMILY_CACHE.parent.mkdir(parents=True, exist_ok=True)
        FAMILY_CACHE.write_text(json.dumps(known, indent=2))
    return known


def split_accessions(value) -> list[str]:
    if not isinstance(value, str):
        return []
    return [a.strip() for a in value.split(';') if a.strip() and a.strip() != 'nan']


def attach(table: pd.DataFrame, pfam_by_uniprot: dict) -> pd.DataFrame:
    """Add `pfam`, `family` (primary) and `family_all` columns to a candidate table."""
    all_accessions = []
    for uniprot in table.uniprot:
        all_accessions.extend(split_accessions(pfam_by_uniprot.get(uniprot)))
    names = pfam_names(all_accessions)

    primary, every, accs = [], [], []
    for uniprot in table.uniprot:
        accessions = split_accessions(pfam_by_uniprot.get(uniprot))
        labels = [names.get(a) or a for a in accessions]
        accs.append(';'.join(accessions))
        every.append(';'.join(labels))
        # the primary family is the first label that is not a secondary domain
        main = next((l for l in labels if l not in SECONDARY_DOMAINS), labels[0] if labels else '')
        primary.append(main)
    out = table.copy()
    out['pfam'] = accs
    out['family'] = primary
    out['family_all'] = every
    return out
