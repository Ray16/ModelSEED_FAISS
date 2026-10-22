"""UniProtKB search client: the one place that talks to the protein knowledgebase.

Extracted from stage 04, which is where the reaction -> enzyme mapping lives.
Two defects measured on 2026-10-09 are fixed here rather than at the call site,
because both are properties of *retrieval*, not of this project's chemistry:

1. Saturation was invisible. The old client stopped after 3 pages of 100 and
   returned the truncated list with no indication it had been cut. Broad ECs
   (6.3.2.1, 6.3.2.36, 6.3.1.11) all hit that ceiling, so which proteins a
   reaction got was decided by UniProt's relevance ranking inside a silently
   truncated pool. `search` now reports saturation so callers can say so.

2. Ordering was not deterministic. Entries tied on every ranking key kept
   whatever order UniProt returned, which is relevance-based and not stable
   across runs or query phrasings. Re-running after an unrelated change dropped
   Q7W557 (AlcC) from the shortlist in favour of an equivalent sibling. Results
   are returned sorted by accession so the caller's own sort is total.

The whole of UniProtKB is searched - Swiss-Prot and TrEMBL. Reviewed status is a
ranking key for the caller, never a filter here: an unreviewed enzyme annotated
to the exact reaction is stronger evidence than a reviewed same-EC homologue.
"""
from __future__ import annotations

import csv
import io
import re
import time
from urllib.parse import quote

from amide_synthesis.infrastructure.http import get_text

API = 'https://rest.uniprot.org/uniprotkb/search'

# Everything downstream reads from these fields: sequences for ordering, lineage
# for the prokaryote flag, Pfam/InterPro for family grouping (07), and the
# catalytic-activity statements that 06 re-parses for ECO evidence codes.
FIELDS = ('accession,id,protein_name,gene_names,organism_name,lineage,ec,length,'
          'sequence,cc_catalytic_activity,xref_pdb,protein_existence,reviewed,'
          'cc_cofactor,cc_pathway,cc_subunit,cc_function,go_f,go_p,'
          'xref_pfam,xref_interpro,xref_alphafolddb,ft_binding,ft_act_site,keyword')

PAGE_SIZE = 100
MAX_PAGES = 5          # was 3; see docstring - broad ECs were silently truncated
TIMEOUT = 60
PAUSE = 0.2            # between pages, to stay a polite client


def next_page(headers):
    """Find the next relation, including when it follows another Link relation."""
    link = next((v for k, v in headers.items() if k.lower() == 'link'), '')
    for part in re.split(r',\s*(?=<)', link):
        match = re.match(r'\s*<([^>]+)>\s*;(.*)', part)
        if match and re.search(r'\brel=[\"\']?next(?:[\"\']|\s|;|$)', match[2]):
            return match[1]
    return None


def search(query: str, size: int = PAGE_SIZE, max_pages: int = MAX_PAGES,
           fields: str = FIELDS):
    """Run one UniProt query, following pagination.

    Returns (rows, failed, saturated):
      rows       list of dicts, sorted by accession (deterministic)
      failed     a page could not be retrieved; rows may be partial
      saturated  the page budget ran out, so `rows` is a truncation
    """
    if size < 1 or max_pages < 1:
        raise ValueError('size and max_pages must be positive')
    url = f'{API}?query={quote(f"({query})")}&format=tsv&fields={fields}&size={size}'
    rows, pages, failed = [], 0, False

    while url and pages < max_pages:
        text, headers, this_failed = get_text(url, timeout=TIMEOUT)
        if this_failed:
            failed = True
            break
        rows.extend(csv.DictReader(io.StringIO(text), delimiter='\t'))
        pages += 1
        url = next_page(headers)
        if url and pages < max_pages:
            time.sleep(PAUSE)

    # A next-page cursor still pending when the budget ran out means the result
    # set is larger than what we are returning.
    saturated = bool(url) and pages >= max_pages
    rows.sort(key=lambda row: row.get('Entry', ''))
    return rows, failed, saturated


def total_results(query: str) -> int:
    """How many entries a query matches, without downloading them.

    Used to size a Pfam family: the enrichment test needs the denominator, and
    paging through 400k entries to count them would be absurd and rude.
    """
    url = f'{API}?query={quote(f"({query})")}&format=tsv&fields=accession&size=1'
    _, headers, failed = get_text(url, timeout=TIMEOUT)
    if failed:
        return 0
    for key, value in headers.items():
        if key.lower() == 'x-total-results':
            try:
                return int(value)
            except ValueError:
                return 0
    return 0
