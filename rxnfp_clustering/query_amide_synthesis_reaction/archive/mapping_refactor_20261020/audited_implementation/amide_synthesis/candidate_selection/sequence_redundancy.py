"""Reproducible sequence redundancy removal, independent of reaction evidence."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import re
import subprocess

from Bio import SeqIO


def cluster_proteins(records, directory, executable, identity=0.95, threads=4):
    """Exact sequences first, then representative-based full-length CD-HIT.

    Identity is identical residues / full shorter-sequence length (CD-HIT -G 1).
    Both sequences need >=90% aligned coverage. These are redundancy groups,
    not assertions that members are duplicate genes or have identical activity.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    by_sequence = defaultdict(list)
    for record in records:
        by_sequence[record['sequence']].append(record)
    exact = {}
    exact_seed = {}
    for sequence, group in by_sequence.items():
        representative = min(group, key=lambda r: (not r['reviewed'], r['fragment'], r['accession']))
        accession = representative['accession']
        exact[accession] = representative
        for record in group:
            exact_seed[record['accession']] = accession
    fasta = directory / 'exact_representatives.fasta'
    ordered = sorted(exact.values(), key=lambda r: (-r['length'], not r['reviewed'], r['accession']))
    fasta.write_text(''.join(f">{r['accession']}\n{r['sequence']}\n" for r in ordered))
    output = directory / f'identity{round(identity * 100)}.fasta'
    mapping = {}
    command = []
    if exact:
        command = [str(executable), '-i', str(fasta), '-o', str(output),
                   '-c', str(identity), '-G', '1', '-aL', '0.9', '-aS', '0.9',
                   '-g', '1', '-n', '5', '-d', '0', '-p', '1',
                   '-T', str(min(4, max(1, threads))), '-M', '2000']
        with (directory / f'identity{round(identity * 100)}.log').open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        clusters = []
        current = []
        for line in output.with_suffix(output.suffix + '.clstr').read_text().splitlines():
            if line.startswith('>Cluster'):
                if current:
                    clusters.append(current)
                current = []
            else:
                match = re.search(r'>(\S+)\.\.\.\s+(.*)$', line)
                if not match:
                    raise ValueError('Unparseable CD-HIT membership line')
                current.append((match[1], match[2]))
        if current:
            clusters.append(current)
        for cluster in clusters:
            seeds = [accession for accession, detail in cluster if detail == '*']
            if len(seeds) != 1:
                raise ValueError('CD-HIT cluster does not have exactly one representative')
            for accession, detail in cluster:
                mapping[accession] = {'representative': seeds[0], 'alignment': detail}
        if set(mapping) != set(exact):
            raise ValueError('CD-HIT omitted or duplicated input sequences')
        if {r.id for r in SeqIO.parse(output, 'fasta')} != {v['representative'] for v in mapping.values()}:
            raise ValueError('CD-HIT representatives/members disagree')
    else:
        output.write_text('')
    members = []
    for record in records:
        seed = exact_seed[record['accession']]
        members.append({
            'accession': record['accession'], 'exact_representative': seed,
            'representative': mapping[seed]['representative'],
            'alignment': mapping[seed]['alignment'],
            'taxon_id': record['taxon_id'], 'proteomes': record['proteomes'],
        })
    return {'raw_accessions': len(records), 'exact_sequences': len(exact),
            'representatives': len({v['representative'] for v in mapping.values()}),
            'members': members, 'command': command, 'identity': identity,
            'coverage_both': 0.9, 'fasta': str(output)}
