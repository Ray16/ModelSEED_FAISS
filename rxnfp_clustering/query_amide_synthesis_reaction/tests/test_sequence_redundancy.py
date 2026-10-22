from pathlib import Path
import random
import pytest
from amide_synthesis.candidate_selection.sequence_redundancy import cluster_proteins


def test_exact_variants_fragments_and_same_genome_paralogs_remain_distinct(tmp_path):
    executable = Path('/homes/rzhu/miniforge3/envs/palm/bin/cd-hit')
    if not executable.exists():
        pytest.skip('CD-HIT is not installed')
    randomizer = random.Random(42)
    amino = 'ACDEFGHIKLMNPQRSTVWY'
    sequence = ''.join(randomizer.choice(amino) for _ in range(150))
    def mutated(count):
        chars = list(sequence)
        for index in range(count):
            chars[index*5] = amino[(amino.index(chars[index*5])+1) % len(amino)]
        return ''.join(chars)
    def protein(accession, seq, fragment=False):
        return {'accession': accession, 'sequence': seq, 'length': len(seq),
                'reviewed': accession == 'A', 'fragment': fragment,
                'taxon_id': 123, 'proteomes': 'UP000000001'}
    records = [protein('A', sequence), protein('B', sequence), protein('C', mutated(4)),
               protein('D', mutated(25)), protein('F', sequence[:50], True)]
    result = cluster_proteins(records, tmp_path/'95', executable, .95, threads=1)
    assert result['raw_accessions'] == 5 and result['exact_sequences'] == 4
    assert result['representatives'] == 3
    members = {r['accession']: r['representative'] for r in result['members']}
    assert members['A'] == members['B'] == members['C']
    assert members['D'] != members['A']  # same genome is not a duplicate rule
    assert members['F'] != members['A']  # a shared short domain is insufficient
    strict = cluster_proteins(records, tmp_path/'99', executable, .99, threads=1)
    assert strict['representatives'] == 4
