"""Shared loader for the rebuilt ModelSEED corpus.

Every script in this folder reads the corpus through here so they cannot drift
onto different snapshots. Run 00_rebuild_corpus.py first.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from amide_synthesis.infrastructure.paths import HERE
CORPUS = HERE / 'data/modelseed_master.parquet'
EMBEDDING = HERE / 'embeddings/ecfp4_drfp_master.npy'


def unit(matrix):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    assert (norms > 0).all(), 'zero-norm vector; mask before normalising'
    return matrix / norms


def manifest():
    path = HERE / 'results/corpus_manifest.json'
    assert path.exists(), 'no corpus manifest; run 00_rebuild_corpus.py to completion'
    return json.loads(path.read_text())


def load():
    """Return (reactions, L2-normalised ECFP4+DRFP) with degenerate rows dropped.

    A reaction whose products and reactants give an identical fingerprint has no
    direction in this space and cannot be ranked by cosine, so it is removed.
    """
    assert CORPUS.exists(), f'missing {CORPUS}; run 00_rebuild_corpus.py first'
    meta = manifest()                      # written last; its absence means an aborted run
    reactions = pd.read_parquet(CORPUS)
    matrix = np.load(EMBEDDING).astype(np.float32)
    assert len(reactions) == len(matrix), 'corpus and embedding are out of sync'
    # Equal row counts do not imply equal content: verify the ordered reaction
    # identities and structures hash to what the manifest recorded.
    digest = hashlib.sha256(
        '\n'.join(reactions.rxn_id + '\t' + reactions.rxn_smiles).encode()).hexdigest()
    assert digest == meta['corpus_sha256'], (
        'corpus content does not match its manifest; rebuild before trusting results')
    assert list(matrix.shape) == meta['embedding_shape'], 'embedding shape drifted'
    if 'embedding_sha256' in meta:
        # Shape is not identity: reordered rows of the same shape would pass.
        digest_emb = hashlib.sha256(np.ascontiguousarray(matrix)).hexdigest()
        assert digest_emb == meta['embedding_sha256'], (
            'embedding content does not match its manifest; rebuild')
    keep = np.linalg.norm(matrix, axis=1) > 0
    dropped = int((~keep).sum())
    if dropped:
        print(f'dropping {dropped:,} zero-norm (degenerate) reactions', flush=True)
    return reactions[keep].reset_index(drop=True), unit(matrix[keep])


def labelled(reactions, matrix):
    """The EC-labelled subset, used wherever an EC call is needed."""
    mask = (reactions.ec.fillna('') != '').to_numpy()
    return reactions[mask].reset_index(drop=True), matrix[mask]


