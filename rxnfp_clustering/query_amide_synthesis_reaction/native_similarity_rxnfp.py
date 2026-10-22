#!/usr/bin/env python
"""Small CPU-only RXNFP worker, compatible with the installed Python 3.6 env."""
import argparse
import hashlib
import json
import os
import pkg_resources
import numpy as np
import torch
from rxnfp.transformer_fingerprints import get_default_model_and_tokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input')
    parser.add_argument('output')
    parser.add_argument('--models', nargs='+', default=['bert_ft', 'bert_pretrained'])
    parser.add_argument('--allow-missing', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    records = json.load(open(args.input))
    smiles = [r['rxn_smiles'] for r in records]
    arrays, metadata = {}, {}
    for variant in args.models:
        model, tokenizer = get_default_model_and_tokenizer(variant, force_no_cuda=True)
        model = model.to('cpu').eval()
        lengths = [len(tokenizer.encode(s, add_special_tokens=True)) for s in smiles]
        unknown = [tokenizer.tokenize(s).count(tokenizer.unk_token) for s in smiles]
        valid = [i for i in range(len(smiles)) if lengths[i] <= model.config.max_position_embeddings and not unknown[i]]
        if len(valid) != len(smiles) and not args.allow_missing:
            raise ValueError('RXNFP inputs exceed token limit or contain unknown tokens')
        vectors = np.full((len(smiles), model.config.hidden_size), np.nan, dtype=np.float32)
        with torch.no_grad():
            for start in range(0, len(valid), 4):
                slots = valid[start:start + 4]
                encoded = tokenizer.batch_encode_plus([smiles[i] for i in slots], padding=True,
                                                       truncation=False, return_tensors='pt')
                result = model(**encoded)
                hidden = result.last_hidden_state if hasattr(result, 'last_hidden_state') else result[0]
                vectors[slots] = hidden[:, 0, :].cpu().numpy()
        arrays[variant] = vectors
        directory = pkg_resources.resource_filename('rxnfp', 'models/transformers/' + variant)
        paths = [os.path.join(directory, f) for f in ['pytorch_model.bin', 'config.json', 'vocab.txt']]
        metadata[variant] = dict(device='cpu', pooling='CLS', token_lengths=lengths,
                                 unknown_tokens=unknown, dimension=int(arrays[variant].shape[1]),
                                 skipped_indices=[i for i in range(len(smiles)) if i not in valid],
                                 file_sha256={p: hashlib.sha256(open(p, 'rb').read()).hexdigest() for p in paths})
    np.savez(args.output, **arrays)
    with open(args.output + '.json', 'w') as stream:
        json.dump(dict(models=metadata, ordered_names=[r['name'] for r in records],
                       rxnfp_version=pkg_resources.get_distribution('rxnfp').version), stream, indent=2)


if __name__ == '__main__':
    main()
