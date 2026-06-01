#!/usr/bin/env python
"""
Generate atom-mapped reaction SMILES using RXNMapper.

Reads data/rxn_data.csv and produces data/mapped_rxns.csv with columns:
id, rxn_smiles, mapped_rxn, confidence.

Spawns one worker process per available GPU for parallel mapping.

Usage:
    conda activate rxnfp
    python -m src.atom_mapping
"""

import csv
import multiprocessing as mp
import os
import sys

import torch
from src.config import MAPPED_RXNS_CSV, RXN_DATA_CSV

BATCH_SIZE = 64


def load_reactions(csv_path):
    """Return list of (id, rxn_smiles) tuples from rxn_data.csv."""
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((row["id"], row["rxn_smiles"]))
    return rows


def worker(gpu_id, reactions, result_queue):
    """Map reactions on a single GPU and put results into the queue."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    from rxnmapper import RXNMapper

    rxn_mapper = RXNMapper()
    results = []
    total = len(reactions)

    for i in range(0, total, BATCH_SIZE):
        batch = reactions[i : i + BATCH_SIZE]
        rxn_ids = [r[0] for r in batch]
        rxn_smiles = [r[1] for r in batch]

        try:
            mapped = rxn_mapper.get_attention_guided_atom_maps(rxn_smiles)
            for rxn_id, smi, result in zip(rxn_ids, rxn_smiles, mapped):
                results.append(
                    {
                        "id": rxn_id,
                        "rxn_smiles": smi,
                        "mapped_rxn": result.get("mapped_rxn", ""),
                        "confidence": result.get("confidence", ""),
                    }
                )
        except Exception:
            for rxn_id, smi in zip(rxn_ids, rxn_smiles):
                try:
                    result = rxn_mapper.get_attention_guided_atom_maps([smi])[0]
                    results.append(
                        {
                            "id": rxn_id,
                            "rxn_smiles": smi,
                            "mapped_rxn": result.get("mapped_rxn", ""),
                            "confidence": result.get("confidence", ""),
                        }
                    )
                except Exception as e2:
                    print(f"[GPU {gpu_id}] Skipping {rxn_id}: {e2}", file=sys.stderr)
                    results.append(
                        {
                            "id": rxn_id,
                            "rxn_smiles": smi,
                            "mapped_rxn": "",
                            "confidence": "",
                        }
                    )

        print(f"  [GPU {gpu_id}] Mapped {min(i + BATCH_SIZE, total):>6d} / {total}")

    result_queue.put(results)


def write_results(results, output_path):
    """Write mapped reaction results to CSV."""
    fieldnames = ["id", "rxn_smiles", "mapped_rxn", "confidence"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"Wrote {len(results)} mapped reactions to {output_path}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    num_gpus = torch.cuda.device_count()
    print(f"Found {num_gpus} GPUs.")

    print("Loading reactions from rxn_data.csv ...")
    reactions = load_reactions(RXN_DATA_CSV)
    total = len(reactions)
    print(f"Loaded {total} reactions.\n")

    chunk_size = (total + num_gpus - 1) // num_gpus
    chunks = [reactions[i : i + chunk_size] for i in range(0, total, chunk_size)]

    print(f"Distributing across {num_gpus} GPUs (~{chunk_size} reactions each).\n")

    result_queue = mp.Queue()
    processes = []
    for gpu_id, chunk in enumerate(chunks):
        p = mp.Process(target=worker, args=(gpu_id, chunk, result_queue))
        p.start()
        processes.append(p)

    all_results = []
    for _ in processes:
        all_results.extend(result_queue.get())

    for p in processes:
        p.join()

    id_order = {rxn[0]: idx for idx, rxn in enumerate(reactions)}
    all_results.sort(key=lambda r: id_order.get(r["id"], 0))

    print(f"\nWriting results to {MAPPED_RXNS_CSV} ...")
    write_results(all_results, MAPPED_RXNS_CSV)
    print("Done.")
