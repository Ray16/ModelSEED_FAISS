# Cofold validation: staged, attempted, blocked on lambda13 node state

Date: 2026-10-13

## What is staged

Three Boltz-2 inputs, each the protein plus adipate, HMDA and ATP — the
nylon-6,6 monomers and the cofactor:

| file | protein | why this one |
|---|---|---|
| `P0DXD6.yaml` | PubC, putrebactin synthase | top candidate; the only enzyme strong on BOTH axes (substrate distance 0.829, external geometric score 0.772) |
| `Q9WY79.yaml` | MurE | surfaced only by the recall fix (substrate distance 0.877); invisible to the pipeline before |
| `OPY85079_1.yaml` | AS17 | **positive control** — a measured strong producer on the nylon-6,6 diad |

AS17 is the point of the exercise: a predictor that cannot rank a known active
highly is not worth running on the other 289.

Intended command, house standard (`--mode exact`, MSA, 50 diffusion samples):

```bash
bash /nfs/lambda_stor_01/homes/rzhu/Tool/uplifting-biomolecular-modeling/boltz2/fold.sh \
  --input <this dir> --mode exact --out <this dir>/out \
  -- --use_msa_server --diffusion_samples 50
```

## Why it did not run

**lambda13 has a driver/library version mismatch.** It is the only
compute-capability->=8.0 node and therefore the only node the kit runs on.

```
kernel module (/proc/driver/nvidia/version) : 565.57.01
userspace NVML library                      : 580.178.04
nvidia-smi                                  : "Failed to initialize NVML: Driver/library version mismatch"
```

The kit bridges **libcuda** with CUDA forward-compat, so the compute path may
well still work — but **NVML is not bridged**, so `nvidia-smi` fails, and
`gpu_reserve` cannot enumerate the node's GPUs:

```
$ ssh lambda13 gpu_reserve pick -n 1 --label probe
bash: line 4: I[$b]: bad array subscript
pick rc=1
```

`gpu_reserve` is the mandatory gate for all GPU work on this cluster, and
running `CUDA_VISIBLE_DEVICES=...` directly to get around it is exactly what
the gate exists to prevent — two jobs on one card, including one's own. It
also could not be verified safe: confirming exclusivity needs per-PCI-bus
process counts, which also needs a working `nvidia-smi`. So this was not
bypassed.

This affects **all three cofolding skills** (`boltz2-fast`, `af3-fast`,
`esmfold2-fast`), since all three run only on lambda13.

## Secondary finding: the failure is silent

`fold.sh` sets `set -euo pipefail` and then does:

```bash
idxs=$("$GR" pick -n "$need" --label boltz2_fast 2>/dev/null | tr '\n' ' ')
```

With `pipefail`, a failing `gpu_reserve pick` makes the whole pipeline
non-zero, `set -e` fires, and the script exits **1 with no output at all** —
it never reaches its own "waiting for a free GPU" message or the
"no free GPU after waiting" error. That is what made this take several probes
to diagnose: a zero-byte log and `rc=1`.

A one-line fix would restore the intended wait-and-retry behaviour:

```bash
idxs=$("$GR" pick -n "$need" --label boltz2_fast 2>/dev/null | tr '\n' ' ') || idxs=""
```

## To resume

Once lambda13's userspace NVIDIA libraries match its kernel module (a driver
reload or a matching `nvidia-utils` downgrade/upgrade — root required), the
command above runs as-is. Nothing else about the staged inputs needs changing.
