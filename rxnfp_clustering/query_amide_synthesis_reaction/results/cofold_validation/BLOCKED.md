# Cofold validation: staged; the driver blocker is GONE, the node is busy

## STATUS 2026-10-13, re-tested — root is no longer required

Everything below was written when `nvidia-smi` on lambda13 failed with a
kernel/NVML version mismatch, and it concluded that a root driver reload was
needed. **That is no longer true.** Re-tested:

```
$ ssh lambda13 nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used ...
0, NVIDIA H100 NVL, 60 %, 48167 MiB
...
7, NVIDIA H100 NVL, 100 %, 48009 MiB
rc=0
```

NVML enumerates all eight H100s. The mismatch was transient node state, not
something needing an administrator. The third "blocked" verdict in this
project to dissolve on re-testing, after SABIO-RK being called dead and the
BRENDA-credentials claim.

**What actually blocks the run now is ordinary contention**: all eight GPUs
are at 60-100% utilisation with 44-59 GB resident, and `gpu_reserve list`
reports zero free slots on lambda13. House rules forbid piling onto an
occupied card, so nothing was launched. This resolves itself without anyone
doing anything — it needs a free GPU, not an intervention.

**`fold.sh` has been fixed so the wait works.** It is written to poll for up
to ~30 minutes, but line 44 assigned from a pipeline under `set -euo
pipefail`, so a failing `pick` — precisely the no-free-GPU case the loop
exists for — aborted the script on attempt 1. Verified:

```
$ bash -c 'set -euo pipefail; idxs=$(false | tr "\n" " "); echo SURVIVED'
rc=1                                    # dies, never retries
$ bash -c 'set -euo pipefail; idxs=$(false | tr "\n" " ") || idxs=""; echo SURVIVED'
SURVIVED                                 # retries as designed
```

`|| idxs=""` added, backup at `fold.sh.bak-20261013`, syntax checked.

**Confirmed in production, not just in a shell demo.** The 12-input run was
launched and is now polling as designed:

```
[boltz2-fast] waiting for 1 free GPU(s) on lambda13 (attempt 1)
[boltz2-fast] waiting for 1 free GPU(s) on lambda13 (attempt 2)
[boltz2-fast] waiting for 1 free GPU(s) on lambda13 (attempt 3)
[boltz2-fast] waiting for 1 free GPU(s) on lambda13 (attempt 4)
```

Before the fix this printed nothing and exited 1 on attempt 1. It now waits
through the `gpu_reserve` gate for up to ~30 minutes and exits 9 cleanly if
no card frees — which costs nothing and takes no GPU from anyone.

### Outcome of that run: clean timeout, node saturated

It polled all 60 attempts and stopped:

```
[boltz2-fast] waiting for 1 free GPU(s) on lambda13 (attempt 60)
[boltz2-fast] no free GPU on lambda13 after waiting
```

All eight H100s stayed at 45-100% utilisation with 44-59 GB resident for the
full 30 minutes — other users' work, continuously. No card was taken, no
output was produced, and `out/` does not exist.

**This is the correct result and it needs nobody.** The technical blocker is
gone; what remains is ordinary queueing on a shared node. Re-running the
command at the top of this file is the whole of the remaining action, and it
is worth doing whenever lambda13 has a quiet card — `gpu_reserve list | grep
lambda13` shows free slots when there are any.

## Original record, superseded above


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

## What is staged, and why these proteins

Restaged 2026-10-13 against the convergence analysis. The original three were
chosen before it existed, and three of the current top four had no input at
all — so a run on the old set would have folded the wrong proteins.

| input | lines | why it is here |
|---|---:|---|
| `Q9RDS6.yaml` **GlnA2** | 4 | natively acylates **cadaverine**, the C5 diamine one methylene short of HMDA; K_M 0.13 mM; pocket span 17 vs HMDA's 6 |
| `Q81RQ8.yaml` **AsbB** | 4 | pocket span **23**; cofactor-bound structure; independent geometric score |
| `P9WJL3.yaml` **MurE** | 4 | **meso-diaminopimelate bound in the crystal**; acid span 10, amine span 5 — the predicted failure is on the acceptor |
| `Q46E79.yaml` **PylC** | 4 | bifunctional ligand bound; cofactor structure; geometric score |
| `O60993.yaml` **TryS** | 2* | near miss by 0.0116; pocket span **30/30**, the largest in the set |
| `P0DXD6.yaml` **PubC** | 3 | kept: the project's original headline, useful as a comparison |
| `Q9WY79.yaml` **MurE** *T. maritima* | 3 | kept: a second MurE, tests whether the result is species-specific |
| `OPY85079_1.yaml` **AS17** | — | **positive control**: measured 1.91 µg/ml on the nylon-6,6 diad |

| `A0A075UYU2.yaml` | — | **the other hypothesis**: ANL, 30.6% identity to AS9 |
| `A1YAM3.yaml` | — | ANL, 30.6% to AS9 |
| `D2TMN3.yaml` **entE** | — | ANL, 30.2% to AS9 |
| `A0ABC8C1M7.yaml` | — | ANL, 29.8% to AS23 |

Each is the protein plus adipate `OC(=O)CCCCC(=O)O`, HMDA `NCCCCCCN` and ATP.

The last four reach **no** independent evidence line in this pipeline. They
are staged because `results/ASSAY_ROUND.md` recommends them under the
competing reading — that the ANL/adenylate-forming panel is the family that
works — and folding only the chemistry-driven picks would test half the
question on a GPU that is hard to get. 12 inputs: 8 chemistry-side (including
the AS17 control), 4 measured-data-side.

AS17 is the one that matters most for interpretation: if the scoring cannot
separate a measured producer from the rest, the negative results below it mean
nothing.

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
