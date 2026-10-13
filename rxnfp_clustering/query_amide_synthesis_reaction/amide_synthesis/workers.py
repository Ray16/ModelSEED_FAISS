"""Pick a worker count that uses real headroom without hogging a shared node.

This box has 80 cores and other users on it, so a fixed fan-out is either
wasteful (when the node is idle) or rude (when it is busy). The default instead
measures the current 1-minute load and claims a bounded share of what is
actually free, re-measured at each call. `N_WORKERS` overrides it outright.
"""
from __future__ import annotations

import os

FLOOR = 4
CEILING = 48          # the user-sanctioned upper bound on this node
SHARE = 0.7           # fraction of idle cores to claim


def worker_count(ceiling: int = CEILING) -> int:
    override = os.environ.get('N_WORKERS')
    if override:
        return max(1, int(override))
    cores = os.cpu_count() or 8
    try:
        load = os.getloadavg()[0]
    except OSError:
        load = cores / 2
    idle = max(0.0, cores - load)
    return int(min(ceiling, max(FLOOR, idle * SHARE)))


def describe(workers: int) -> str:
    try:
        load = os.getloadavg()[0]
    except OSError:
        load = float('nan')
    return (f'{workers} workers '
            f'(cores={os.cpu_count()}, load={load:.1f}, cap={CEILING})')
