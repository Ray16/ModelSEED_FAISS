"""Compatibility handoff; original campaign implementation is archived."""
from pathlib import Path
import sys
RUN = Path(__file__).resolve().parents[1]
PROJECT = next(parent for parent in RUN.parents if (parent/'amide_synthesis').is_dir())
sys.path.insert(0, str(PROJECT))
from amide_synthesis.structure_modeling.poses import inspect, summary
from amide_synthesis.structure_modeling.poses import aggregate as _aggregate

def aggregate():
    return _aggregate(RUN, write=True)

if __name__ == '__main__':
    import json
    print(json.dumps(aggregate(), indent=2))
