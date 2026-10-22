"""Central artifact paths for this review, independent of the working directory."""
from pathlib import Path
import json
RUN=Path(__file__).resolve().parents[1]
def artifact(name):
    paths=json.loads((RUN/'artifacts.json').read_text())['files']
    return RUN/paths.get(str(name),str(name))
def relative(name):
    return str(artifact(name).relative_to(RUN))
