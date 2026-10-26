"""Exercise the relocated utilities through their public command interface."""
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest

from amide_synthesis import cli
from amide_synthesis.infrastructure.paths import PROJECT


@pytest.mark.parametrize('command,expected_option', [
    (['--help'], 'discover'),
    (['discover', '--help'], '--phase'),
    (['rank', '--help'], '--count'),
    (['native-go', '--help'], '--evidence-dir'),
    (['evaluate', 'native-audit', '--help'], '--refresh'),
    (['evaluate', 'native-similarity', '--help'], '--skip-rxnfp'),
    (['evaluate', 'native-draw', '--help'], '--output-root'),
    (['evaluate', 'selected-reactions', '--help'], '--table'),
    (['evaluate', 'polymerization-target', '--help'], '--output'),
])
def test_help_never_creates_outputs_or_fetches_data(command, expected_option, tmp_path):
    work = tmp_path / 'not-created'
    env = dict(os.environ, AMIDE_SYNTHESIS_WORKDIR=str(work),
               AMIDE_SYNTHESIS_OFFLINE='1')
    result = subprocess.run([sys.executable, '-m', 'amide_synthesis', *command],
                            cwd=PROJECT, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert 'usage:' in result.stdout
    assert expected_option in result.stdout
    assert not work.exists()


def test_command_forwards_flags_and_restores_argv_on_failure():
    previous = sys.argv
    received = []

    def fail(module, run_name):
        received.append((module, run_name, list(sys.argv)))
        raise SystemExit(2)

    with patch.object(cli.runpy, 'run_module', side_effect=fail):
        with pytest.raises(SystemExit) as error:
            cli.main(['rank', '--count', '10', '--output', 'new run'])
    assert error.value.code == 2
    module = 'amide_synthesis.reaction_search.rank_modelseed_nylon'
    assert received == [(module, '__main__', [module, '--count', '10', '--output', 'new run'])]
    assert sys.argv is previous


def test_rxnfp_worker_boots_with_its_legacy_interpreter(tmp_path):
    lock = json.loads((PROJECT / 'config/top10_reproduction.lock.json').read_text())
    python = Path(lock['rxnfp_python'])
    if not python.is_file():
        pytest.skip('Separate RXNFP environment is not installed')
    worker = PROJECT / 'amide_synthesis/infrastructure/legacy_workers/native_similarity_rxnfp.py'
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2',
               OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
    result = subprocess.run([str(python), str(worker), '--help'], cwd=tmp_path,
                            env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert '--allow-missing' in result.stdout
    assert not list(tmp_path.iterdir())


def test_polymerization_output_keeps_balance_and_relocated_provenance(tmp_path):
    from amide_synthesis.evaluation.draw_polymerization_target import main

    main(['--output', str(tmp_path)])
    manifest = json.loads((tmp_path / 'manifest.json').read_text())
    assert manifest['validation_passed']
    assert all(Path(p).is_file() for p in manifest['source_sha256'])
    assert {Path(p).parent for p in manifest['source_sha256']} == {
        PROJECT / 'amide_synthesis/evaluation'}
    structures = json.loads((tmp_path / 'structures.json').read_text())
    assert structures['validated_chain_lengths'] == [1, 2, 3, 10]
    assert (tmp_path / 'nylon66_polymerization.png').stat().st_size > 0
