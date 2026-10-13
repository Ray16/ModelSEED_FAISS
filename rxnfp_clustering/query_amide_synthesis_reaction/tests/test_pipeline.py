import contextlib
import io
import os
import sys
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from amide_synthesis import pipeline


class PipelineTests(unittest.TestCase):
    def test_importing_stages_creates_no_output_directories(self):
        with tempfile.TemporaryDirectory() as temp:
            work = Path(temp) / 'not-created'
            env = dict(os.environ, AMIDE_SYNTHESIS_WORKDIR=str(work),
                       AMIDE_SYNTHESIS_OFFLINE='1', N_WORKERS='1')
            code = ('import importlib; from amide_synthesis.pipeline import STAGES; '
                    '[importlib.import_module("amide_synthesis." + s.module) for s in STAGES]')
            result = subprocess.run([sys.executable, '-c', code], cwd=pipeline.PROJECT,
                                    env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(work.exists())

    def test_stage_order_and_offline_selection(self):
        stages = pipeline.select_stages(only=['native', 'table', 'enzymes'], offline=True)
        self.assertEqual([s.name for s in stages], ['table', 'native'])
        self.assertNotIn('corpus', [s.name for s in pipeline.select_stages(skip_corpus=True)])

    def test_unknown_stage_and_empty_only_are_errors(self):
        for argv in (['--only', 'typo'], ['--only']):
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    pipeline.main(argv)
                self.assertEqual(error.exception.code, 2)

    def test_dry_run_uses_current_python_without_creating_directories(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(pipeline, 'ensure_directories') as mkdir:
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(pipeline.main(['--dry-run', '--only', 'table']), 0)
        self.assertIn(sys.executable, output.getvalue())
        mkdir.assert_not_called()

    def test_stage_failure_stops_pipeline_and_offline_flag_reaches_children(self):
        with patch.object(pipeline, 'ensure_directories'), patch.object(pipeline, 'run_stage', return_value=7) as run:
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(pipeline.main(['--only', 'table', 'native', '--offline']), 1)
        self.assertEqual(run.call_count, 1)
        self.assertEqual(run.call_args.args[0].name, 'table')
        self.assertEqual(run.call_args.args[2]['AMIDE_SYNTHESIS_OFFLINE'], '1')

    def test_missing_interpreter_returns_failure(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(pipeline, 'ensure_directories'):
            with patch.object(pipeline, 'LOGS', Path(temp)), patch.object(pipeline, 'run_stage', side_effect=FileNotFoundError('missing')):
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(pipeline.main(['--python', '/missing', '--only', 'table']), 1)
