"""A transient GPU-discovery timeout must not terminate the fleet controller."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from amide_synthesis.structure_modeling import lambda_two_stage as fleet


class StopAfterLaunch(Exception):pass


class ControllerRecoveryTests(unittest.TestCase):
    def test_discovery_timeout_retries_and_launches_without_touching_results(self):
        with tempfile.TemporaryDirectory(prefix='lambda_controller_recovery_') as directory:
            root=Path(directory)
            for name in ['state','results','failures','claims']:(root/name).mkdir()
            (root/'manifest.json').write_text(json.dumps({'campaign_id':'synthetic_recovery_test',
                'sequence_count':1,'inputs':[{'accession':'pending_fixture'}]}))
            evidence=root/'results/preserved.txt';evidence.write_text('completed output stays intact')
            sleeps=[];launches=[]
            def sleep(seconds):
                sleeps.append(seconds)
                if len(sleeps)==1:
                    state=json.loads((root/'state/coordinator.json').read_text())
                    self.assertEqual(state['phase'],'waiting_for_gpu_discovery')
                else:raise StopAfterLaunch()
            def rpc(host,operation,payload):
                if operation=='resources':return {'cpus':list(range(16)),'load':0,'online':80}
                if operation=='launch':
                    launches.extend(payload['workers'])
                    return [{'host':host,'pid':123,'process_start_ticks':'fixture','gpu':0}]
                raise AssertionError(operation)
            responses=[subprocess.TimeoutExpired('gpu_reserve list',55),
                subprocess.CompletedProcess(['gpu_reserve'],0,'lambda1 0\n','')]
            with patch.object(fleet.subprocess,'run',side_effect=responses),patch.object(fleet,'rpc',side_effect=rpc),patch.object(fleet.time,'sleep',side_effect=sleep):
                with self.assertRaises(StopAfterLaunch):fleet.coordinator(root)
            self.assertEqual(len(launches),1)
            self.assertEqual(len(launches[0]['cpus']),8)
            self.assertEqual(evidence.read_text(),'completed output stays intact')
            self.assertEqual(len(list((root/'claims').iterdir())),0)


if __name__=='__main__':unittest.main()
