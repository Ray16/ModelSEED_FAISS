"""Refresh the report while this authorized detached cofold campaign runs."""
from pathlib import Path
import json,subprocess,sys,time,os
RUN=Path(__file__).resolve().parents[1]

def main():
    deadline=time.monotonic()+24*3600
    while time.monotonic()<deadline:
        p=RUN/'cofold/state/coordinator.json';state=json.loads(p.read_text()) if p.exists() else {}
        subprocess.run([sys.executable,str(RUN/'scripts/render_report.py')],check=True,stdout=subprocess.DEVNULL)
        print(time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),state.get('status'),len(list((RUN/'cofold/state').glob('*.done.json'))),flush=True)
        if state.get('status') in ['complete','incomplete']:return
        if state.get('host')==os.uname().nodename and state.get('pid'):
            try:
                actual=Path(f"/proc/{state['pid']}/stat").read_text().split(') ',1)[1].split()[19]
                alive=actual==state.get('process_start')
            except OSError:alive=False
            if not alive:
                state['status']='coordinator_exited_before_final_status';p.write_text(json.dumps(state,indent=2))
                subprocess.run([sys.executable,str(RUN/'scripts/render_report.py')],check=True,stdout=subprocess.DEVNULL)
                return
        time.sleep(60)
    raise RuntimeError('Report refresher reached its 24-hour limit')

if __name__=='__main__':main()
