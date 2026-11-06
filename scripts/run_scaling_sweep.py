"""Serial resource-bounded scaling probes on this project's GPU processes only."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True)
    p.add_argument('--environments',type=int,nargs='+',default=[64,128,256,512,1024])
    p.add_argument('--memory-budget-mib',type=int,default=28000)
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    results=[];peak_by_n={}
    for n in a.environments:
        live=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
        if live:raise RuntimeError('GPU occupied before benchmark: '+live)
        if len(peak_by_n)>=2:
            x,y=sorted(peak_by_n)[-2:];slope=max(0,(peak_by_n[y]-peak_by_n[x])/(y-x))
            estimate=peak_by_n[y]+(n-y)*slope+2048
            if estimate>a.memory_budget_mib:
                results.append({'num_envs':n,'state':'not_run_memory_estimate','estimated_peak_plus_reserve_mib':estimate,'budget_mib':a.memory_budget_mib})
                (out/'results.json').write_text(json.dumps(results,indent=2)+'\n');continue
        command=['scripts/python.sh','scripts/profile_ppo_scaling.py','--num-envs',str(n),'--out',str(out/f'env_{n}')]
        with (out/f'env_{n}.log').open('xb') as log:
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
            state={'pid':os.getpid(),'state':'running','num_envs':n,'child_pid':child.pid,'command':command}
            (out/'status.json').write_text(json.dumps(state,indent=2)+'\n')
            start=time.monotonic();samples=[];reason=None
            while child.poll() is None:
                row=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
                memory,util=[float(x) for x in row.split(',')];samples.append({'elapsed_s':time.monotonic()-start,'memory_mib':memory,'gpu_utilization':util})
                if memory>a.memory_budget_mib:reason='memory_budget_exceeded';child.terminate();break
                if time.monotonic()-start>1800:reason='stage_timeout';child.terminate();break
                time.sleep(2)
            try:code=child.wait(timeout=20)
            except subprocess.TimeoutExpired:child.kill();code=child.wait()
        (out/f'env_{n}_resources.json').write_text(json.dumps(samples,indent=2)+'\n')
        peak=max((r['memory_mib'] for r in samples),default=0);peak_by_n[n]=peak
        path=out/f'env_{n}/result.json'
        row={'num_envs':n,'state':'completed' if code==0 and path.exists() else 'failed','exit_code':code,'peak_card_mib':peak,'stop_reason':reason}
        if path.exists():row['result']=json.loads(path.read_text())
        results.append(row);(out/'results.json').write_text(json.dumps(results,indent=2)+'\n')
        if code:break
    (out/'status.json').write_text(json.dumps({'pid':os.getpid(),'state':'finished',
        'attempted_environment_counts':[r['num_envs'] for r in results if 'exit_code' in r],
        'completed_environment_counts':[r['num_envs'] for r in results if r['state']=='completed'],
        'skipped_environment_counts':[r['num_envs'] for r in results if r['state']=='not_run_memory_estimate']},indent=2)+'\n')


if __name__=='__main__':main()
