"""Read-only inventory; output stays in this project."""
import importlib.metadata as md
import json, platform, subprocess, sys
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parents[1]
def cmd(args):
 p=subprocess.run(args,text=True,capture_output=True)
 return {'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
result={'timestamp':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),'python':sys.version,'executable':sys.executable,'platform':platform.platform(),'os_release':Path('/etc/os-release').read_text(),'gpu':cmd(['nvidia-smi']),'disk':cmd(['df','-h',str(ROOT)]),'packages':{}}
for p in ['torch','torchvision','isaacsim','isaaclab','omni-drones','numpy','gym','gymnasium','stable-baselines3','tensorboard','scipy','PyYAML']:
 try:result['packages'][p]=md.version(p)
 except md.PackageNotFoundError:result['packages'][p]=None
import torch
result['torch_cuda']={'runtime':torch.version.cuda,'available':torch.cuda.is_available(),'arch':torch.cuda.get_arch_list(),'smoke':(torch.ones(1,device='cuda')+1).tolist()}
(ROOT/'audit/environment.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
