"""Paired baseline/target-speed results without claiming speed matching or causality."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from compare_memory_ablation import paired_counts


def main():
    p=argparse.ArgumentParser();p.add_argument('baseline');p.add_argument('fixed');p.add_argument('--out',required=True)
    a=p.parse_args();roots=[Path(a.baseline),Path(a.fixed)];out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    for root in roots:
        if json.loads((root/'long_run/status.json').read_text())['state']!='completed':raise ValueError('Pipeline not completed')
    for name in ['initial/latest.pt','dataset/manifest.json','vae/best.pt','lstm/best.pt']:
        hashes=[hashlib.sha256((root/name).read_bytes()).hexdigest() for root in roots]
        if hashes[0]!=hashes[1]:raise ValueError('Shared input differs: '+name)
    cfgs=[json.loads((root/'final/config.json').read_text()) for root in roots]
    comparable=copy.deepcopy(cfgs)
    for cfg in comparable:cfg.pop('_run',None)
    mode=comparable[1]['reward'].pop('speed_mode');target=comparable[1]['reward'].pop('target_speed')
    if mode!='target' or comparable[0]!=comparable[1]:raise ValueError('Unexpected configuration differences')
    summaries=[json.loads((root/'evaluation_full/summary.json').read_text()) for root in roots]
    if summaries[0]['training_timesteps']!=summaries[1]['training_timesteps']:raise ValueError('Unequal training budgets')
    paired=paired_counts(*[json.loads((root/'evaluation_full/episodes.json').read_text()) for root in roots])
    labels={'past_current_only_success':'baseline_only_success','current_only_success':'fixed_only_success'}
    paired={density:{labels.get(key,key):value for key,value in counts.items()} for density,counts in paired.items()}
    result={'baseline':summaries[0],'fixed':summaries[1],'paired':paired,'target_speed':target,
            'scope':'Single training seed, one declared target/reward coefficient. Actual speeds may differ. No Pareto curve, equal-speed dominance, or cross-seed claim.'}
    (out/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# 自适应速度与目标速度奖励对照','',result['scope'],'',
           '| 密度 | 基线到达率 | 目标速度对照到达率 | 基线碰撞率 | 对照碰撞率 | 基线平均速度 | 对照平均速度 |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for density,x in summaries[0]['results'].items():
        y=summaries[1]['results'][density]
        lines.append(f"| {density} | {x['success_rate']:.3f} | {y['success_rate']:.3f} | {x['collision_rate']:.3f} | {y['collision_rate']:.3f} | {x['mean_speed_all_episodes']:.3f} | {y['mean_speed_all_episodes']:.3f} |")
    lines+=['',f'目标速度为{target}m/s，final速度项系数为{cfgs[1]["reward"]["final"][1]}。这些是本项目事先声明的参数，不冒称官方精确配置；目标速度奖励不保证每步恒速。']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
