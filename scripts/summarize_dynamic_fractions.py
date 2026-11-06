"""Audit and summarize the four paired 30%/50% zero-shot evaluations."""
import argparse,hashlib,json
from collections import Counter
from pathlib import Path
import numpy as np
from scripts.summarize_dynamic_evaluation import pairs
from mavrl_isaac.fraction_tracks import fraction_count


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);args=parser.parse_args()
    root=Path(args.root);protocol=read(root/'protocol.json');status=read(root/'status.json')
    if status['state']!='completed':raise ValueError('All four evaluations must complete')
    metrics={};episodes={};manifests={};sources={};snapshots=[]
    for percent in [30,50]:
        metrics[str(percent)]={};episodes[percent]={};manifests[percent]={}
        for name in ['baseline','current_only']:
            folder=root/f'pct{percent}_{name}_full';report=read(folder/'summary.json')
            rows=read(folder/'episodes.json');scenes=read(folder/'scenes.json')
            assert len(rows)==len(scenes)==300
            assert report['sha256']==status['policy_sha256'][name]==sha(protocol['policies'][name])
            snapshots.append(report['config']['_run']['source_sha256'])
            data={}
            for density in ['sparse','medium','dense']:
                subset=[r for r in rows if r['density']==density];n=len(subset);assert n==100
                counts=Counter(r['reason'] for r in subset)
                collisions=Counter(r['collision_type'] for r in subset if r['reason']=='collision')
                moving_counts=[r['num_dynamic_obstacles'] for r in subset]
                for r in subset:
                    assert r['num_dynamic_obstacles']==fraction_count(r['num_obstacles'],percent/100)
                    s=scenes[f"{density}/{r['scene_seed']}"]
                    assert len(s['original_cylinders'])==r['num_obstacles']
                    assert len(s['moving']['indices'])==r['num_dynamic_obstacles']
                    np.testing.assert_array_equal(np.array(s['moving']['radii']),np.array(s['original_cylinders'])[s['moving']['indices'],2])
                values={'episodes':n,'counts':dict(counts),'arrival_rate':counts['success']/n,
                    'arrival_95ci':report['results'][density]['success_95ci'],
                    'collision_counts':dict(collisions),'mean_speed':float(np.mean([r['mean_speed'] for r in subset])),
                    'mean_duration_s':float(np.mean([r['duration_s'] for r in subset])),
                    'moving_count_min':min(moving_counts),'moving_count_max':max(moving_counts),
                    'moving_count_mean':float(np.mean(moving_counts)),
                    'actual_fraction_mean':float(np.mean([r['num_dynamic_obstacles']/r['num_obstacles'] for r in subset]))}
                assert values['arrival_rate']==report['results'][density]['success_rate']
                assert np.isclose(values['mean_speed'],report['results'][density]['mean_speed_all_episodes'])
                data[density]=values
            metrics[str(percent)][name]=data;episodes[percent][name]=rows;manifests[percent][name]=scenes
            for filename in ['summary.json','episodes.json','trajectories.json','scenes.json','config.json']:
                path=folder/filename;sources[str(path)]=sha(path)
        assert manifests[percent]['baseline']==manifests[percent]['current_only']
    assert all(s==snapshots[0] for s in snapshots)
    for key,a in manifests[30]['baseline'].items():
        b=manifests[50]['baseline'][key];k=len(a['moving']['indices'])
        assert a['original_cylinders']==b['original_cylinders']
        assert all(v==b['moving'][field][:k] for field,v in a['moving'].items())
    model_pairs={str(p):pairs(episodes[p]['baseline'],episodes[p]['current_only']) for p in [30,50]}
    fraction_pairs={name:pairs(episodes[30][name],episodes[50][name]) for name in ['baseline','current_only']}
    static=read('artifacts/delivery_20261106/summary.json');three=read('artifacts/dynamic_existing_20261106/comparison.json')
    references={name:{d:{'static_arrival_rate':static['metrics'][name][d]['success_rate'],
        'three_movers_arrival_rate':three['metrics'][name][d]['arrival_rate']} for d in ['sparse','medium','dense']} for name in ['baseline','current_only']}
    result={'protocol':protocol,'metrics':metrics,'paired_models':model_pairs,'paired_30_vs_50':fraction_pairs,
        'historical_references':references,'source_sha256':sources,
        'validation':{'all_full_source_snapshots_equal':True,'same_tasks_between_models':True,'nested_motion':True,
            'requested_counts_and_radii_verified':True,'small_replay':read(root/'smoke_replay_checks.json')},
        'scope':'One training seed and one motion speed/length. 30% and 50% share nested identical motion. Previous 3-mover reference used a different, more conservative trajectory planner; differences from it are not solely moving percentage.'}
    (root/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# 原有圆柱 30% / 50% 动态比例评估','',
        '直接加载原静态训练的 baseline 与 current-only 权重，无训练、奖励修改或安全过滤器。每个比例、每个模型分别完成三档密度×100 个任务，共 1200 次正式导航。','',
        '运动保持 1.0 m/s、4 m 直线往返，从原位置（轨迹中点）出发；场地、任务、障碍物总数、尺寸和高度保持原值。动态数量为 floor(N×比例+0.5)，奇数地图的 50% 向上取整。30% 的移动集合及轨迹是 50% 的固定前缀。','',
        '| 比例 | 模型 | 密度 | 到达率 | 静止/移动/同时碰撞数 | 越界/超时数 | 平均速度 m/s |',
        '|---|---|---|---:|---:|---:|---:|']
    for percent,data in metrics.items():
        for name,values in data.items():
            for d,m in values.items():
                c=m['collision_counts'];n=m['counts']
                lines.append(f"| {percent}% | {name} | {d} | {m['arrival_rate']:.0%} | {c.get('static',0)}/{c.get('dynamic',0)}/{c.get('both',0)} | {n.get('boundary',0)}/{n.get('timeout',0)} | {m['mean_speed']:.3f} |")
    lines+=['','| 比例 | 密度 | 每图动态数范围 | 平均动态数 | 平均实际动态比例 |','|---|---|---:|---:|---:|']
    for percent,data in metrics.items():
        for d,m in data['baseline'].items():
            lines.append(f"| {percent}% | {d} | {m['moving_count_min']}–{m['moving_count_max']} | {m['moving_count_mean']:.2f} | {m['actual_fraction_mean']:.2%} |")
    lines+=['','速度沿用原 FLU 水平分量模长，先求每个任务的均值，再按任务等权平均；不把两组当作等速度对照。','',
        '| 比例 | 密度 | 两者成功 | 仅 baseline 成功 | 仅 current-only 成功 | 两者失败 |','|---|---|---:|---:|---:|---:|']
    for p,data in model_pairs.items():
        for d,c in data.items():lines.append(f"| {p}% | {d} | {c['both_success']} | {c['left_only_success']} | {c['right_only_success']} | {c['both_fail']} |")
    lines+=['','| 模型 | 密度 | 30% 成功→50% 失败 | 30% 失败→50% 成功 |','|---|---|---:|---:|']
    for name,data in fraction_pairs.items():
        for d,c in data.items():lines.append(f"| {name} | {d} | {c['left_only_success']} | {c['right_only_success']} |")
    lines+=['','历史参考：','',
        '| 模型 | 密度 | 原静态 | 上轮 3 个移动 | 本轮 30% | 本轮 50% |','|---|---|---:|---:|---:|---:|']
    for name,data in references.items():
        for d,m in data.items():lines.append(f"| {name} | {d} | {m['static_arrival_rate']:.0%} | {m['three_movers_arrival_rate']:.0%} | {metrics['30'][name][d]['arrival_rate']:.0%} | {metrics['50'][name][d]['arrival_rate']:.0%} |")
    lines+=['','实现与证据边界：','',
        '- 35 项 CPU 测试通过。全部 300 张原地图预先生成共享 50% 轨迹，并检查完整周期内移动圆柱之间不相交；每条轨迹还避开所有其他圆柱的原始静止位置和起终点净空。未丢弃地图、降低动态数量或修改原始几何。',
        '- 原来的轨迹包围圆互不相交约束过于保守，无法支持部分密集地图的高比例。本轮改为精确的相对运动距离检查，圆柱实体表面间隔至少 1 mm；30% 与 50% 采用同一规则。',
        '- 上轮 3 个动态圆柱使用不同的轨迹筛选规则，因此它与本轮的差异不能完全归因于动态比例。原静态、上一轮权重和日志均保留。',
        '- 50% 档每个模型各 9 个任务进行了两次独立运行，episode、trajectory、scene 的 SHA256 一致；这些 36 次检查不计入正式样本。四组正式评估源码一致，两组模型的任务/运动参数一致，两个比例的轨迹满足嵌套关系。',
        '- 单训练种子、单运动速度与轨迹长度；current-only 保留 LSTM 状态。初始地图可行性不保证动态全过程存在可行路径，重复探针不保证跨硬件或所有未来帧绝对确定性。',
        '- 图中误差线是场景任务到达率的 Wilson 95% 区间，不包含训练种子随机性。','',
        '配置与轨迹来源见 `protocol.json`、`master_tracks.json`；全部原始结果保存在各 `pct*_full/` 目录；机器可读统计见 `comparison.json`。','',
        '![动态比例到达率比较](comparison.png)','']
    (root/'RESULTS.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(12,4.3),sharey=True,layout='constrained');x=np.arange(2)
    for ax,d in zip(axes,['sparse','medium','dense']):
        for offset,name,color in [(-.18,'baseline','#167D9A'),(.18,'current_only','#D97836')]:
            data=[metrics[str(p)][name][d] for p in [30,50]];rates=np.array([m['arrival_rate'] for m in data])*100
            ci=np.array([m['arrival_95ci'] for m in data])*100
            bars=ax.bar(x+offset,rates,.36,label=name.replace('_','-'),color=color,
                yerr=np.maximum(np.vstack((rates-ci[:,0],ci[:,1]-rates)),0),capsize=3)
            ax.bar_label(bars,fmt='%.0f%%',padding=4,fontsize=10)
        ax.set_title(d.capitalize());ax.set_xticks(x,['30%','50%']);ax.set_xlabel('Target moving fraction')
        ax.set_ylim(0,110);ax.set_yticks(np.arange(0,101,20));ax.spines[['top','right']].set_visible(False)
        ax.set_axisbelow(True);ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Arrival rate (%)');axes[-1].legend(loc='upper right',frameon=False)
    fig.suptitle('Existing static-trained policies | 1 m/s, 4 m tracks | 100 paired tasks per density\nError bars: Wilson 95% intervals across tasks',fontsize=11)
    fig.savefig(root/'comparison.png',dpi=180);fig.savefig(root/'comparison.svg');plt.close(fig)
    print(json.dumps({'metrics':metrics,'paired_models':model_pairs,'paired_30_vs_50':fraction_pairs},indent=2))


if __name__=='__main__':main()
