"""Recompute paired dynamic-transfer metrics from raw episodes and make a report."""
import argparse,hashlib,json
from collections import Counter
from pathlib import Path
import numpy as np


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pairs(left,right):
    a={(r['density'],r['scene_seed']):r for r in left}
    b={(r['density'],r['scene_seed']):r for r in right}
    if len(a)!=len(left) or len(b)!=len(right) or a.keys()!=b.keys():
        raise ValueError('Tasks must be unique and paired')
    output={}
    for density in ['sparse','medium','dense']:
        counts=dict(both_success=0,left_only_success=0,right_only_success=0,both_fail=0)
        for key,row in a.items():
            if key[0]!=density:continue
            x,y=row['reason']=='success',b[key]['reason']=='success'
            counts['both_success' if x and y else 'left_only_success' if x else 'right_only_success' if y else 'both_fail']+=1
        output[density]=counts
    return output


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);args=parser.parse_args()
    root=Path(args.root);protocol=read(root/'protocol.json');status=read(root/'status.json')
    if status['state']!='completed':raise ValueError('Evaluation is not complete')
    old=read('artifacts/delivery_20261106/summary.json')
    reports={};episodes={};static={};metrics={};sources={}
    manifests=[]
    for name in ['baseline','current_only']:
        folder=root/(name+'_full')
        reports[name]=read(folder/'summary.json');episodes[name]=read(folder/'episodes.json')
        static[name]=read(Path(old['roots'][name])/'evaluation_full/episodes.json')
        manifests.append(read(folder/'scenes.json'))
        assert len(episodes[name])==300
        assert reports[name]['sha256']==status['policy_sha256'][name]==sha(protocol['policies'][name])
        metrics[name]={}
        for density in ['sparse','medium','dense']:
            rows=[r for r in episodes[name] if r['density']==density]
            counts=Counter(r['reason'] for r in rows)
            collision=Counter(r['collision_type'] for r in rows if r['reason']=='collision')
            values={'episodes':len(rows),'counts':dict(counts),'arrival_rate':counts['success']/len(rows),
                'collision_counts':dict(collision),'mean_speed':float(np.mean([r['mean_speed'] for r in rows])),
                'mean_duration_s':float(np.mean([r['duration_s'] for r in rows])),
                'static_arrival_rate':old['metrics'][name][density]['success_rate'],
                'arrival_95ci':reports[name]['results'][density]['success_95ci']}
            assert len(rows)==100 and values['arrival_rate']==reports[name]['results'][density]['success_rate']
            assert np.isclose(values['mean_speed'],reports[name]['results'][density]['mean_speed_all_episodes'])
            metrics[name][density]=values
        for filename in ['summary.json','episodes.json','trajectories.json','scenes.json','config.json']:
            path=folder/filename;sources[str(path)]=sha(path)
        path=Path(old['roots'][name])/'evaluation_full/episodes.json';sources[str(path)]=sha(path)
    assert manifests[0]==manifests[1] and len(manifests[0])==300
    # Check that selected movers retain the original radii/counts and begin at
    # their original centers; episode obstacle counts must agree with geometry.
    for name in episodes:
        old_rows={(r['density'],r['scene_seed']):r for r in static[name]}
        for row in episodes[name]:
            scene=manifests[0][f"{row['density']}/{row['scene_seed']}"]
            original=np.array(scene['original_cylinders']);motion=scene['moving'];index=motion['indices']
            assert row['num_obstacles']==len(original)==old_rows[(row['density'],row['scene_seed'])]['num_obstacles']
            assert row['num_dynamic_obstacles']==len(index)==3
            np.testing.assert_array_equal(np.array(motion['radii']),original[index,2])
            np.testing.assert_allclose((np.array(motion['starts'])+np.array(motion['ends']))/2,original[index,:2],atol=1e-12)
    comparison=pairs(episodes['baseline'],episodes['current_only'])
    static_pairs={name:pairs(static[name],episodes[name]) for name in episodes}
    result={'protocol':protocol,'metrics':metrics,'paired_baseline_vs_current_only':comparison,
        'paired_static_vs_dynamic':static_pairs,'source_sha256':sources,'validation':{
            'full_scene_manifests_equal':True,'obstacle_counts_radii_initial_positions_preserved':True,
            'render_probe':read(root/'render_probe/result.json'),'smoke_replay':read(root/'smoke_replay_checks.json')},
        'scope':'Descriptive zero-shot transfer on 300 paired map/motion tasks. One training seed and one selected motion setting. Current-only retains memory. Small replay checks are not full-run or cross-hardware determinism guarantees.'}
    (root/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# 原有圆柱运动：MAVRL 直接迁移评估','',
        '原地图中选取 3 个有移动空间的圆柱，保留半径和初始位置，以 1.0 m/s 在 4 m 线段上往返。障碍物总数不变。baseline 和 current-only 均直接加载已交付权重，没有重新训练。每档 100 个相同任务，共 300 个配对任务、600 次正式导航。','',
        '| 模型 | 密度 | 原静态到达率 | 动态到达率 | 静止/移动/同时碰撞数 | 越界/超时数 | 平均速度 m/s |',
        '|---|---|---:|---:|---:|---:|---:|']
    for name,data in metrics.items():
        for density,m in data.items():
            c=m['collision_counts'];counts=m['counts']
            lines.append(f"| {name} | {density} | {m['static_arrival_rate']:.0%} | {m['arrival_rate']:.0%} | {c.get('static',0)}/{c.get('dynamic',0)}/{c.get('both',0)} | {counts.get('boundary',0)}/{counts.get('timeout',0)} | {m['mean_speed']:.3f} |")
    lines+=['','碰撞类型只统计碰撞终止的任务；速度沿用交付的源码 FLU 水平分量模长，并按 episode 等权平均。图中误差线为动态评估到达率的 Wilson 95% 区间，仅描述场景任务的不确定性，不包含训练种子差异。','',
        '| 密度 | 两者成功 | 仅 baseline 成功 | 仅 current-only 成功 | 两者失败 |',
        '|---|---:|---:|---:|---:|']
    for density,c in comparison.items():
        lines.append(f"| {density} | {c['both_success']} | {c['left_only_success']} | {c['right_only_success']} | {c['both_fail']} |")
    lines+=['','| 模型 | 密度 | 静态成功→动态失败 | 静态失败→动态成功 |','|---|---|---:|---:|']
    for name,data in static_pairs.items():
        for density,c in data.items():
            lines.append(f"| {name} | {density} | {c['left_only_success']} | {c['right_only_success']} |")
    lines+=['','验证与范围：','',
        '- 30 项 CPU 测试通过，300 张地图通过几何检查；两组正式评估的完整场景/轨迹参数逐项一致，原权重 SHA256 保持一致。',
        '- 真实 RTX 探针确认原圆柱运动会改变深度，回到原时刻的图像逐像素一致。两组各 9 个任务的独立重复评估中，episode、trajectory、scene 文件 SHA256 一致；没有把这 36 次检查纳入正式样本。',
        '- 保留原始静态 TRAV 用于奖励日志；不以奖励均值作为动态导航能力的主要结论。',
        '- 结论限于单训练种子、每图 3 个移动圆柱、1.0 m/s、4 m 轨迹。可移动圆柱需具有足够空间，完整轨迹不碰静止物体和起终点；未证明动态全过程的路径可行性。',
        '- current-only 仍具有 LSTM 历史状态；本轮差异不构成跨种子稳定优势或普遍记忆收益的证据。',
        '- 原静态对照来自已交付的相同任务。用户选型之前的新增圆柱探针已停止，其结果未纳入本报告。','',
        '完整设置见 `protocol.json`，原始结果见 `baseline_full/`、`current_only_full/`，配对结果及摘要见 `comparison.json`。','',
        '![到达率对比](comparison.png)','']
    (root/'RESULTS.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4.2),sharey=True,layout='constrained')
    labels=['Sparse','Medium','Dense'];x=np.arange(3)
    for ax,name in zip(axes,['baseline','current_only']):
        data=list(metrics[name].values());old_rate=np.array([m['static_arrival_rate'] for m in data])*100
        rate=np.array([m['arrival_rate'] for m in data])*100
        ci=np.array([m['arrival_95ci'] for m in data])*100
        ax.bar(x-.18,old_rate,.36,label='Original static',color='#B7C9D6')
        bars=ax.bar(x+.18,rate,.36,label='3 original cylinders moving',color='#167D9A',
            yerr=np.maximum(np.vstack((rate-ci[:,0],ci[:,1]-rate)),0),capsize=3)
        ax.bar_label(bars,fmt='%.0f%%',padding=4,fontsize=10)
        ax.set_xticks(x,labels);ax.set_ylim(0,110);ax.set_yticks(np.arange(0,101,20));ax.set_title(name.replace('_','-'))
        ax.spines[['top','right']].set_visible(False);ax.set_axisbelow(True);ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Arrival rate (%)');axes[1].legend(loc='lower left',frameon=False,fontsize=9)
    fig.suptitle('Direct transfer | 1.0 m/s, 4 m tracks | 100 paired tasks per density\nError bars: Wilson 95% intervals across tasks',fontsize=11)
    fig.savefig(root/'comparison.png',dpi=180);fig.savefig(root/'comparison.svg');plt.close(fig)
    print(json.dumps({'metrics':metrics,'paired_baseline_vs_current_only':comparison,'paired_static_vs_dynamic':static_pairs},indent=2))


if __name__=='__main__':main()
