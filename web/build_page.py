"""Fill web/page_template.html with numbers taken straight from results/*.json -> web/dist/index.html
(and write docs/validation.md from the same numbers, so page and docs cannot drift apart).

  python web/build_page.py
"""
import collections
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
sys.path.insert(0, os.path.join(ROOT, 'sim'))
R = lambda *p: os.path.join(ROOT, 'results', *p)

CN = dict(block='积木（4 cm）', ball='网球（6.7 cm）', remote='遥控器（15×4.5×2 cm）', car='玩具车（9×4.4×3 cm）', sock='卷好的袜子')
COND = dict(standard='标准：物体离墙和家具 ≥30 cm，贴胶带', near_walls='贴边：物体离墙和家具 12 cm 起，贴胶带', bare_jaws='裸夹爪：不贴胶带，其余同标准')


def pct(a, b):
    return f'{100 * a / b:.0f}%' if b else '-'


def main():
    T = json.load(open(R('trials.json'))); G = json.load(open(R('grasp_bench.json'))); B = json.load(open(R('base_bench.json')))
    M = json.load(open(R('model_facts.json')))
    agg = {}
    for c in T['conditions']:
        eps = [e for e in T['episodes'] if e['condition'] == c['name']]
        d = sum(e['delivered'] for e in eps); tot = sum(e['total'] for e in eps)
        why = collections.defaultdict(list)
        for e in eps:
            for k, v in e['outcome'].items():
                if v != '已送达': why[v].append(k)
        agg[c['name']] = dict(n=len(eps), d=d, tot=tot, all5=sum(e['delivered'] == e['total'] for e in eps), t=sum(e['sim_time'] for e in eps) / max(d, 1),
                              gf=sum(e['grasp_fail'] for e in eps), ga=sum(e['grasp_attempts'] for e in eps), drop=sum(e['dropped'] for e in eps),
                              coll=sum(e['collisions'] for e in eps), tilt=sum(e['max_tilt'] > 3 for e in eps), max_tilt=max(e['max_tilt'] for e in eps),
                              odom=sum(e['odom_err'] for e in eps) / len(eps), dist=sum(e['dist'] for e in eps) / len(eps), why=why, stuck=sum(e['stuck'] for e in eps))
    s = agg['standard']
    gt = {k: (v['success'], v['n']) for k, v in G['1.0'].items()}; gb = {k: (v['success'], v['n']) for k, v in G['0.35'].items()}
    gt_all = (sum(a for a, _ in gt.values()), sum(n for _, n in gt.values())); gb_all = (sum(a for a, _ in gb.values()), sum(n for _, n in gb.values()))

    trial_rows = ''.join(
        f'<tr><td>{COND[n]}</td><td class="n">{a["n"]}</td><td class="n">{a["d"]}/{a["tot"]}（{pct(a["d"], a["tot"])}）</td><td class="n">{a["all5"]}/{a["n"]}</td>'
        f'<td class="n">{a["t"]:.0f} 秒</td><td class="n">{a["gf"]}/{a["ga"]}</td><td class="n">{a["drop"]}</td><td class="n">{a["coll"]}</td></tr>' for n, a in agg.items())
    fail_rows = ''
    for n, a in agg.items():
        if not a['why']:
            fail_rows += f'<li><b>{COND[n].split("：")[0]}</b>：全部送达。</li>'; continue
        parts = []
        for reason, kinds in sorted(a['why'].items(), key=lambda kv: -len(kv[1])):
            cnt = collections.Counter(kinds)
            parts.append(f'{len(kinds)} 件「{reason}」（' + '、'.join(f'{CN[k].split("（")[0]} {v}' for k, v in cnt.most_common()) + '）')
        fail_rows += f'<li><b>{COND[n].split("：")[0]}</b>：{a["tot"] - a["d"]} 件没送达。' + '；'.join(parts) + '。</li>'
    grasp_rows = ''.join(f'<tr><td>{CN[k]}</td><td class="n {"good" if gt[k][0] >= 0.8 * gt[k][1] else "bad"}">{gt[k][0]}/{gt[k][1]}</td>'
                         f'<td class="n {"good" if gb[k][0] >= 0.8 * gb[k][1] else "bad"}">{gb[k][0]}/{gb[k][1]}</td></tr>' for k in CN)
    lab = dict(forward='前进 0.15 m/s', sideways='横移 0.15 m/s', diagonal='斜向 0.10 + 0.10 m/s', spin='原地转 0.8 rad/s', overspeed='前进 0.40 m/s（超速）')
    def got(k):
        g = B[k]['got']
        if k == 'spin': return f'{g[2]:.2f} rad/s'
        if k == 'diagonal': return f'{g[0]:.3f} + {g[1]:.3f} m/s'
        return f'{(g[0] if k != "sideways" else g[1]):.3f} m/s'
    base_rows = ''.join(f'<tr><td>{lab[k]}</td><td class="n">{got(k)}</td><td class="n">{B[k]["tilt_deg"]:.2f}°</td></tr>' for k in lab)

    nw, bj = agg['near_walls'], agg['bare_jaws']
    valid = [
        ('称重', '整机和机械臂分别上秤', f'整机 {M["robot_mass"]:.2f} kg，其中机械臂 {M["arm_mass"]:.2f} kg', '差 10% 以上就改 <code>model.py</code> 的 <code>P</code> 重跑'),
        ('底盘速度', '指令 0.15 m/s，前进、横移各跑 4 秒，量实际位移', f'前进 {B["forward"]["got"][0]:.3f}，横移 {B["sideways"]["got"][1]:.3f} m/s', '在指令的 ±10% 以内'),
        ('原地旋转', '指令 0.8 rad/s 转 4 秒，量转过的角度', f'{B["spin"]["got"][2]:.2f} rad/s', '±10% 以内'),
        ('最高速度', '指令 0.4 m/s', f'到 {B["overspeed"]["got"][0]:.2f} m/s 饱和', '记下实测值，写回 <code>MAX_WHEEL_SPEED</code>'),
        ('里程计漂移', '走一个 1 m 的正方形回到起点，量闭合误差', f'台架接近 0（滚子是理想的）；整机试验平均 {s["odom"]:.2f} m / {s["dist"]:.0f} m 行程', '只做记录。实物一定更大，所以定位必须靠雷达'),
        ('雷达自遮挡', '收臂状态原地采一圈，数被机身滤波丢掉的点', f'{M["lidar_self_hits"]}/{M["lidar_rays"]} 线', '正前方盲区不超过仿真值 + 10 线'),
        ('定位精度', '在 3 个已知点用卷尺核对定位输出', '假设 1.5 cm、1°', '不超过假设值；超过就把实测值写进 <code>controller.py</code> 重跑'),
        ('相机定位', '物体放在 0.25 m 和 1.5 m 处，核对相机给出的位置', '假设近处 4 mm，1.5 m 处 2.8 cm', '同上'),
        ('单物体抓取', '机器人停稳，物体在正前方 20–26 cm，五种各 10 次', '贴胶带：' + '，'.join(f'{CN[k].split("（")[0]} {gt[k][0]}/{gt[k][1]}' for k in CN), '每种不比仿真少 2 次以上'),
        ('胶带对照', '撕掉胶带，网球夹 10 次', f'{gb["ball"][0]}/{gb["ball"][1]}', '应该明显变差。如果没变差，说明仿真的摩擦假设偏保守'),
        ('整机任务', '4×3 m 场地摆 5 件，离墙 ≥30 cm，做 10 次', f'送达 {pct(s["d"], s["tot"])}，每件 {s["t"]:.0f} 秒，{s["n"]} 次试验共碰撞 {s["coll"]} 次', '送达率不低于仿真 15 个百分点，没有倾翻'),
        ('贴边任务', '同上，物体离墙 12 cm 起', f'送达 {pct(nw["d"], nw["tot"])}', '只做记录'),
    ]
    valid_rows = ''.join(f'<tr><td class="n">{i + 1}</td><td>{a}</td><td>{b}</td><td>{c}</td><td>{d}</td></tr>' for i, (a, b, c, d) in enumerate(valid))

    round_rows = ''
    for fn, name in (('trials_round1.json', '第一轮：初版控制器'), ('trials_round2.json', '第二轮：轻合夹爪、近看再对位、远侧净空检查'),
                     ('trials_round3.json', '第三轮：加了只看雷达、只管平移的防撞层'), ('trials.json', '第四轮：防撞层按真实外形算，也管转身（当前版本）')):
        alle = json.load(open(R(fn)))['episodes']; eps = [e for e in alle if e['condition'] == 'standard']
        d = sum(e['delivered'] for e in eps); tot = sum(e['total'] for e in eps)
        round_rows += (f'<tr><td>{name}</td><td class="n">{d}/{tot}</td><td class="n">{sum(e["collisions"] > 0 for e in alle)} 次试验</td>'
                       f'<td class="n">{sum(e["max_tilt"] > 30 for e in alle)}</td><td class="n">{sum(e["max_tilt"] > 3 for e in alle)}/{len(alle)}</td></tr>')
    video = json.load(open(R('episode.json')))
    rep = {
        'VIDEO_NOTE': f'第 {video["seed"]} 号随机布局，仿真时间 {video["sim_time"] / 60:.1f} 分钟，送达 {video["delivered"]}/{video["total"]}',
        'STD_DELIVERED': f'{s["d"]}/{s["tot"]}', 'STD_PCT': pct(s['d'], s['tot']), 'STD_N': str(s['n']), 'STD_TIME': f'{s["t"]:.0f}',
        'GRASP_TAPE': f'{gt_all[0]}/{gt_all[1]}', 'GRASP_TAPE_PCT': pct(*gt_all), 'GRASP_BARE': f'{gb_all[0]}/{gb_all[1]}', 'GRASP_BARE_PCT': pct(*gb_all),
        'TRIAL_ROWS': trial_rows, 'FAIL_ROWS': fail_rows, 'GRASP_N': str(gt['block'][1]), 'GRASP_ROWS': grasp_rows, 'BASE_ROWS': base_rows,
        'BARE_BALL': f'{gb["ball"][0]}/{gb["ball"][1]}', 'TAPE_BALL': f'{gt["ball"][0]}/{gt["ball"][1]}', 'BARE_PCT': pct(bj['d'], bj['tot']),
        'TAPE_REMOTE': f'{gt["remote"][0]}/{gt["remote"][1]}', 'VALID_ROWS': valid_rows, 'OTHER_N': f'{nw["n"]} 次和 {bj["n"]} 次',
        'ROUND_ROWS': round_rows, 'NO_STANCE': f'{sum(v.get("no_stance", 0) for v in G["1.0"].values())}',
        'ROBOT_MASS': f'{M["robot_mass"]:.2f}',
        'BARE_GF': f'{bj["gf"]}/{bj["ga"]}', 'STD_GF': f'{s["gf"]}/{s["ga"]}', 'STD_TILT': f'{s["tilt"]}/{s["n"]}', 'STD_COLL': str(s['coll']),
    }
    html = open(os.path.join(HERE, 'page_template.html'), encoding='utf8').read()
    for k, v in rep.items(): html = html.replace('{{' + k + '}}', v)
    assert '{{' not in html, html[html.index('{{'):html.index('{{') + 40]
    os.makedirs(os.path.join(HERE, 'dist'), exist_ok=True)
    open(os.path.join(HERE, 'dist', 'index.html'), 'w', encoding='utf8').write(html)
    for f in ('episode.mp4', 'episode.jpg'): shutil.copy(R(f), os.path.join(HERE, 'dist', f))

    md = ['# 实物验收方案', '', '按顺序做，前一项不过先别做后一项。「仿真值」来自 `results/` 里的这一轮试验，由 `web/build_page.py` 自动填入。', '',
          '| # | 测什么 | 怎么测 | 仿真值 | 通过 |', '|---|---|---|---|---|']
    md += [f'| {i + 1} | {a} | {b} | {c} | {d} |'.replace('<code>', '`').replace('</code>', '`') for i, (a, b, c, d) in enumerate(valid)]
    md += ['', '## 整机试验汇总', '', '| 条件 | 试验次数 | 送达 | 五件全送达 | 每件用时 | 抓取失败/尝试 | 途中掉落 | 碰撞 | 机身倾斜超过 3° 的试验 |', '|---|---|---|---|---|---|---|---|---|']
    md += [f'| {COND[n]} | {a["n"]} | {a["d"]}/{a["tot"]}（{pct(a["d"], a["tot"])}） | {a["all5"]}/{a["n"]} | {a["t"]:.0f} 秒 | {a["gf"]}/{a["ga"]} | {a["drop"]} | {a["coll"]} | {a["tilt"]}（最大 {a["max_tilt"]:.1f}°） |' for n, a in agg.items()]
    md += ['', '## 仿真里什么是真的，什么是假设', '',
           '真物理：轮子滚子与地面的接触摩擦；舵机力矩上限 2.94 N·m 和由阻尼决定的转速上限；机械臂的质量、惯量、限位（SO-101 官方模型）；夹爪与物体的接触；雷达射线投射（1 cm 噪声、被自身遮挡）。', '',
           '假设（每条都对应上表一项实测）：定位 = 真值 + 1.5 cm / 1° 慢变噪声；识别 = 视野内且无遮挡就一定认出，位置噪声 5 mm + 1.5% 距离；收纳筐位置已知；底盘重量是估的；打印件不变形、舵机无回差；袜子用偏软的椭球代替。', '',
           '## 实测后怎么回填', '', '1. 硬件参数改 `sim/model.py` 顶部的 `P`（质量、摩擦、轮径、力矩上限）。', '2. 传感器噪声改 `sim/controller.py` 顶部的常量。',
           '3. 依次重跑 `python test_base.py`、`python test_grasp.py 10`、`python run_trials.py`，再 `python ../web/build_page.py`。', '']
    open(os.path.join(ROOT, 'docs', 'validation.md'), 'w', encoding='utf8').write('\n'.join(md))
    print('built', len(html), 'bytes;', {k: (v if len(v) < 40 else '...') for k, v in rep.items() if k not in ('TRIAL_ROWS', 'FAIL_ROWS', 'GRASP_ROWS', 'BASE_ROWS', 'VALID_ROWS', 'ROUND_ROWS')})


if __name__ == '__main__':
    main()
