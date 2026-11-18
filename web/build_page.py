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

CN = dict(cube='方块', ball='小球', car='玩具车', banana='香蕉玩具', lbrick='L 形积木', rock='石头', sock='袜子（软）', plush='毛绒玩具（软）')
SIZE = dict(cube='边长 3–5 cm', ball='直径 5.2–6.8 cm', car='长 8–10 cm，宽 4–5 cm', banana='长 12–16 cm，粗 2.8–3.6 cm', lbrick='两臂 4–7 cm，厚 2–2.8 cm，高 1.8–3 cm',
            rock='直径约 3–6 cm，偏扁', sock='长 11–15 cm，粗 3.6–4.6 cm', plush='身体直径 4.4–5.6 cm，四肢 2–3 cm')
HOW = dict(cube='硬，立方体', ball='硬，圆球，会滚', car='硬，车身加四个轮子', banana='硬，弯的长条', lbrick='硬，L 形，两条臂不一样长',
           rock='硬，随机凸多面体，每次形状不同', sock='软，3 段用弹性铰链连起来，会垂、会弯', plush='软，身体加头和四肢共 6 段，一捏就扁')
COND = dict(standard='标准：物体离墙和家具 ≥30 cm，贴胶带', near_walls='贴边：物体离墙和家具 12 cm 起，贴胶带', bare_jaws='裸夹爪：不贴胶带，其余同标准',
            clutter='扎堆：有两对物体边缘只隔 0.5–5 cm，其余同标准')


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
            parts.append(f'{len(kinds)} 件「{reason}」（' + '、'.join(f'{CN[k]} {v}' for k, v in cnt.most_common()) + '）')
        fail_rows += f'<li><b>{COND[n].split("：")[0]}</b>：{a["tot"] - a["d"]} 件没送达。' + '；'.join(parts) + '。</li>'
    grasp_rows = ''.join(f'<tr><td>{CN[k]}</td><td class="n {"good" if gt[k][0] >= 0.8 * gt[k][1] else "bad"}">{gt[k][0]}/{gt[k][1]}</td>'
                         f'<td class="n {"good" if gb[k][0] >= 0.8 * gb[k][1] else "bad"}">{gb[k][0]}/{gb[k][1]}</td>'
                         f'<td class="n">{G["1.0"][k]["misjudged"] + G["0.35"][k]["misjudged"]}</td></tr>' for k in CN)
    # per kind: sizes and weights that actually appeared, and deliveries, over every episode of every condition
    seen = collections.defaultdict(lambda: dict(n=0, d=0, size=[], mass=[]))
    for e in T['episodes']:
        for k, o in e['objects'].items():
            v = seen[k]; v['n'] += 1; v['d'] += e['outcome'][k] == '已送达'; v['size'].append(o['size_cm']); v['mass'].append(o['mass_g'])
    object_rows = ''.join(
        f'<tr><td>{CN[k]}</td><td>{HOW[k]}</td><td class="n">{SIZE[k]}</td><td class="n">{min(seen[k]["mass"]):.0f}–{max(seen[k]["mass"]):.0f} g</td>'
        f'<td class="n {"good" if gt[k][0] >= 0.8 * gt[k][1] else "bad"}">{gt[k][0]}/{gt[k][1]}</td>'
        f'<td class="n {"good" if seen[k]["d"] >= 0.8 * seen[k]["n"] else "bad"}">{seen[k]["d"]}/{seen[k]["n"]}</td></tr>' for k in CN if seen[k]['n'])
    logs = [m for e in T['episodes'] for _, m in e['log']]
    n_soft = sum('被压扁了' in m for m in logs); n_place = sum(len(e['places']) for e in T['episodes'])
    shake_kinds = []
    for e in T['episodes']:
        for t, m in e['log']:
            if '抖一抖' in m:
                nxt = [x for x in e['places'] if x[0] >= t]; shake_kinds.append(nxt[0][2] if nxt else None)
    n_shake = len(shake_kinds); n_shake_soft = sum(k in ('sock', 'plush') for k in shake_kinds)
    lab = dict(forward='前进 0.15 m/s', sideways='横移 0.15 m/s', diagonal='斜向 0.10 + 0.10 m/s', spin='原地转 0.8 rad/s', overspeed='前进 0.40 m/s（超速）')
    def got(k):
        g = B[k]['got']
        if k == 'spin': return f'{g[2]:.2f} rad/s'
        if k == 'diagonal': return f'{g[0]:.3f} + {g[1]:.3f} m/s'
        return f'{(g[0] if k != "sideways" else g[1]):.3f} m/s'
    base_rows = ''.join(f'<tr><td>{lab[k]}</td><td class="n">{got(k)}</td><td class="n">{B[k]["tilt_deg"]:.2f}°</td></tr>' for k in lab)

    # segmentation given by the simulator (round 7) vs depth-only segmentation (this round), same seeds and objects
    R4 = json.load(open(R('trials_round7.json')))['episodes']
    sensor_rows = ''
    for n in agg:
        e4 = [e for e in R4 if e['condition'] == n]; e5 = [e for e in T['episodes'] if e['condition'] == n]
        d5, t5 = sum(e['delivered'] for e in e5), sum(e['total'] for e in e5)
        if not e4:
            sensor_rows += f'<tr><td>{COND[n].split("：")[0]}</td><td class="n">（这一轮新加）</td><td class="n">{d5}/{t5}（{pct(d5, t5)}）</td><td class="n">{sum(e["sim_time"] for e in e5) / max(d5, 1):.0f} 秒</td></tr>'
            continue
        d4, t4 = sum(e['delivered'] for e in e4), sum(e['total'] for e in e4)
        sensor_rows += (f'<tr><td>{COND[n].split("：")[0]}</td><td class="n">{d4}/{t4}（{pct(d4, t4)}）</td><td class="n">{d5}/{t5}（{pct(d5, t5)}）</td>'
                        f'<td class="n">{sum(e["sim_time"] for e in e4) / max(d4, 1):.0f} → {sum(e["sim_time"] for e in e5) / max(d5, 1):.0f} 秒</td></tr>')
    allep = T['episodes']
    loc_mean = sum(e['loc_err_mean'] for e in allep) / len(allep); loc_max = max(e['loc_err_max'] for e in allep)
    head_max = max(e['loc_head_err_max_deg'] for e in allep); odom = sum(e['odom_err'] for e in allep) / len(allep)
    PBall = json.load(open(R('perception_bench.json'))); PB = {k: v for k, v in PBall.items() if not k.startswith('_')}
    pairs, furn, empty = PBall['_pairs'], PBall['_furniture'], PBall['_empty']
    seg_pair_rows = ''.join(f'<tr><td>{g.replace("mm", " mm").replace("<", "小于 ").replace(">", "大于 ")}</td><td class="n">{v["n"]}</td><td class="n">{v["apart"]}</td><td class="n">{v["merged"]}</td></tr>'
                            for g, v in pairs.items() if v['n'])
    seg_furn = '，'.join(f'{float(c) * 100:.0f} cm 处 {v["found"]}/{v["n"]}' for c, v in furn.items())
    seg_empty = f'在没放任何东西的房间里随机看了 {empty["looks"]} 次，误报 {empty["false_detections"]} 个'
    cl = agg.get('clutter')
    det_near = sum(PB[k]['0.235']['near_edge_err_mm'] for k in PB) / len(PB); far = [PB[k]['1.5']['centre_err_mm'] for k in PB if PB[k]['1.5']['centre_err_mm'] is not None]; det_far = sum(far) / len(far)
    det_rate = (sum(PB[k][d]['detected'] for k in PB for d in ('0.6', '1.0', '1.5')), sum(PB[k][d]['n'] for k in PB for d in ('0.6', '1.0', '1.5')))
    nw, bj = agg['near_walls'], agg['bare_jaws']
    valid = [
        ('称重', '整机和机械臂分别上秤', f'整机 {M["robot_mass"]:.2f} kg，其中机械臂 {M["arm_mass"]:.2f} kg', '差 10% 以上就改 <code>model.py</code> 的 <code>P</code> 重跑'),
        ('底盘速度', '指令 0.15 m/s，前进、横移各跑 4 秒，量实际位移', f'前进 {B["forward"]["got"][0]:.3f}，横移 {B["sideways"]["got"][1]:.3f} m/s', '在指令的 ±10% 以内'),
        ('原地旋转', '指令 0.8 rad/s 转 4 秒，量转过的角度', f'{B["spin"]["got"][2]:.2f} rad/s', '±10% 以内'),
        ('最高速度', '指令 0.4 m/s', f'到 {B["overspeed"]["got"][0]:.2f} m/s 饱和', '记下实测值，写回 <code>MAX_WHEEL_SPEED</code>'),
        ('里程计漂移', '走一个 1 m 的正方形回到起点，量闭合误差', f'按 1.5% 轮径误差、3% 打滑建模：整机试验单靠里程计平均漂 {s["odom"]:.1f} m / {s["dist"]:.0f} m 行程', '只做记录，实测值写回 <code>robot.py</code> 的 <code>ODOM_SCALE</code>、<code>ODOM_SLIP</code>'),
        ('雷达自遮挡', '收臂状态原地采一圈，数被机身滤波丢掉的点', f'{M["lidar_self_hits"]}/{M["lidar_rays"]} 线', '正前方盲区不超过仿真值 + 10 线'),
        ('定位精度', '跑一局取放，途中停 5 次，每次用卷尺核对定位输出', f'扫描匹配：平均 {100 * loc_mean:.1f} cm，最大 {100 * loc_max:.1f} cm，航向最大 {head_max:.1f}°', '平均不超过 3 cm；超了先查雷达安装和自遮挡滤波'),
        ('相机测量', '手边的小东西分别放在 0.25 m 和 1.5 m 处，用尺子核对相机量出的近侧边缘和轮廓中心', f'近侧边缘平均差 {det_near:.1f} mm，1.5 m 处轮廓中心差 {det_far:.0f} mm（<code>test_perception.py</code>）', '近侧边缘不超过 6 mm，否则夹爪会落偏'),
        ('分割', '地上摆 8 件软硬不一的东西，其中两对挨着放（间隙 1 cm、3 cm），一件贴着家具，拍 50 张，数漏掉的、两件粘成一件的、凭空多出来的', f'0.6–1.5 m 整件可见时检出 {det_rate[0]}/{det_rate[1]}；挨着放的见上表；贴着家具 {seg_furn}；{seg_empty}', '粘连和漏检不比仿真多一倍以上；地毯、深色或反光的东西单独记录'),
        ('腕部相机', '夹住袜子、毛绒玩具、方块各 10 次，空夹 10 次，看腕部相机说「夹爪里有东西」对不对', f'台架上判断错 {sum(G[m][k]["misjudged"] for m in G for k in G[m])} 次', '判断错不超过 2 次'),
        ('单物体抓取', '机器人停稳，物体在正前方 20–26 cm。找 8 件对应的实物（方块、球、玩具车、香蕉、L 形积木、石头、袜子、毛绒玩具），每件 10 次', '贴胶带：' + '，'.join(f'{CN[k].split("（")[0]} {gt[k][0]}/{gt[k][1]}' for k in CN), '每种不比仿真少 2 次以上'),
        ('软物体放下', '把袜子和毛绒玩具各放进筐 10 次，看有没有粘在夹爪上，粘住时机器人发现了没有', f'整机试验 {n_place} 次放筐里抖了 {n_shake} 次（软的 {n_shake_soft} 次）', '粘住了都要被发现'),
        ('胶带对照', '撕掉胶带，小球夹 10 次', f'{gb["ball"][0]}/{gb["ball"][1]}', '应该明显变差。如果没变差，说明仿真的摩擦假设偏保守'),
        ('整机任务', '4×3 m 场地摆 5 件，离墙 ≥30 cm，做 10 次', f'送达 {pct(s["d"], s["tot"])}，每件 {s["t"]:.0f} 秒，{s["n"]} 次试验共碰撞 {s["coll"]} 次', '送达率不低于仿真 15 个百分点，没有倾翻'),
        ('贴边任务', '同上，物体离墙 12 cm 起', f'送达 {pct(nw["d"], nw["tot"])}', '只做记录'),
        ('扎堆任务', '同上，其中两对物体挨着放', f'送达 {pct(cl["d"], cl["tot"])}' if cl else '-', '只做记录'),
    ]
    valid_rows = ''.join(f'<tr><td class="n">{i + 1}</td><td>{a}</td><td>{b}</td><td>{c}</td><td>{d}</td></tr>' for i, (a, b, c, d) in enumerate(valid))

    round_rows = ''
    for fn, name in (('trials_round1.json', '第一轮：初版控制器'), ('trials_round2.json', '第二轮：轻合夹爪、近看再对位、远侧净空检查'),
                     ('trials_round3.json', '第三轮：加了只看雷达、只管平移的防撞层'), ('trials_round4.json', '第四轮：防撞层按真实外形算，也管转身'),
                     ('trials_round5.json', '第五轮：定位和识别换成真算法'), ('trials_round6.json', '第六轮：防撞层加上贴着障碍滑行'),
                     ('trials_round7.json', '第七轮：八种不认识的物体，含软的和不规则的'),
                     ('trials.json', '第八轮：分割只用深度图，腕部深度相机确认夹持（当前版本）')):
        alle = json.load(open(R(fn)))['episodes']; eps = [e for e in alle if e['condition'] == 'standard']
        d = sum(e['delivered'] for e in eps); tot = sum(e['total'] for e in eps)
        round_rows += (f'<tr><td>{name}</td><td class="n">{d}/{tot}</td><td class="n">{sum(e["collisions"] > 0 for e in alle)} 次试验</td>'
                       f'<td class="n">{sum(e["max_tilt"] > 30 for e in alle)}</td><td class="n">{sum(e["max_tilt"] > 3 for e in alle)}/{len(alle)}</td>'
                       f'<td class="n">{sum(e["stuck"] for e in alle)}</td></tr>')
    video = json.load(open(R('episode.json')))
    rep = {
        'VIDEO_NOTE': f'第 {video["seed"]} 号随机布局，仿真时间 {video["sim_time"] / 60:.1f} 分钟，送达 {video["delivered"]}/{video["total"]}',
        'STD_DELIVERED': f'{s["d"]}/{s["tot"]}', 'STD_PCT': pct(s['d'], s['tot']), 'STD_N': str(s['n']), 'STD_TIME': f'{s["t"]:.0f}',
        'GRASP_TAPE': f'{gt_all[0]}/{gt_all[1]}', 'GRASP_TAPE_PCT': pct(*gt_all), 'GRASP_BARE': f'{gb_all[0]}/{gb_all[1]}', 'GRASP_BARE_PCT': pct(*gb_all),
        'TRIAL_ROWS': trial_rows, 'FAIL_ROWS': fail_rows, 'GRASP_N': str(G['1.0']['cube']['n'] + G['1.0']['cube']['no_stance']), 'GRASP_ROWS': grasp_rows, 'BASE_ROWS': base_rows,
        'OBJECT_ROWS': object_rows,
        'DET_MISS': '；'.join(f'{CN[k]}在 {d} m 处只检出 {PB[k][d]["detected"]}/{PB[k][d]["n"]}' for k in PB for d in ('1.0', '1.5') if PB[k][d]['detected'] < 0.8 * PB[k][d]['n']),
        'SOFT_HOLDS': f'这一轮全部试验里，靠这一眼确认的夹持有 {n_soft} 次。',
        'SHAKES': f'这一轮 {n_place} 次放筐里触发了 {n_shake} 次' + (f'，其中 {n_shake_soft} 次是软东西' if n_shake else '') + '。上一版用前置相机看「有没有东西跟着夹爪升上去」，193 次里误报了 16 次（夹爪挡住了筐里已有的东西）；改成腕部相机直接看两片夹爪之间以后，不再受筐里东西的干扰。',
        'SEG_PAIR_ROWS': seg_pair_rows, 'SEG_FURN': seg_furn, 'SEG_EMPTY': seg_empty,
        'CLUTTER': (f'{cl["d"]}/{cl["tot"]}（{pct(cl["d"], cl["tot"])}）' if cl else '-'),
        'BARE_BALL': f'{gb["ball"][0]}/{gb["ball"][1]}', 'TAPE_BALL': f'{gt["ball"][0]}/{gt["ball"][1]}', 'BARE_PCT': pct(bj['d'], bj['tot']),
        'VALID_ROWS': valid_rows, 'OTHER_N': f'{nw["n"]} 次和 {bj["n"]} 次',
        'TAPE_TITLE': '夹爪必须贴防滑胶带' if s['d'] / s['tot'] - bj['d'] / bj['tot'] >= 0.15 else '防滑胶带有用，但已经不是生死线',
        'TAPE_NOTE': '' if s['d'] / s['tot'] - bj['d'] / bj['tot'] >= 0.15 else '早几轮不贴胶带时网球几乎夹不住；这一版夹法会挑更窄、更居中的一条下手，失败了还会换一侧再试，摩擦小的影响被吃掉了大半。胶带便宜，仍然建议贴。',
        'SENSOR_ROWS': sensor_rows, 'ODOM_DRIFT': f'{odom:.1f}', 'LOC_MEAN': f'{100 * loc_mean:.1f}', 'LOC_MAX': f'{100 * loc_max:.1f}',
        'HEAD_MAX': f'{head_max:.1f}', 'DET_NEAR': f'{det_near:.1f}', 'DET_FAR': f'{det_far:.0f}',
        'STUCK5': str(sum(e['stuck'] for e in json.load(open(R('trials_round5.json')))['episodes'])), 'STUCK6': str(sum(e['stuck'] for e in T['episodes'])),
        'ROUND_ROWS': round_rows, 'NO_STANCE': f'{sum(v.get("no_stance", 0) for m in G for v in G[m].values())}',
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
    md += ['', '## 分割：挨着放的两件', '', '| 边缘间隙 | 试验 | 分得开 | 粘成一件 |', '|---|---|---|---|']
    md += [f'| {g} | {v["n"]} | {v["apart"]} | {v["merged"]} |' for g, v in pairs.items() if v['n']]
    md += ['', '## 整机试验汇总', '', '| 条件 | 试验次数 | 送达 | 五件全送达 | 每件用时 | 抓取失败/尝试 | 途中掉落 | 碰撞 | 机身倾斜超过 3° 的试验 |', '|---|---|---|---|---|---|---|---|---|']
    md += [f'| {COND[n]} | {a["n"]} | {a["d"]}/{a["tot"]}（{pct(a["d"], a["tot"])}） | {a["all5"]}/{a["n"]} | {a["t"]:.0f} 秒 | {a["gf"]}/{a["ga"]} | {a["drop"]} | {a["coll"]} | {a["tilt"]}（最大 {a["max_tilt"]:.1f}°） |' for n, a in agg.items()]
    md += ['', '## 仿真里什么是真的，什么是假设', '',
           '真物理：轮子滚子与地面的接触摩擦；舵机力矩上限 2.94 N·m 和由阻尼决定的转速上限；机械臂的质量、惯量、限位（SO-101 官方模型）；夹爪与物体的接触；雷达射线投射（1 cm 噪声、被自身遮挡）。', '',
           '定位和感知也是真算法：轮子里程计（1.5% 轮径误差、3% 打滑）加雷达扫描匹配；物体只从渲染出的深度图里找：拟合地面、去掉家具、按俯视连通分成一件件，再量轮廓和高度，不用任何已知类别或尺寸，也不用训练好的模型。夹爪里有没有东西由腕部深度相机判断。', '',
           '假设（每条都对应上表一项实测）：深度噪声只按距离建模，没有反光、深色、透明表面的缺失；机器人认得出自己的手臂（按运动学屏蔽）；收纳筐大致位置和尺寸已知；底盘重量是估的；打印件不变形、舵机无回差；软物体是多段刚体加弹性铰链和柔性接触的近似，不是布料仿真。', '',
           '## 实测后怎么回填', '', '1. 硬件参数改 `sim/model.py` 顶部的 `P`（质量、摩擦、轮径、力矩上限）。', '2. 传感器噪声改 `sim/controller.py` 顶部的常量。',
           '3. 物体的尺寸、重量、软硬在 `sim/objects.py`，可以照着手边的实物改或加新种类。', '4. 依次重跑 `python test_base.py`、`python test_perception.py`、`python test_grasp.py 10`、`python run_trials.py`，再 `python ../web/build_page.py`。', '']
    open(os.path.join(ROOT, 'docs', 'validation.md'), 'w', encoding='utf8').write('\n'.join(md))
    print('built', len(html), 'bytes;', {k: (v if len(v) < 40 else '...') for k, v in rep.items() if k not in ('TRIAL_ROWS', 'FAIL_ROWS', 'GRASP_ROWS', 'BASE_ROWS', 'VALID_ROWS', 'ROUND_ROWS')})


if __name__ == '__main__':
    main()
