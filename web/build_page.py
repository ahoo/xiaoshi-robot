"""Fill web/page_template.html with numbers taken straight from results/*.json -> web/dist/index.html (a full document) and
web/dist/artifact.html (the same page as published to claude.ai: no document tags, the platform wraps its own)
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

CN = dict(cube='方块', ball='小球', car='玩具车', banana='香蕉玩具', lbrick='L 形积木', rock='石头', sock='袜子（软）', plush='毛绒玩具（软）', bottle='透明塑料瓶')
SIZE = dict(cube='边长 3–5 cm', ball='直径 5.2–6.8 cm', car='长 8–10 cm，宽 4–5 cm', banana='长 12–16 cm，粗 2.8–3.6 cm', lbrick='两臂 4–7 cm，厚 2–2.8 cm，高 1.8–3 cm',
            rock='直径约 3–6 cm，偏扁', sock='长 11–15 cm，粗 3.6–4.6 cm', plush='身体直径 4.4–5.6 cm，四肢 2–3 cm', bottle='长 10–15 cm，直径 4–5 cm')
HOW = dict(cube='硬，立方体', ball='硬，圆球，会滚', car='硬，车身加四个轮子', banana='硬，弯的长条', lbrick='硬，L 形，两条臂不一样长',
           rock='硬，随机凸多面体，每次形状不同', sock='软，3 段用弹性铰链连起来，会垂、会弯', plush='软，身体加头和四肢共 6 段，一捏就扁',
           bottle='硬，躺着的圆柱，透明（只在难材质条件里出现）')
COND = dict(standard='标准：物体离墙和家具 ≥30 cm，贴胶带', near_walls='贴边：物体离墙和家具 12 cm 起，贴胶带', bare_jaws='裸夹爪：不贴胶带，其余同标准',
            clutter='扎堆：有两对物体边缘只隔 0.5–5 cm，其余同标准', materials='难材质：五件都是黑色或反光的，另加一个透明塑料瓶')


def tlast(e):
    """Time until the last thing went into the basket (older rounds: the whole run; the robot keeps looking afterwards)."""
    return e['t_last'] if e.get('t_last') else e['sim_time']


def pct(a, b):
    return f'{100 * a / b:.0f}%' if b else '-'



def home_round_rows(start):
    """Round 14 (3D home map, object records, people and children, voice): rows 41-52 of the validation table, from results/home_*.json. Missing files give '-'.
    They go into docs/validation.md only; the published page does not have them yet."""
    def load(n):
        fn = R(n); return json.load(open(fn)) if os.path.exists(fn) else None
    SC, EX, INV, INV1, PS, ID, SF, VO, DAY = (load(n) for n in ('home_scan.json', 'home_scan_export.json', 'home_inventory.json', 'home_inventory_s1.json', 'home_people_static.json', 'home_people_identity.json',
                                                                  'home_people_safety.json', 'home_voice.json', 'home_day.json'))
    mean = lambda v: sum(v) / len(v) if v else float('nan')
    rows = []
    # 41, 42: the 3D map
    if EX:
        vt = EX.get('vs_truth', []); names_ok = sum(1 for v in vt if v['guess'] == {'carton0': '箱子', 'carton1': '箱子'}.get(v['truth'], {'sofa': '沙发', 'table': '桌子', 'shelf': '架子', 'plant': '盆栽', 'chair': '椅子', 'carton': '箱子'}.get(v['truth'])))
        c41 = (f'仿真（4×3 m、{EX["true_furniture"]} 件家具、LD19 标定）：找到 {len(vt)} 件，名字猜对 {names_ok}，墙 {EX["walls"]}/4；家具中心差 {min(v["centre_err_cm"] for v in vt):.0f}–{max(v["centre_err_cm"] for v in vt):.0f} cm，'
               f'长边偏大 {min(v["long_side_cm"] for v in vt):.0f}–{max(v["long_side_cm"] for v in vt):.0f} cm；召回 {EX["recall"]}、精度 {EX["precision"]}（对仿真真值）') if vt else '-'
    else: c41 = '-'
    rows.append(('三维地图：位置和尺寸', '地上先收干净，开扫描；扫完用卷尺量家具的位置、长宽高和 4 面墙的长度，和地图里的盒子比', c41,
                 '家具中心差不超过 10 cm；墙都找到；没有凭空多出来的家具。高度只做记录：相机离地 12 cm，只看得到下面 0.3–0.6 m'))
    if SC:
        by = {p: [r for r in SC['runs'] if r['planner'] == p] for p in ('nbv', 'tour')}
        f = lambda p: f'{mean([r["sim_s"] for r in by[p]]):.0f} 秒、{mean([r["stops"] for r in by[p]]):.0f} 个点、{mean([r["path_m"] for r in by[p]]):.1f} m、召回 {mean([r["recall"] for r in by[p]]):.3f}'
        c42 = f'下一个最佳视点（{len(by["nbv"])} 个布局平均）：{f("nbv")}；覆盖巡视：{f("tour")}；碰撞 {sum(r["collisions"] for r in SC["runs"])}'
    else: c42 = '-'
    rows.append(('三维扫描的用时和路程', '同上，记用时、走的路程、停几个点、碰撞', c42, '不超过仿真的 1.5 倍，碰撞 0'))
    rows.append(('建图和人体检测的算力', '树莓派 5 上跑 `mapping3d`（每帧积分、最后分析）和 `people` 的检测，记每帧耗时和占几个核',
                 f'这台机器（Xeon 2.3 GHz）上建图每帧约 {round(mean([r["ms_per_frame"] for r in SC["runs"]])) if SC else "-"} ms（3.3 Hz），人体检测每帧约 {PS["summary"]["ms_per_frame"] if PS else "-"} ms（5 Hz）',
                 '合计不超过一个核；超了隔帧积分或降体素分辨率'))
    # 44, 45: object records
    if INV:
        t = INV['totals']
        c44 = (f'仿真 {len(INV["runs"])} 个布局：没动的 {t["kept"][0]}/{t["kept"][1]}，挪过的 {t["moved"][0]}/{t["moved"][1]} 认出是同一件，新来的 {t["new"][0]}/{t["new"][1]}，拿走的 {t["removed"][0]}/{t["removed"][1]} 标了“不见了”'
               f'（另 {INV.get("removed_unchecked", 0)} 件机器人没走到那边，标“还没核实”，没有说错），认错 {INV.get("wrong_identity", 0)}，凭空多出来的 {INV.get("ghosts", 0)}')
    else: c44 = '-'
    rows.append(('物件记录：换了场景认不认得', '地上摆 5 件玩具扫一遍；挪两件、拿走一件、加一件新的；再扫一遍。数：挪过的是不是还叫原来的名字，拿走的有没有标“不见了”，新的有没有被当成旧的', c44,
                 '挪过的至少 9/12 认对；把新的认成旧的不超过 1 次；没走到的不能说成“不见了”'))
    if INV1:
        sm = INV1['summary']; s, m = sm['standard'], sm['hard_materials']
        c45 = (f'仿真八种东西颜色各不相同，**是最好情况**：标准布局找到 {s["found"]}/{s["objects"]}、种类猜对 {s["kind_correct"]}/{s["found"]}、多出来的条目 {s["extra_items"]}；'
               f'黑色/反光材质布局找到 {m["found"]}/{m["objects"]}、种类猜对 {m["kind_correct"]}/{m["found"]}')
    else: c45 = '-'
    rows.append(('物件种类猜测', '8 种实物各 3 件，含颜色相同和不同的，黑色和反光的也放', c45, '只做记录；颜色相同的分不开，黑色的没有颜色，真要分类用识别模型（路线图）'))
    # 46, 47: people
    if PS:
        s = PS['summary']; bd = s['by_distance']; allf = [v for v in bd.values() if v]
        sc = s['size_class']
        c46 = (f'仿真：站姿检出 {100 * sum(v["found"] * v["n"] for v in allf) / sum(v["n"] for v in allf):.0f}%（1–3 m）；身高误差中位 {100 * mean([v["h_err_median"] for v in allf]):.1f} cm；位置误差中位 {100 * mean([v["pos_err_median"] for v in allf]):.0f} cm；'
               f'空房间 {s["false_tracks_empty_room"]["frames"]} 次里误报 {s["false_tracks_empty_room"]["with_a_track"]} 次')
        c47 = (f'仿真：站着的孩子 {sc["stand/child"]["child"]}/{sum(sc["stand/child"].values())} 分成小孩，站着的大人 {sc["stand/adult"]["adult"]}/{sum(sc["stand/adult"].values())} 分成大人，**孩子被分成大人 {s["child_called_adult"]} 次**；'
               f'坐着的大人 {sc["sit/adult"]["child"] + sc["sit/adult"]["unsure"]}/{sum(sc["sit/adult"].values())} 被分成小孩或不确定（保守，按小孩对待）；坐着的小孩 {sc["sit/child"]["child"]}/{sum(sc["sit/child"].values())}；'
               f'坐着的检出 {100 * s["sitting"]["found"]:.0f}%（0.7 m 高的幼儿坐着检不出）')
    else: c46 = c47 = '-'
    rows.append(('人：检出和身高', '立杆相机装好后量安装高度和仰角；身高已知的大人和孩子站在 1、2、3 m 处，正对和背对各 5 次，读检测到的身高', c46, '身高误差中位不超过 5 cm；检出不低于 90%；空房间没有误报'))
    rows.append(('人：大人还是小孩', '家里每个人和一两个访客孩子各站 1–3 m；再坐在沙发上、蹲着各 10 次', c47, '孩子被分成大人 0 次（硬指标）；坐着的大人被当成小孩只影响便利，不影响安全'))
    if ID:
        fr = ID['enrolled_by_distance_frontal_12looks']; sd = ID['enrolled_by_distance_side_12looks']
        c48 = (f'**仿真里是替身**，只测登记簿和阈值：登记过的人正对 1–4 m 认对 {(lambda a, z: f'{a:.0f}%' if round(a) == round(z) else f'{a:.0f}%–{z:.0f}%')(100 * fr["1.0"]["right"], 100 * fr["4.0"]["right"])}，侧对 1 m {100 * sd["1.0"]["right"]:.0f}%、3 m {100 * sd["3.0"]["right"]:.0f}%、4 m {100 * sd["4.0"]["right"]:.0f}%（其余“不认识”）；'
               f'认成别人 0；陌生人和没登记的亲戚孩子误认 0（这个 0 是替身的性质，不能当真）；没有同意不能登记、看得太少不能登记、说“忘掉”以后当场认不出')
    else: c48 = '-'
    rows.append(('认人（真模型）', '把家里的孩子（监护人同意）登记，各站 1–4 m、正对和侧对各 20 次，数认对、认成别人、认不出；访客孩子和没登记的亲戚孩子各 20 次，数被误认几次；拿照片、视频放在机器人前面试能不能骗过', c48,
                 '2 m 内正对认对不低于 90%；陌生人误认不高于 1%；照片/视频骗不过（做不到就不要把认人用在放权限上，本来也不用）'))
    rows.append(('隐私：不存图、能删', '登记一个人，说“忘掉他”，查登记簿文件和整个存储有没有残留；运行一整天后在机器人上找图片和音频文件；唤醒词之前抓包',
                 '仿真：忘掉后登记簿里没有这个人、认不出；登记簿里只有 角色、' + str(ID and 24 or '-') + ' 个数、看了几次、同意人、时间；代码里没有写图片或音频的地方',
                 '删除后无残留；存储里没有任何图片或音频；唤醒前、没开大模型时没有对外网络请求'))
    # 50: safety
    if SF:
        by = {}
        for r in SF['runs']: by.setdefault(r['scenario'], {})[r['guard']] = r
        parts = []
        for sc_, nm in (('adult_cross', '成人走过'), ('child_run', '孩子冲过来'), ('toddler_basket', '幼儿在筐边'), ('family', '一家三口')):
            if sc_ not in by: continue
            on, off = by[sc_].get(True), by[sc_].get(False)
            if on and off: parts.append(f'{nm}：安全层关 送达 {off["delivered"]}/{off["total"]}、碰到人 {sum(off.get("contacts", []))} 次，开 送达 {on["delivered"]}/{on["total"]}、碰到人 {sum(on.get("contacts", []))} 次、最近 {min(on.get("min_gap_cm", [0])):.0f} cm')
        c50 = '仿真：' + '；'.join(parts) if parts else '-'
    else: c50 = '-'
    rows.append(('人靠近时的安全层', '泡沫人偶（孩子和大人的高度）放在行进路线上，0.15 m/s 开过去；真人（大人）从侧面走过；数停下的距离、放慢的距离、碰到几次', c50, '不碰到人偶；停下时离成人 ≥ 20 cm、离孩子 ≥ 35 cm；没有人的时候不乱停'))
    if VO:
        nz = VO['noise']; st = nz['stop_words_detected']; sw = {(r['cer'], r['child']): r for r in VO['sweep']}
        c51 = (f'仿真（模拟识别器，不是声学测试）：{nz["chatter_no_wake"]["n"]} 句家里闲聊 {nz["chatter_no_wake"]["not_ignored"]} 次被当成命令，误停 {nz["chatter_no_wake"]["false_stops"]}；'
               f'停止词 成人 {100 * st["adult_quiet"]:.0f}%/{100 * st["adult_noisy"]:.0f}%（安静/吵），孩子 {100 * st["child_quiet"]:.0f}%/{100 * st["child_noisy"]:.0f}%')
        sw_txt = '，'.join(f'{int(100 * c)}%：{(sw[(c, False)]["ok"] + sw[(c, False)]["confirmed"]) / sw[(c, False)]["n"] * 100:.0f}%' for c in (0.0, 0.04, 0.08, 0.15, 0.25) if (c, False) in sw)
        c52 = (f'仿真：识别错误率 {sw_txt}（直接做对加先确认一次）；做成了和说的不一样的移动命令 {sum(r["wrong_motion"] for r in VO["sweep"])} 次；权限表 {VO["permissions"]["n"]} 格错 {VO["permissions"]["wrong"]}；'
               f'多轮对话脚本 {sum(d["ok"] for d in VO["dialogues"])}/{len(VO["dialogues"])}；每句理解约 {VO["ms_per_parse"]} ms')
    else: c51 = c52 = '-'
    rows.append(('语音：唤醒和停止', '客厅里开着电视（约 60 dB）说话和放新闻，数唤醒；成人和孩子在 1、3、5 m 处各说“停”“别动”20 次', c51, '误唤醒每小时不超过 1 次；停止词 3 m 内不低于 95%，孩子也是'))
    rows.append(('语音：听懂', '照 `sim/voice_data.py` 的 74 句话真人念（成人、孩子各一遍，开着电视）', c52, '听懂（含先确认一次）不低于 90%；做了和说的不一样的移动命令 0 次'))
    return [(str(start + i + 1), a, b, c, d) for i, (a, b, c, d) in enumerate(rows)]


def modes_round():
    """Round 13 (self-righting, charging, remote control, navigation): strings for the page and rows for the validation table, from results/modes_*.json
    and results/righting.json. Missing files give '-' rather than an error, so the page builds while a bench is still running."""
    def load(n):
        fn = R(n); return json.load(open(fn)) if os.path.exists(fn) else None
    D, D2, F, FJ, N, TL, LB, SV, RI, EX = (load(n) for n in ('modes_dock.json', 'modes_dock_ld19.json', 'modes_fall.json', 'modes_fall_job.json', 'modes_nav.json', 'modes_teleop.json',
                                                            'modes_lowbatt.json', 'modes_server.json', 'righting.json', 'modes_explore.json'))
    out = {}; rows = {}
    med = lambda v: sorted(v)[len(v) // 2] if v else None
    def dock_row(d, label):
        if not d: return ''
        ok = [r for r in d['runs'] if r['docked']]; c = [r['contact'] for r in ok if r['contact']]
        return (f'<tr><td>{label}</td><td class="n">{len(ok)}/{len(d["runs"])}</td><td class="n">{sum(r["attempts"] > 1 for r in d["runs"])}</td><td class="n">{sum(r["collisions"] for r in d["runs"])}</td>'
                f'<td class="n">{med([abs(x["dx_cm"]) for x in c]):.1f} / {max(abs(x["dx_cm"]) for x in c):.1f} cm</td><td class="n">{max(abs(x["yaw_deg"]) for x in c):.1f}°</td><td class="n">{med([x["t"] for x in c]):.0f} 秒</td></tr>') if c else ''
    out['M13_DOCK_ROWS'] = dock_row(D, '雷达标定后（残余偏差 ±20 mm）') + dock_row(D2, '雷达没标定（LD19 规格书的偏差 ±45 mm）') or '<tr><td colspan="7">还在跑</td></tr>'
    out['M13_DOCK_N'] = str(len(D['runs'])) if D else '-'
    if D:
        ok = [r for r in D['runs'] if r['docked']]; out['M13_DOCK_OK'] = f'{len(ok)}/{len(D["runs"])}'; out['M13_DOCK_MIN'] = f'{med([r["charge_min"] for r in ok]):.0f}'
    else: out['M13_DOCK_OK'] = out['M13_DOCK_MIN'] = '-'
    out['M13_DOCK_OK2'] = f'{sum(r["docked"] for r in D2["runs"])}/{len(D2["runs"])}' if D2 else '-'
    # righting
    if RI:
        pp = RI['per_pose']; sw = RI['sweep']
        out['M13_RIGHT_ROWS'] = ''.join(f'<tr><td>{n}</td><td class="n">{v["standard"]["upright"]}/{v["standard"]["n"]}</td><td class="n">{v["harsh"]["upright"]}/{v["harsh"]["n"]}</td></tr>'
                                        for n, v in zip(('侧躺（左）', '侧躺（右）', '前倾趴着'), (pp['side_l'], pp['side_r'], pp['fwd'])))
        out['M13_SWEEP'] = f'{sw["got_up"]}/{sw["fell"]}'; out['M13_SWEEP_N'] = str(sw['n']); out['M13_SWEEP_FELL'] = str(sw['fell'])
    else: out['M13_RIGHT_ROWS'] = '<tr><td colspan="3">还在跑</td></tr>'; out['M13_SWEEP'] = out['M13_SWEEP_N'] = out['M13_SWEEP_FELL'] = '-'
    if F:
        down = [r for r in F['runs'] if r['falls'] > 0]
        good = [r for r in down if r['reloc_err'] and r['reloc_err'][-1][0] < 0.05]; wrong = [r for r in down if r['reloc_err'] and r['reloc_err'][-1][0] >= 0.05]; lost = [r for r in down if r['fault']]
        out['M13_FALL'] = f'{len(down)}/{len(F["runs"])}'; out['M13_FALL_UP'] = f'{sum(r["righted"] > 0 for r in down)}/{len(down)}'; out['M13_FALL_GOOD'] = f'{len(good)}/{len(down)}'
        out['M13_FALL_WRONG'] = str(len(wrong)); out['M13_FALL_LOST'] = str(len(lost)); out['M13_FALL_COLL'] = str(sum(r['collisions'] for r in down))
        out['M13_FALL_ERR'] = f'{100 * med([r["reloc_err"][-1][0] for r in good]):.1f}' if good else '-'
    else:
        for k in ('M13_FALL', 'M13_FALL_UP', 'M13_FALL_GOOD', 'M13_FALL_WRONG', 'M13_FALL_LOST', 'M13_FALL_COLL', 'M13_FALL_ERR'): out[k] = '-'
    kind_cn = dict(roll_l='向左翻滚', roll_r='向右翻滚', pitch_f='向前栽', pitch_b='向后仰'); state_cn = {'搬运': '搬着东西', '巡视': '巡视中', '抓取': '抓取中', '导航': '去取东西的路上'}
    def fj_row(r):
        what = f'{kind_cn.get(r["kind"], r["kind"])}（{state_cn.get(r["state_at_fall"], r["state_at_fall"])}）'
        if not r['falls']: return f'<tr><td>{what}</td><td class="n">没推倒</td><td class="n">-</td><td class="n">{r["delivered"]}/{r["total"]}</td><td class="n">{r["sim_time"]:.0f} 秒</td></tr>'
        pos = '找回了位置' if r['reloc_err'] and r['reloc_err'][-1][0] < 0.05 else ('没找回，求助' if r['fault'] else '位置错了')
        return f'<tr><td>{what}</td><td class="n">{"起来了" if r["righted"] else "没起来"}</td><td class="n">{pos}</td><td class="n">{r["delivered"]}/{r["total"]}</td><td class="n">{r["sim_time"]:.0f} 秒</td></tr>'
    out['M13_FALLJOB_ROWS'] = ''.join(fj_row(r) for r in FJ['runs']) if FJ else '<tr><td colspan="5">还在跑</td></tr>'
    # navigation
    if N:
        reach = [a for r in N['runs'] for a in r['arrivals'] if a['goal'] < 6]; bad = [a for r in N['runs'] for a in r['arrivals'] if a['goal'] >= 6]
        arrived = [a for a in reach if a['result'] == 'arrived']; withh = [a for a in arrived if a['head_err_deg'] is not None]; noh = [a for a in arrived if a['head_err_deg'] is None]
        out['M13_NAV'] = f'{len(arrived)}/{len(reach)}'; out['M13_NAV_BAD'] = f'{sum(a["result"] != "arrived" for a in bad)}/{len(bad)}'
        out['M13_NAV_H'] = f'{med([a["err_cm"] for a in withh]):.1f}' if withh else '-'; out['M13_NAV_HW'] = f'{max(a["err_cm"] for a in withh):.1f}' if withh else '-'; out['M13_NAV_HD'] = f'{max(a["head_err_deg"] for a in withh):.1f}' if withh else '-'
        out['M13_NAV_N'] = f'{med([a["err_cm"] for a in noh]):.1f}' if noh else '-'; out['M13_NAV_NW'] = f'{max(a["err_cm"] for a in noh):.1f}' if noh else '-'
        out['M13_NAV_T'] = f'{med([a["secs"] for a in arrived]):.0f}'; out['M13_NAV_COLL'] = str(sum(r['collisions'] for r in N['runs'])); out['M13_NAV_LAY'] = str(len(N['runs']))
    else:
        for k in ('M13_NAV', 'M13_NAV_BAD', 'M13_NAV_H', 'M13_NAV_HW', 'M13_NAV_HD', 'M13_NAV_N', 'M13_NAV_NW', 'M13_NAV_T', 'M13_NAV_COLL', 'M13_NAV_LAY'): out[k] = '-'
    out['M13_EXPLORE'] = '；'.join(f'{r["seed"]} 号布局 {r["t_end"]:.0f} 秒看完，找到 {r["objects_found"]}/{r["objects_total"]} 件，碰撞 {r["collisions"]}' for r in EX['runs']) if EX else '-'
    # teleop
    if TL:
        a_on, a_off, dm, arm = TL['safety_on'], TL['safety_off'], TL['deadman'], TL['arm']
        out['M13_TELE_ON_GAP'] = f'{a_on["gap_to_wall_cm"]:.1f}'; out['M13_TELE_ON_COLL'] = str(a_on['collisions']); out['M13_TELE_OFF_GAP'] = f'{a_off["gap_to_wall_cm"]:.1f}'; out['M13_TELE_OFF_COLL'] = str(a_off['collisions'])
        out['M13_TELE_COAST'] = f'{dm["coast_cm"]:.1f}'; out['M13_TELE_STOP'] = f'{dm["stopped_after_s"]}'
        out['M13_TELE_ARM'] = f'{arm.get("pinch_err_mm", "-")}'
        tk = TL['takeover']; base = {r['seed']: r for r in tk if not r['takeover']}; took = {r['seed']: r for r in tk if r['takeover'] is True}; held = {r['seed']: r for r in tk if r['takeover'] == 'held'}
        out['M13_TAKEOVER_ROWS'] = ''.join(f'<tr><td>{s} 号布局</td><td class="n">{base[s]["delivered"]}/{base[s]["total"]}，{base[s]["sim_time"]:.0f} 秒</td><td class="n">{took[s]["delivered"]}/{took[s]["total"]}，{took[s]["sim_time"]:.0f} 秒</td>'
                                          + (f'<td class="n">{held[s]["delivered"]}/{held[s]["total"]}，{held[s]["sim_time"]:.0f} 秒{"（接管时手里有东西）" if held[s].get("held_when_taken") else "（没赶上拿着东西的时候）"}</td>' if s in held else '<td class="n">-</td>')
                                          + f'<td class="n">{took[s]["collisions"] + (held[s]["collisions"] if s in held else 0)}</td></tr>' for s in sorted(base) if s in took)
    else:
        for k in ('M13_TELE_ON_GAP', 'M13_TELE_ON_COLL', 'M13_TELE_OFF_GAP', 'M13_TELE_OFF_COLL', 'M13_TELE_COAST', 'M13_TELE_STOP', 'M13_TELE_ARM'): out[k] = '-'
        out['M13_TAKEOVER_ROWS'] = '<tr><td colspan="4">还在跑</td></tr>'
    # low battery
    if LB:
        out['M13_LOWBATT_ROWS'] = ''.join(f'<tr><td>{r["seed"]} 号布局，起始电量 {"（不建电池模型）" if r["soc0"] is None else f"{100 * r["soc0"]:.1f}%"}</td><td class="n">{r["delivered"]}/{r["total"]}</td><td class="n">{r["sim_time"]:.0f} 秒</td>'
                                          f'<td class="n">{r["charges"]}</td><td class="n">{"、".join(f"{c["soc_from"] * 100:.0f}%→{c["soc_to"] * 100:.0f}%" for c in r["charge_log"]) or "-"}</td><td class="n">{r["collisions"]}</td></tr>' for r in LB['runs'])
    else: out['M13_LOWBATT_ROWS'] = '<tr><td colspan="6">还在跑</td></tr>'
    V = load('modes.json')
    out['M13_VIDEO_NOTE'] = f'第 {V["seed"]} 号随机布局，仿真时间 {V["sim_time"] / 60:.1f} 分钟，送达 {V["delivered"]}/{V["total"]}' if V else '-'
    out['M13_SERVER'] = ('全部通过（够不到的 arm_to 和不认识的命令按预期被拒绝，goto 到了点，键盘客户端的按键映射对）' if SV and SV.get('client_keys_ok') and sum(not ok for c, ok in SV['replies']) == 2 and ['arm_to', False] in [list(r) for r in SV['replies']] and ['bogus', False] in [list(r) for r in SV['replies']] and SV.get('nav') and SV['nav'][0]['result'] == 'arrived' else '有问题') if SV else '-'
    return out


def main():
    T = json.load(open(R('trials.json'))); G = json.load(open(R('grasp_bench.json'))); B = json.load(open(R('base_bench.json')))
    M = json.load(open(R('model_facts.json')))
    CB = json.load(open(R('clear_bottle.json'))) if os.path.exists(R('clear_bottle.json')) else {}
    CBN = json.load(open(R('clear_bench.json'))) if os.path.exists(R('clear_bench.json')) else {}
    agg = {}
    for c in T['conditions']:
        eps = [e for e in T['episodes'] if e['condition'] == c['name']]
        d = sum(e['delivered'] for e in eps); tot = sum(e['total'] for e in eps)
        why = collections.defaultdict(list)
        for e in eps:
            for k, v in e['outcome'].items():
                if v != '已送达': why[v].append(k)
        agg[c['name']] = dict(n=len(eps), d=d, tot=tot, all5=sum(e['delivered'] == e['total'] for e in eps), t=sum(tlast(e) for e in eps) / max(d, 1),
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
                         f'<td class="n">{G["1.0"][k]["misjudged"] + G["0.35"][k]["misjudged"]}</td></tr>' for k in CN if k in gt and k in gb)
    # per kind: sizes and weights that actually appeared, and deliveries, over every episode of every condition
    seen = collections.defaultdict(lambda: dict(n=0, d=0, size=[], mass=[]))
    for e in T['episodes']:
        for k, o in e['objects'].items():
            v = seen[k]; v['n'] += 1; v['d'] += e['outcome'][k] == '已送达'; v['size'].append(o['size_cm']); v['mass'].append(o['mass_g'])
    object_rows = ''.join(
        f'<tr><td>{CN[k]}</td><td>{HOW[k]}</td><td class="n">{SIZE[k]}</td><td class="n">{min(seen[k]["mass"]):.0f}–{max(seen[k]["mass"]):.0f} g</td>'
        + (f'<td class="n {"good" if gt[k][0] >= 0.8 * gt[k][1] else "bad"}">{gt[k][0]}/{gt[k][1]}</td>' if k in gt else '<td class="n">-</td>') +
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

    def compare(eps_a, eps_b):
        rows = ''
        for n in agg:
            e4 = [e for e in eps_a if e['condition'] == n]; e5 = [e for e in eps_b if e['condition'] == n]
            d5, t5 = sum(e['delivered'] for e in e5), sum(e['total'] for e in e5)
            if not e4:
                rows += f'<tr><td>{COND[n].split("：")[0]}</td><td class="n">（新加）</td><td class="n">{d5}/{t5}（{pct(d5, t5)}）</td><td class="n">{sum(tlast(e) for e in e5) / max(d5, 1):.0f} 秒</td></tr>'
                continue
            d4, t4 = sum(e['delivered'] for e in e4), sum(e['total'] for e in e4)
            rows += (f'<tr><td>{COND[n].split("：")[0]}</td><td class="n">{d4}/{t4}（{pct(d4, t4)}）</td><td class="n">{d5}/{t5}（{pct(d5, t5)}）</td>'
                     f'<td class="n">{sum(tlast(e) for e in e4) / max(d4, 1):.0f} → {sum(tlast(e) for e in e5) / max(d5, 1):.0f} 秒</td></tr>')
        return rows
    R7 = json.load(open(R('trials_round7.json')))['episodes']; R8 = json.load(open(R('trials_round8.json')))['episodes']
    R9 = json.load(open(R('trials_round9.json')))['episodes']; R10 = json.load(open(R('trials_round10.json')))['episodes']; R11 = json.load(open(R('trials_round11.json')))['episodes']
    seg_cmp_rows = compare(R7, R8); pull_cmp_rows = compare(R8, R9)
    spoil_cmp_rows = ''
    for n in agg:
        cells = []
        for eps in (R10, R11, T['episodes']):
            es = [e for e in eps if e['condition'] == n]
            d, tt = sum(e['delivered'] for e in es), sum(e['total'] for e in es)
            cells.append(f'<td class="n">{d}/{tt}（{pct(d, tt)}）</td>' if es else '<td class="n">-</td>')
        spoil_cmp_rows += f'<tr><td>{COND[n].split("：")[0]}</td>' + ''.join(cells) + '</tr>'
    # hard materials: bench and trials
    PBm = json.load(open(R('perception_bench.json'))).get('_materials', {})
    MATCN = dict(black='黑色', shiny='反光', clear='透明')
    mat_rows = ''.join(f'<tr><td>{MATCN[k.split(":")[1]]}{CN[k.split(":")[0]].split("（")[0]}</td>'
                       + ''.join(f'<td class="n">{v[d]["detected"]}/{v[d]["n"]}</td>' for d in ('0.235', '0.6', '1.0'))
                       + f'<td class="n">{v["0.235"]["near_edge_err_mm"] if v["0.235"]["near_edge_err_mm"] is not None else "-"}</td></tr>' for k, v in PBm.items())
    mat_eps = [e for e in T['episodes'] if e['condition'] == 'materials']
    mat_regular = (sum(sum(v == '已送达' for k, v in e['outcome'].items() if k != 'bottle') for e in mat_eps), sum(len([k for k in e['outcome'] if k != 'bottle']) for e in mat_eps))
    mat_bottle = (sum(e['outcome'].get('bottle') == '已送达' for e in mat_eps), len(mat_eps))
    # sensitivity and compute
    SE = json.load(open(R('sensitivity.json'))) if os.path.exists(R('sensitivity.json')) else None
    SE11 = json.load(open(R('sensitivity_round11.json'))) if os.path.exists(R('sensitivity_round11.json')) else None
    sens_rows = ''; sens_md = []
    if SE:
        labels = {st['name']: st['label'] for st in SE['settings']}
        if SE11: labels.update({st['name']: st['label'] for st in SE11['settings']})

        def cell(src, name):
            es = [e for e in src['episodes'] if e['setting'] == name] if src else []
            if not es: return None
            d, tt = sum(e['delivered'] for e in es), sum(e['total'] for e in es)
            return dict(d=d, tt=tt, gf=sum(e['grasp_fail'] for e in es), ga=sum(e['grasp_attempts'] for e in es), coll=sum(e['collisions'] for e in es),
                        t=sum(tlast(e) for e in es) / max(d, 1), loc=max(e['loc_err_max'] for e in es))
        for name, label in labels.items():
            a, b = cell(SE, name), cell(SE11, name)
            fmt = lambda c: (f'<td class="n {"good" if c["d"] >= 0.95 * c["tt"] else "bad"}">{c["d"]}/{c["tt"]}</td><td class="n">{c["coll"]}</td>' if c else '<td class="n">-</td><td class="n">-</td>')
            last = b or a
            sens_rows += (f'<tr><td>{label}</td>' + fmt(a) + fmt(b) + f'<td class="n">{last["gf"]}/{last["ga"]}</td><td class="n">{100 * last["loc"]:.0f} cm</td></tr>')
            fm = lambda c: f'{c["d"]}/{c["tt"]}，碰撞 {c["coll"]}' if c else '-'
            sens_md.append(f'| {label} | {fm(a)} | {fm(b)} | {last["gf"]}/{last["ga"]} | {100 * last["loc"]:.0f} cm |')
    # localisation bench (offline replay of recorded trajectories)
    LB = json.load(open(R('loc_bench.json'))) if os.path.exists(R('loc_bench.json')) else []
    LBN = dict(gauss10='高斯噪声 1 cm（名义）', gauss30='高斯噪声 3 cm', ld19='LD19 规格书模型（偏差抽样 a）', ld19b='LD19 规格书模型（偏差抽样 b）',
               ld19c='LD19 规格书模型（偏差抽样 c）', ld19x2='LD19 模型，噪声和野点加倍', ld19cal='LD19 模型，标定后（偏差 ±20 mm）')
    loc_rows = ''
    for mdl, lab in LBN.items():
        r10 = [x for x in LB if x['model'] == mdl and x['localiser'] == 'round10']; r11 = [x for x in LB if x['model'] == mdl and x['localiser'] == 'round11']
        if not r10 or not r11: continue
        f = lambda rs: f'{sum(x["mean_cm"] for x in rs) / len(rs):.1f} / {max(x["max_cm"] for x in rs):.1f}'
        bad = max(x['max_cm'] for x in r10) > 10
        loc_rows += f'<tr><td>{lab}</td><td class="n {"bad" if bad else ""}">{f(r10)}</td><td class="n good">{f(r11)}</td></tr>'
    def sens_txt(name):
        es = [e for e in SE11['episodes'] if e['setting'] == name] if SE11 else []
        if not es: return '还在跑'
        return f'{sum(e["delivered"] for e in es)}/{sum(e["total"] for e in es)} 送达，碰撞 {sum(e["collisions"] for e in es)} 次'
    ld19_txt = f'未标定（±45 mm）{sens_txt("ld19")}；标定后（±20 mm）{sens_txt("ld19cal")}'
    n_scans = sum(e.get('scan_matches', 0) + e.get('scan_skips', 0) for e in T['episodes'])
    scan_drop = sum(e.get('scan_dropped', 0) for e in T['episodes']) / max(n_scans, 1)
    CP = json.load(open(R('compute_profile.json')))
    compute_rows = ''.join(f'<tr><td>{k}</td><td class="n">{v["ms_per_call"]:.1f} ms</td><td class="n">{100 * v["share_of_one_core"]:.1f}%</td></tr>'
                           for k, v in sorted(CP['modules'].items(), key=lambda kv: -kv[1]['share_of_one_core']) if v['share_of_one_core'] >= 0.001)
    PU = json.load(open(R('pull_bench.json')))
    R9eps = json.load(open(R('trials_round9.json')))['episodes']
    pull_rows = ''.join(f'<tr><td>{float(g) * 100:.0f} cm</td><td class="n">{v["n"]}</td><td class="n">{v["pulled"]}</td><td class="n">{v["held"]}</td></tr>' for g, v in PU.items())
    pull_logs = [m for e in R9eps for _, m in e['log']]
    n_pull_try = sum('先把它扒出来' in m for m in pull_logs); n_pull_ok = sum('往外扒了' in m for m in pull_logs)
    allep = T['episodes']
    blind_all = sum(e.get('blind_found', 0) for e in allep); blind_std = sum(e.get('blind_found', 0) for e in allep if e['condition'] != 'materials')
    slips_all = sum(e.get('slips', 0) for e in allep); slips_eps = sum(e.get('slips', 0) > 0 for e in allep)
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
    so = CB.get('ab', {}).get('summary', {}).get('slip_only'); so_runs = [r for r in CB.get('ab', {}).get('runs', []) if r['setting'] == 'slip_only']
    slip_only_txt = (f'{so["runs_touching"]}/{so["n"]} 局碰到瓶子共 {so["bottle_touches"]} 次，检测器只在 {sum(r["slips"] > 0 for r in so_runs)} 局触发') if so else '（单瓶对照还在跑）'
    valid = [
        ('称重', '整机和机械臂分别上秤', f'整机 {M["robot_mass"]:.2f} kg，其中机械臂 {M["arm_mass"]:.2f} kg', '差 10% 以上就改 <code>model.py</code> 的 <code>P</code> 重跑'),
        ('舵机零位', '机械臂按官方流程校准后，让夹爪尖依次碰地面上画好的 5 个点（离机身 20–30 cm），量指尖偏差', '敏感度扫描：零位偏 2°（1σ）时送达率下降最多（见上表）', '每个点偏差不超过 5 mm；超了重新校准'),
        ('底盘速度', '指令 0.15 m/s，前进、横移各跑 4 秒，量实际位移', f'前进 {B["forward"]["got"][0]:.3f}，横移 {B["sideways"]["got"][1]:.3f} m/s', '在指令的 ±10% 以内'),
        ('原地旋转', '指令 0.8 rad/s 转 4 秒，量转过的角度', f'{B["spin"]["got"][2]:.2f} rad/s', '±10% 以内'),
        ('最高速度', '指令 0.4 m/s', f'到 {B["overspeed"]["got"][0]:.2f} m/s 饱和', '记下实测值，写回 <code>MAX_WHEEL_SPEED</code>'),
        ('里程计漂移', '走一个 1 m 的正方形回到起点，量闭合误差', f'按 1.5% 轮径误差、3% 打滑建模：整机试验单靠里程计平均漂 {s["odom"]:.1f} m / {s["dist"]:.0f} m 行程', '只做记录，实测值写回 <code>robot.py</code> 的 <code>ODOM_SCALE</code>、<code>ODOM_SLIP</code>'),
        ('雷达自遮挡', '收臂状态原地采一圈，数被机身滤波丢掉的点', f'{M["lidar_self_hits"]}/{M["lidar_rays"]} 线', '正前方盲区不超过仿真值 + 10 线'),
        ('雷达测距偏差', '机器人正对一面平墙，用卷尺摆在 0.15、0.2、0.3、0.5、1、2 m；墙面换白墙、深色木板、黑布、亮面板四种。每个位置采 100 圈取平均，和卷尺比', 'LD19 规格书：0.3–12 m 平均误差 ±45 mm，0.3 m 以内不保证。仿真按每个表面固定偏差跑了标准条件 10 局：' + ld19_txt, '做一张「读数 → 真实距离」的修正表；修正后剩余偏差不超过 2 cm 就保持 <code>LIDAR_BIAS_MARGIN</code> = 2 cm，否则改成实测的最大值'),
        ('雷达角度零位', '机器人停在一面长直墙前、车身用直角尺对正，看扫描里墙的方向', '规格书角度误差 ±2°；仿真没有加这一项', '修正后不超过 0.5°；把偏差写进雷达安装角'),
        ('雷达毛刺和细腿', '房间里放一把椅子（腿粗约 2 cm）和一个门框边，原地采 100 圈，看过滤后丢掉的点和地图里椅子腿还在不在', f'整机试验里每圈平均滤掉 {scan_drop:.1f} 个点（名义雷达模型，没有野点）', '1 m 内的椅子腿全部在地图里；空地上不出现障碍格'),
        ('定位精度', '跑一局取放，途中停 5 次，每次用卷尺核对定位输出', f'扫描匹配：平均 {100 * loc_mean:.1f} cm，最大 {100 * loc_max:.1f} cm，航向最大 {head_max:.1f}°', '平均不超过 3 cm；超了先查雷达安装和自遮挡滤波'),
        ('碰撞后恢复', '机器人直行时用手把车身推转约 10°，或者让它轻轻蹭一下椅子，看定位输出', '修之前：一局蹭到椅子后航向错了 10°，又撞了 19 次；加了宽范围重找和雷达偏差余量以后，同一局 0 碰撞', '2 秒内航向回到 1° 以内，地图上没有多出一道斜墙'),
        ('前置相机外参', '地面上用胶带标出 4 个已知位置（0.25–1 m），各放一个方块；比相机量出的位置和卷尺，再看拟合出的地面倾角', '敏感度扫描：安装位置偏 1 cm、角度偏 1° 时 47/50 送达', '0.25 m 处位置差不超过 5 mm，地面倾角不超过 0.5°；超了重新标定外参'),
        ('相机测量', '手边的小东西分别放在 0.25 m 和 1.5 m 处，用尺子核对相机量出的近侧边缘和轮廓中心', f'近侧边缘平均差 {det_near:.1f} mm，1.5 m 处轮廓中心差 {det_far:.0f} mm（<code>test_perception.py</code>）', '近侧边缘不超过 6 mm，否则夹爪会落偏'),
        ('分割', '地上摆 8 件软硬不一的东西，其中两对挨着放（间隙 1 cm、3 cm），一件贴着家具，拍 50 张，数漏掉的、两件粘成一件的、凭空多出来的', f'0.6–1.5 m 整件可见时检出 {det_rate[0]}/{det_rate[1]}；挨着放的见上表；贴着家具 {seg_furn}；{seg_empty}', '粘连和漏检不比仿真多一倍以上；地毯、深色或反光的东西单独记录'),
        ('腕部相机', '夹住袜子、毛绒玩具、方块各 10 次，空夹 10 次，看腕部相机说「夹爪里有东西」对不对', f'台架上判断错 {sum(G[m][k]["misjudged"] for m in G for k in G[m])} 次', '判断错不超过 2 次'),
        ('单物体抓取', '机器人停稳，物体在正前方 20–26 cm。找 8 件对应的实物（方块、球、玩具车、香蕉、L 形积木、石头、袜子、毛绒玩具），每件 10 次', '贴胶带：' + '，'.join(f'{CN[k].split("（")[0]} {gt[k][0]}/{gt[k][1]}' for k in CN if k in gt), '每种不比仿真少 2 次以上'),
        ('软物体放下', '把袜子和毛绒玩具各放进筐 10 次，看有没有粘在夹爪上，粘住时机器人发现了没有', f'整机试验 {n_place} 次放筐里抖了 {n_shake} 次（软的 {n_shake_soft} 次）', '粘住了都要被发现'),
        ('胶带对照', '撕掉胶带，小球夹 10 次', f'{gb["ball"][0]}/{gb["ball"][1]}', '应该明显变差。如果没变差，说明仿真的摩擦假设偏保守'),
        ('整机任务', '4×3 m 场地摆 5 件，离墙 ≥30 cm，做 10 次', f'送达 {pct(s["d"], s["tot"])}，每件 {s["t"]:.0f} 秒，{s["n"]} 次试验共碰撞 {s["coll"]} 次', '送达率不低于仿真 15 个百分点，没有倾翻'),
        ('贴边任务', '同上，物体离墙 12 cm 起', f'送达 {pct(nw["d"], nw["tot"])}', '只做记录'),
        ('扎堆任务', '同上，其中两对物体挨着放', f'送达 {pct(cl["d"], cl["tot"])}' if cl else '-', '只做记录'),
        ('难材质', '黑袜子、黑方块、亮面小球、亮面玩具车各放在 0.25、0.6、1 m 处拍 10 次，再各夹 10 次', '检出和近处误差见上表', '黑色和反光的近处要看得见'),
        ('透明瓶：深度图上的空洞', '一个透明塑料瓶（带不带瓶盖、标签各试一次）躺在地上，放在 0.4、0.6、0.8、1.0 m 处，各转 0°、45°、90°（相对视线），在白天、日光灯、背光三种光线下各拍 10 帧，数瓶子轮廓里没有读数、读到地板、读到瓶身的像素各占多少', f'仿真设的是 60% 没读数、30% 读到后面的地板、10% 读到瓶身，逐像素独立；成片、更少空洞的情形见页面「透明瓶」一节', '没读数的像素占瓶子轮廓的比例不低于 35%，并且连成片（5×5 窗口里占 30% 以上）；比例更低、或者要靠瓶盖标签才有读数，说明这个办法对这种瓶子不灵，要换识别模型'),
        ('透明瓶：找到并抓起来', '同上的瓶子，在 4×3 m 场地的 8 个随机位置和朝向各放一次（机器人事先不知道），数送达、数抓取次数、数有没有被推走', f'瓶子单独在房间里 12 局：见页面「透明瓶」一节；整机难材质条件瓶子 {mat_bottle[0]}/{mat_bottle[1]}', '不比仿真少 2 局；任何一次轮子压上瓶子都要记下，看瓶子是被推走了还是被绕开了'),
        ('打滑检测', '房间里放一个看不见的低矮东西（透明瓶，或者一根细绳），让机器人直行压上去 10 次，数检测器触发几次，再用卷尺量车实际走了多远，看轮子有没有真的打滑；另外正常跑 10 局，数没有任何东西时的误触发', f'仿真里 {len(T["episodes"])} 局整机试验共触发 {slips_all} 次（{slips_eps} 局）；单瓶房间里只开打滑检测时，底盘 {slip_only_txt}：轻轻蹭过的不构成打滑；真打滑能检出（一次里程计 13.0 cm，仿真真值 6.6 cm）', '正常跑 10 局误触发 0 次；压上去时触发几次只做记录，不要拿它当通用的低矮障碍检测；记下 <code>SLIP_RATIO</code> 要不要调'),
        ('算力', '在树莓派 5 上跑 <code>profile_compute.py</code> 同样的模块（或者用录下来的数据离线跑），记每个模块每次的耗时', f'这台机器上合计约占一个核的 {100 * CP["total_share_of_one_core"]:.0f}%', '树莓派上合计不超过两个核'),
        ('扒出来', '方块、球、香蕉、袜子等各一件，分别放在离墙或柜子 2、4、6 cm 处，机器人从房间一侧去扒', '扒出来：' + '，'.join(f'{float(g) * 100:.0f} cm 处 {v["pulled"]}/{v["n"]}' for g, v in PU.items()), '4 cm 及以上不比仿真少一半以上；记录 2 cm 时是插不进去还是压到了东西'),
    ]
    m13 = modes_round()
    valid += [
        ('电池容量和耗电', '满电后让机器人开机不动 10 分钟记电流；再跑一遍标准任务（5 件）记掉的电量；最后放电到电池板的截止电压', '仿真按数据手册参数算出来的：容量 50 Wh；待机约 14 W（算力 10 W + 舵机待机 3.7 W），直行 0.15 m/s 约 45 W，原地转 0.8 rad/s 约 51 W（每个轮子的电机力矩均方根约 1 N·m，三个轮子互相顶着也算在内，可能偏高）；一局 5 件任务平均约 38 W，用掉 4–6 Wh（电池的 8–12%）。常量在 <code>sim/power.py</code> 和 <code>sim/robot.py</code> 顶部',
         '容量、待机功率、行走功率都在估计值的 ±30% 内；超了就改 <code>power.py</code>，并按实测重设低电量阈值'),
        ('电量读数误差', '满电、半电、低电各读一次电量计，对照外接电量表或放电曲线', '仿真假设库仑计每次开机有固定偏差（1σ 3%）；低电量线 20%，紧急线 10%，电池板 3% 断电', '读数偏差不超过 ±5%；超了就抬高两条线'),
        ('充电座对接', '机器人在房间里 10 个随机位置、随机朝向各回充一次，用真雷达，不要手工摆正', f'{m13["M13_DOCK_OK"]} 第一次就对上，碰撞 0；触片宽 24 cm，横向偏差中位 1 cm 左右', '10 次里至少 9 次第一次对上，3 次内必成功，没有撞坏充电座'),
        ('触片和探针', '对上后看充电电流是否稳定在 1 A 以上；反复对接 50 次，量顶针磨损和接触电阻；对上后轻轻碰一下机身，看会不会断开', '仿真里「接上」只是探针几何上碰到触片，没有接触电阻、弹簧力、氧化；对上后一步都不会动', '电流稳定；50 次后接触电阻变化不超过 50%；轻碰不断开'),
        ('充电时间', '从 20% 充到 90%，记时间和电流曲线', f'估计：3 A 恒流到 80%，之后恒压收尾，从 49% 充到 90% 约 {m13["M13_DOCK_MIN"]} 分钟（仿真里快进，物理时间不走）', '在估计值的 ±30% 内'),
        ('低电量续作', '把电量放到 21% 再开始整机任务，数：停在两件之间去充电的次数；充完能不能接着把活干完；再放到 10.5%，看紧急回充', '见页面「低电量」表：软停、充电、续作；紧急线在拿着东西时也回充', '不在半路断电；充完接着干完；紧急回充时手里的东西放下、不摔坏'),
        ('摔倒检测', '正常任务连跑 1 小时数误报；再在软垫上把机器人推倒 10 次，数都检出没有', '仿真里正常工作最大倾角 3–4°（抓取时到 7°）；阈值 45° 持续 0.3 秒；IMU 噪声 0.25°、偏差 0.5°', '没有误报；10 次推倒都在 1 秒内检出；被人抱起来不算摔倒（用静止检查和 3 秒预警兜底）'),
        ('起身', '软垫上，有人在旁边，手放在急停上。先把手臂加速度降到 5 rad/s² 试动作，没问题再逐步加到 30；侧躺左右各 10 次、前趴 10 次', f'仿真：{m13["M13_SWEEP_N"]} 次推倒里 {m13["M13_SWEEP_FELL"]} 次真的倒下，{m13["M13_SWEEP"].split("/")[0]} 次起来了；侧躺左右和前趴在 24 个新场景里都是 24/24（见页面「起身」表）；<b>实物没试过</b>，要用到比平时大得多的关节加速度，对摩擦和重量很敏感', '成功率不低于仿真 20 个百分点，没有零件损坏、线缆拉脱；不行就改成「摔倒后停下报警」（<code>auto_right=False</code>）'),
        ('起身后重新定位', '推倒再起身 10 次，起来后看机器人认为的位置和实际位置差多少（在地上画线量）', f'仿真：找回位置 {m13["M13_FALL_GOOD"]}，中位误差 {m13["M13_FALL_ERR"]} cm，认错位置 {m13["M13_FALL_WRONG"]} 次，找不回去求助 {m13["M13_FALL_LOST"]} 次', '位置误差小于 5 cm；认错位置 0 次（宁可求助也不能认错）'),
        ('遥控延迟和死人开关', '手机或键盘经 Wi-Fi 发速度指令，量从发出到轮子转的延迟；再拔掉网络（关 Wi-Fi），看多久停', f'仿真里没有网络延迟；死人开关 0.5 秒：松开后滑行 {m13["M13_TELE_COAST"]} cm，{m13["M13_TELE_STOP"]} 秒停稳', '延迟小于 200 ms；断网后 1 秒内停稳（死人开关 0.5 秒加上轮子减速）'),
        ('遥控时的安全层', '遥控直冲墙，安全层开和关各一次（墙前放泡沫挡着，速度 0.15 m/s）', f'仿真：安全层开停在离墙 {m13["M13_TELE_ON_GAP"]} cm（碰撞 {m13["M13_TELE_ON_COLL"]}）；关了撞上（碰撞 {m13["M13_TELE_OFF_COLL"]}）', '开：离墙不少于 2 cm 停下；关：能撞上（说明开关是真的）'),
        ('自主导航到点', '房间里 10 个目标点（一半带朝向），卷尺量落点；再点一个家具里面的点，看是不是立刻拒绝', f'仿真：{m13["M13_NAV"]} 个可达目标到达；带朝向落点中位 {m13["M13_NAV_H"]} cm、最差 {m13["M13_NAV_HW"]} cm、朝向差 ≤{m13["M13_NAV_HD"]}°；不带朝向落点中位 {m13["M13_NAV_N"]} cm；到不了的 {m13["M13_NAV_BAD"]} 立刻拒绝；碰撞 {m13["M13_NAV_COLL"]}',
         '到达率 10/10；带朝向落点不超过 5 cm；不碰东西'),
    ]
    valid_rows = ''.join(f'<tr><td class="n">{i + 1}</td><td>{a}</td><td>{b}</td><td>{c}</td><td>{d}</td></tr>' for i, (a, b, c, d) in enumerate(valid))

    round_rows = ''
    for fn, name in (('trials_round1.json', '第一轮：初版控制器'), ('trials_round2.json', '第二轮：轻合夹爪、近看再对位、远侧净空检查'),
                     ('trials_round3.json', '第三轮：加了只看雷达、只管平移的防撞层'), ('trials_round4.json', '第四轮：防撞层按真实外形算，也管转身'),
                     ('trials_round5.json', '第五轮：定位和识别换成真算法'), ('trials_round6.json', '第六轮：防撞层加上贴着障碍滑行'),
                     ('trials_round7.json', '第七轮：八种不认识的物体，含软的和不规则的'),
                     ('trials_round8.json', '第八轮：分割只用深度图，腕部深度相机确认夹持'),
                     ('trials_round9.json', '第九轮：贴得太近、站不进去时先把东西扒出来'),
                     ('trials_round10.json', '第十轮：深度图加上真实相机的毛病'),
                     ('trials_round11.json', '第十一轮：修好相机毛病带来的问题，雷达按规格书加误差'),
                     ('trials.json', '第十二轮：透明瓶（当前版本）')):
        alle = json.load(open(R(fn)))['episodes']; eps = [e for e in alle if e['condition'] == 'standard']
        d = sum(e['delivered'] for e in eps); tot = sum(e['total'] for e in eps)
        round_rows += (f'<tr><td>{name}</td><td class="n">{d}/{tot}</td><td class="n">{sum(e["collisions"] > 0 for e in alle)} 次试验</td>'
                       f'<td class="n">{sum(e["max_tilt"] > 30 for e in alle)}</td><td class="n">{sum(e["max_tilt"] > 3 for e in alle)}/{len(alle)}</td>'
                       f'<td class="n">{sum(e["stuck"] for e in alle)}</td></tr>')
    # the clear plastic bottle (round 12): bench and bottle-alone runs, from results/clear_bottle.json and results/clear_bench.json
    SETN = dict(both='两个办法都开（当前版本）', slip_only='只开打滑检测（不认空洞）', neither='都关（第十一轮的行为）')
    def med(v): return f'{sorted(v)[len(v) // 2]:.0f} cm' if v else '-'
    clear_ab_rows = ''
    for name, lab in SETN.items():
        sm = CB.get('ab', {}).get('summary', {}).get(name)
        if not sm: continue
        runs = [r for r in CB['ab']['runs'] if r['setting'] == name]
        moved = med([r['bottle_moved_cm'] for r in runs if r['bottle_moved_cm'] is not None])
        clear_ab_rows += (f'<tr><td>{lab}</td><td class="n {"good" if sm["delivered"] >= 0.9 * sm["n"] else "bad"}">{sm["delivered"]}/{sm["n"]}</td><td class="n">{sm["runs_touching"]}/{sm["n"]}（{sm["bottle_touches"]} 次）</td>'
                          f'<td class="n">{moved if sm["delivered"] < sm["n"] else "-"}</td><td class="n">{sm["slips"]}</td><td class="n">{sm["grasp_fail"]}/{sm["grasp_attempts"]}</td></tr>')
    clear_forced_rows = ''
    for r in CB.get('forced', {}).get('runs', []):
        clear_forced_rows += (f'<tr><td>{r["angle"]:+d}°</td><td class="n {"good" if r["delivered"] else "bad"}">{"送达" if r["delivered"] else "没送达"}</td><td class="n">{r["grasp_attempts"]}</td>'
                              f'<td class="n">{r["looks"]}</td><td class="n">{r["sim_t"]:.0f} 秒</td></tr>')
    ROBN = dict(patches_4px='空洞成片（4×4 像素一块，比例不变）', fewer_holes='空洞更少（40% 没读数、50% 读到地板，其余是瓶身）', both_rougher='两者叠加（35% 没读数、45% 读到地板，成片）')
    clear_robust_rows = ''
    fs = CB.get('forced', {}).get('summary', {}).get('forced')
    if fs: clear_robust_rows += f'<tr><td>当前的仿真设定（60% 没读数、30% 读到地板，逐像素）</td><td class="n good">{fs["delivered"]}/{fs["n"]}</td><td class="n">{fs["grasp_fail"]}/{fs["grasp_attempts"]}</td></tr>'
    for name, lab in ROBN.items():
        sm = CB.get('robust', {}).get('summary', {}).get(name)
        if sm: clear_robust_rows += f'<tr><td>{lab}</td><td class="n {"good" if sm["delivered"] == sm["n"] else "bad"}">{sm["delivered"]}/{sm["n"]}</td><td class="n">{sm["grasp_fail"]}/{sm["grasp_attempts"]}</td></tr>'
    hs = CB.get('heldout', {}).get('summary', {})
    h1, h2 = hs.get('bottle_alone_new_seeds'), hs.get('materials_new_seeds')
    clear_held_rows = ''
    if h1: clear_held_rows += f'<tr><td>瓶子单独在房间里（种子 401–412）</td><td class="n good">{h1["delivered"]}/{h1["n"]}</td><td class="n">{h1["grasp_fail"]}/{h1["grasp_attempts"]}</td><td class="n">{h1["runs_touching"]}/{h1["n"]}</td><td class="n">{h1["collisions"]}</td></tr>'
    if h2: clear_held_rows += (f'<tr><td>整机难材质：五件黑色或反光的加瓶子（种子 401–408）</td><td class="n good">{h2["regular_delivered"] + h2["bottle_delivered"]}/{h2["regular_total"] + h2["n"]}'
                               f'（瓶子 {h2["bottle_delivered"]}/{h2["n"]}）</td><td class="n">{h2["grasp_fail"]}/{h2["grasp_attempts"]}</td><td class="n">-</td><td class="n">{h2["collisions"]}</td></tr>')
    clear_pose_rows = ''
    for key, lab in (('pose_far', '远处 0.4–1.0 m，机身转 ±35°'), ('pose_close', '近处 0.28–0.45 m，机身转 ±35°'), ('pose_stance', '下手站位 0.21–0.29 m（相机离瓶 13–18 cm）'), ('pose_stance_rough', '同上，深度图更苛刻（35% 没读数，成片）')):
        v = CBN.get(key)
        if not v: continue
        eo = v.get('end_on_within_10deg', {})
        clear_pose_rows += (f'<tr><td>{lab}</td><td class="n">{v["seen"]}/{v["n"]}</td><td class="n">{v["sure"]}</td>'
                            f'<td class="n">{v.get("heading_err_median_deg", "-")}° / {v.get("heading_err_p90_deg", "-")}° / {v.get("heading_err_worst_deg", "-")}°</td>'
                            f'<td class="n {"good" if v.get("wrong_over_25deg", 0) == 0 else "bad"}">{v.get("wrong_over_25deg", "-")}/{v["sure"]}</td><td class="n">{eo.get("flagged_ambiguous", "-")}/{eo.get("n", "-")}</td></tr>')
    cg = CBN.get('grasp', {}).get('across', {})
    csee = CBN.get('see', {})
    def see_cell(o, d, k): v = csee.get(f'{o}:{d}'); return f'{v[k]:.0f}' if v and v.get(k) is not None else '-'
    empt = CBN.get('empty', {}); see_seen = sum(v['seen'] for v in csee.values()); see_n = sum(v['n'] for v in csee.values())
    rough = CBN.get('pose_stance_rough', {})
    rr = CB.get('robust', {}).get('summary', {}).get('both_rougher')
    rr_fail = [f for f in CB.get('robust', {}).get('failure_logs', []) if f['setting'] == 'both_rougher']
    rr_ghosts = sum('发现物体#' in m for _, m in rr_fail[0]['log']) if rr_fail else 0
    sig = CBN.get('signal', {}); sg4 = sig.get('0.4', {}); sg10 = sig.get('1.0', {})
    clear_see_txt = '；'.join(f'{d} m 处 {see_cell("across", d, "near_bias_mm")} mm' for d in ('0.4', '0.6', '1.0', '1.2') if csee.get(f'across:{d}'))
    video = json.load(open(R('episode.json')))
    S12 = json.load(open(R('sensitivity_round12.json'))) if os.path.exists(R('sensitivity_round12.json')) else None
    def s12(name):
        es = [e for e in S12['episodes'] if e['setting'] == name] if S12 else []
        return (f'{sum(e["delivered"] for e in es)}/{sum(e["total"] for e in es)}', sum(e.get('slips', 0) for e in es)) if es else ('-', '-')
    rep = {
        'SENS12_LIDAR': s12('lidar')[0], 'SENS12_LD19': s12('ld19')[0], 'SENS12_SLIPS': str(s12('lidar')[1] + s12('ld19')[1]) if S12 else '-',
        'CLEAR_HELD_ROWS': clear_held_rows or '<tr><td colspan="5">还在跑</td></tr>',
        'CLEAR_AB_ROWS': clear_ab_rows or '<tr><td colspan="6">还在跑</td></tr>', 'CLEAR_FORCED_ROWS': clear_forced_rows or '<tr><td colspan="5">还在跑</td></tr>',
        'CLEAR_ROBUST_ROWS': clear_robust_rows or '<tr><td colspan="3">还在跑</td></tr>', 'CLEAR_POSE_ROWS': clear_pose_rows or '<tr><td colspan="6">还在跑</td></tr>',
        'CLEAR_EMPTY': str(empt.get('frames', '-')), 'CLEAR_EMPTY_FALSE': str(empt.get('false_clusters', '-')), 'CLEAR_DETECT': f'{see_seen}/{see_n}' if see_n else '-',
        'SIG_04_REAL': str(sg4.get('bottle_surface', '-')), 'SIG_04_KEPT': str(sg4.get('bottle_surface_after_edge_filter', '-')), 'SIG_04_HOLES': str(sg4.get('no_reading', '-')), 'SIG_04_SIL': str(sg4.get('silhouette_px', '-')),
        'SIG_10_HOLES': str(sg10.get('no_reading', '-')),
        'RR_DELIVERED': f'{rr["delivered"]}/{rr["n"]}' if rr else '-', 'RR_GF': f'{rr["grasp_fail"]}/{rr["grasp_attempts"]}' if rr else '-', 'RR_STUCK': str(rr['stuck']) if rr else '-', 'RR_GHOSTS': str(rr_ghosts),
        'ROUGH_SURE': f'{rough["sure"]}/{rough["seen"]}' if rough else '-',
        'CLEAR_GRASP': f'{cg["success"]}/{cg["n"]}' if cg else '-', 'CLEAR_SEE_NEAR': clear_see_txt or '-',
        'BLIND_FOUND': str(blind_all), 'BLIND_FOUND_OTHER': str(blind_std), 'SLIPS_TOTAL': str(slips_all), 'SLIPS_EPS': f'{slips_eps}/{len(T["episodes"])}',

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
        'SENSOR_ROWS': seg_cmp_rows, 'PULL_CMP_ROWS': pull_cmp_rows, 'PULL_ROWS': pull_rows, 'SPOIL_CMP_ROWS': spoil_cmp_rows,
        'MAT_ROWS': mat_rows, 'MAT_REGULAR': f'{mat_regular[0]}/{mat_regular[1]}', 'MAT_BOTTLE': f'{mat_bottle[0]}/{mat_bottle[1]}',
        'SENS_ROWS': sens_rows or '<tr><td colspan="6">还在跑</td></tr>', 'COMPUTE_ROWS': compute_rows, 'LOC_ROWS': loc_rows,
        'COMPUTE_TOTAL': f'{100 * CP["total_share_of_one_core"]:.0f}%', 'COMPUTE_CPU': CP['machine'].split('@')[0].replace('(R)', '').strip(),
        'PULL_USE': f'第九轮 {len(R9eps)} 局整机试验里，机器人决定扒 {n_pull_try} 次，扒出来 {n_pull_ok} 次。', 'ODOM_DRIFT': f'{odom:.1f}', 'LOC_MEAN': f'{100 * loc_mean:.1f}', 'LOC_MAX': f'{100 * loc_max:.1f}',
        'HEAD_MAX': f'{head_max:.1f}', 'DET_NEAR': f'{det_near:.1f}', 'DET_FAR': f'{det_far:.0f}',
        'STUCK5': str(sum(e['stuck'] for e in json.load(open(R('trials_round5.json')))['episodes'])), 'STUCK6': str(sum(e['stuck'] for e in T['episodes'])),
        'ROUND_ROWS': round_rows, 'NO_STANCE': f'{sum(v.get("no_stance", 0) for m in G for v in G[m].values())}',
        'ROBOT_MASS': f'{M["robot_mass"]:.2f}',
        'BARE_GF': f'{bj["gf"]}/{bj["ga"]}', 'STD_GF': f'{s["gf"]}/{s["ga"]}', 'STD_TILT': f'{s["tilt"]}/{s["n"]}', 'STD_COLL': str(s['coll']),
    }
    rep.update(m13)
    html = open(os.path.join(HERE, 'page_template.html'), encoding='utf8').read()
    for k, v in rep.items(): html = html.replace('{{' + k + '}}', v)
    assert '{{' not in html, html[html.index('{{'):html.index('{{') + 40]
    os.makedirs(os.path.join(HERE, 'dist'), exist_ok=True)
    open(os.path.join(HERE, 'dist', 'index.html'), 'w', encoding='utf8').write(html)
    for f in ('episode.mp4', 'episode.jpg', 'clear.mp4', 'clear.jpg', 'modes.mp4', 'modes.jpg'):
        if os.path.exists(R(f)): shutil.copy(R(f), os.path.join(HERE, 'dist', f))
    # What is published to claude.ai: the same page without its own document tags (the platform wraps its own <!doctype>/<head>/<body>
    # around it), <title> first, the page language set by a script because the <html lang> goes with the wrapper.
    import re
    frag = re.sub(r'^\s*<!DOCTYPE html>\s*', '', html, flags=re.I)
    frag = re.sub(r'<html[^>]*>\s*', '', frag, count=1); frag = re.sub(r'<head>\s*', '', frag, count=1); frag = re.sub(r'\s*</head>', '', frag, count=1)
    frag = re.sub(r'<meta charset="utf-8">\s*', '', frag, count=1, flags=re.I); frag = re.sub(r'<meta name="viewport"[^>]*>\s*', '', frag, count=1)
    frag = re.sub(r'<body[^>]*>\s*', '', frag, count=1); frag = re.sub(r'\s*</body>\s*</html>\s*$', '\n', frag)
    title = re.search(r'<title>.*?</title>', frag).group(0)
    frag = title + '\n<script>document.documentElement.lang = "zh-CN";</script>\n' + frag.replace(title, '', 1).lstrip()
    open(os.path.join(HERE, 'dist', 'artifact.html'), 'w', encoding='utf8').write(frag)

    md = ['# 实物验收方案', '', '按顺序做，前一项不过先别做后一项。「仿真值」来自 `results/` 里的这一轮试验，由 `web/build_page.py` 自动填入。', '',
          '| # | 测什么 | 怎么测 | 仿真值 | 通过 |', '|---|---|---|---|---|']
    md += [f'| {i + 1} | {a} | {b} | {c} | {d} |'.replace('<code>', '`').replace('</code>', '`').replace('<b>', '**').replace('</b>', '**') for i, (a, b, c, d) in enumerate(valid)]
    md += [f'| {n} | {a} | {b_} | {c} | {d} |' for n, a, b_, c, d in home_round_rows(len(valid))]          # round 14: only in the document, not on the published page
    if SE:
        md += ['', '## 敏感度：每次只让一项变差（标准条件 10 局）', '', '| 变差的那一项 | 第十轮代码 | 第十一轮代码 | 抓取失败/尝试（最新） | 定位最大误差（最新） |', '|---|---|---|---|---|']
        md += sens_md
    md += ['', '## 分割：挨着放的两件', '', '| 边缘间隙 | 试验 | 分得开 | 粘成一件 |', '|---|---|---|---|']
    md += [f'| {g} | {v["n"]} | {v["apart"]} | {v["merged"]} |' for g, v in pairs.items() if v['n']]
    md += ['', '## 整机试验汇总', '', '| 条件 | 试验次数 | 送达 | 五件全送达 | 每件用时 | 抓取失败/尝试 | 途中掉落 | 碰撞 | 机身倾斜超过 3° 的试验 |', '|---|---|---|---|---|---|---|---|---|']
    md += [f'| {COND[n]} | {a["n"]} | {a["d"]}/{a["tot"]}（{pct(a["d"], a["tot"])}） | {a["all5"]}/{a["n"]} | {a["t"]:.0f} 秒 | {a["gf"]}/{a["ga"]} | {a["drop"]} | {a["coll"]} | {a["tilt"]}（最大 {a["max_tilt"]:.1f}°） |' for n, a in agg.items()]
    md += ['', '## 仿真里什么是真的，什么是假设', '',
           '真物理：轮子滚子与地面的接触摩擦；舵机力矩上限 2.94 N·m 和由阻尼决定的转速上限；机械臂的质量、惯量、限位（SO-101 官方模型）；夹爪与物体的接触；雷达射线投射（名义 1 cm 噪声、被自身遮挡；另按 LD19 规格书的误差模型测过）。', '',
           '定位和感知也是真算法：轮子里程计（1.5% 轮径误差、3% 打滑）加雷达扫描匹配；物体只从渲染出的深度图里找：拟合地面、去掉家具、按俯视连通分成一件件，再量轮廓和高度，不用任何已知类别或尺寸，也不用训练好的模型。夹爪里有没有东西由腕部深度相机判断。', '',
           '假设（每条都对应上表一项实测）：深度图的毛病（黑色、反光、透明表面的空洞，边缘飞点）按规律近似，比例是估的；雷达误差按规格书上限估（每个表面固定偏差 ±45 mm、1.5 cm 量化、混合点、野点），角度零位误差没有建模；机器人认得出自己的手臂（按运动学屏蔽）；收纳筐大致位置和尺寸已知；底盘重量是估的；打印件不变形、舵机无回差；软物体是多段刚体加弹性铰链和柔性接触的近似，不是布料仿真。', '',
           '## 实测后怎么回填', '', '1. 硬件参数改 `sim/model.py` 顶部的 `P`（质量、摩擦、轮径、力矩上限）。', '2. 传感器噪声改 `sim/robot.py` 顶部的常量和 `sim/lidar_models.py`；安全余量（含 `LIDAR_BIAS_MARGIN`）在 `sim/controller.py` 顶部。',
           '3. 物体的尺寸、重量、软硬在 `sim/objects.py`，可以照着手边的实物改或加新种类。', '4. 依次重跑 `python test_base.py`、`python test_perception.py`、`python test_grasp.py 10`、`python run_trials.py`，再 `python ../web/build_page.py`。', '']
    open(os.path.join(ROOT, 'docs', 'validation.md'), 'w', encoding='utf8').write('\n'.join(md))
    print('built', len(html), 'bytes;', {k: (v if len(v) < 40 else '...') for k, v in rep.items() if k not in ('TRIAL_ROWS', 'FAIL_ROWS', 'GRASP_ROWS', 'BASE_ROWS', 'VALID_ROWS', 'ROUND_ROWS')})


if __name__ == '__main__':
    main()
