"""Write web/playground/data/model.json: the numbers, rules and word lists of the real code, plus what the real code answers to a long list of test cases, for the web playground (web/playground/).

    python export_playground.py

The playground (JavaScript, runs in any browser, nothing to install) is a *behaviour model*: it re-implements the decision rules of the real system (the mode arbitration of modes.py, the safety
layer Guard of people.py, the permissions and the question-and-confirm dialogue of voice.py, the battery of power.py) over a simplified 2-D world. It does not run the physics, the grasping
or the real perception. So that it can be checked rather than believed, this file also records "golden" answers of the real code (a grid of safety-layer situations, a list of sentences said by
four kinds of speaker, some multi-turn dialogues, the charging curve); the playground runs the same cases through its own rules and shows where the two agree and where they do not.
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, 'web', 'playground', 'data', 'model.json')


def constants():
    import controller as C
    import modes
    import people as P
    import power as PW
    import voice as V
    from model import BASKET, DOCK_POSE, FURNITURE, HOME, ROOM_H, ROOM_W
    return dict(
        limits=dict(v_max=C.V_MAX, w_max=C.W_MAX, robot_r=P.Guard.CFG['robot_r'], person_r=P.Guard.CFG['person_r']),
        guard=dict(P.Guard.CFG),
        classify=dict(child_below=P.CHILD_BELOW, adult_from=P.ADULT_FROM, adult_width=P.ADULT_WIDTH, group_foot=P.GROUP_FOOT, max_foot=P.MAX_FOOT),
        modes=dict(modes.DEFAULTS),
        battery=dict(capacity_wh=PW.CAPACITY_WH, p_run_w=PW.P_RUN_W, p_docked_w=PW.P_DOCKED_W, charger_w=PW.CHARGER_W, cc_until=PW.CC_UNTIL, cv_taper=PW.CV_TAPER, v_empty=PW.V_EMPTY, v_full=PW.V_FULL,
                     r_int=PW.R_INT, soc_cutoff=PW.SOC_CUTOFF, est_bias_sigma=0.03, servo_idle_w=3.15),
        world=dict(room=[ROOM_W, ROOM_H], furniture=[list(f) for f in FURNITURE], basket=dict(BASKET), home=list(HOME), dock=list(DOCK_POSE), doors=[list(d) for d in P.DOORS]),
        people=[dict(name=n, cn=s['cn'], cls=s['cls'], h=s['h'], w=s['w'], hip=s['hip'], clothes=list(s['clothes'])) for n, s in P.SPECS.items()],
        voice=dict(wake=V.WAKE, wake_alt=V.WAKE_ALT, session_s=V.SESSION_S, pending_s=V.PENDING_S, items={k: list(v) for k, v in V.ITEM_WORDS.items()}, places={k: list(v) for k, v in V.PLACE_WORDS.items()},
                   trig={k: [list(p) for p in v] for k, v in V.TRIG.items()}, yes=list(V.YES), no=list(V.NO), needs=V.NEEDS, motion=sorted(V.MOTION), sensitive=sorted(V.SENSITIVE),
                   stop_words=list(V.NLU.STOP_WORDS), role_rank=V.ROLE_RANK, min_role=V.MIN_ROLE, mode_cn=V.MODE_CN, mode_en=V.MODE_EN, asr_conf=0.9),
    )


def pinyin_table():
    """Toneless pinyin of every character of the CJK block that pypinyin knows, as {syllable: "chars"}: the real parser matches on (reduced) pinyin, so a typed homophone is no mistake;
    the page needs the same table to do the same."""
    from pypinyin import Style, pinyin
    by = {}
    for c in range(0x4E00, 0xA000):
        ch = chr(c); r = pinyin(ch, style=Style.NORMAL, errors='ignore')
        if r and r[0] and r[0][0]: by.setdefault(r[0][0], []).append(ch)
    return {k: ''.join(v) for k, v in by.items()}


def layouts():
    from episode import make_layout
    out = {}
    for seed in range(1, 13):
        objs, boxes = make_layout(seed)
        out[str(seed)] = dict(objects=[dict(kind=o[0], x=o[1], y=o[2], th=o[3], r=round(float(o[4]['radius']), 3)) for o in objs], boxes=[list(b) for b in boxes])
    return out


def grasp_rates():
    try:
        g = json.load(open(os.path.join(ROOT, 'results', 'grasp_bench.json')))['1.0']
        return {k: dict(success=v['success'], n=v['n']) for k, v in g.items()}
    except Exception:
        return {}


def accept_summary():
    try:
        r = json.load(open(os.path.join(ROOT, 'results', 'acceptance_sim.json')))
        import accept
        titles = {it['id'] if isinstance(it, dict) else it[0]: (it['title'] if isinstance(it, dict) else it[1]) for it in accept.ITEMS}
        return dict(when=r.get('when'), code=r.get('code'), items={k: dict(title=titles.get(k, ''), verdict=v.get('verdict'), accepted=accept.ACCEPTED.get(k), lines=[dict(label=l.get('label'), measured=l.get('measured'), need=l.get('need'), ok=l.get('ok')) for l in v.get('lines', [])][:6]) for k, v in r['items'].items()})
    except Exception as e:
        print('accept summary skipped:', type(e).__name__, e)
        return {}


# ---------------------------------------------------------------------------------------------------------------- golden: the voice
UTTS = [
    '', '停下', '停下来', '别动', '暂停', '等一下', '先别动', '继续', '接着干', '算了', '取消', '不用了',
    '收拾一下玩具', '收拾玩具', '整理房间', '捡玩具', '把玩具收一收', '把地上的东西收起来', '打扫一下',
    '把小球放进筐里', '把积木放到筐里', '把玩具车收好', '帮我把毛绒玩具放进筐里', '放进筐里', '把它放进筐里',
    '小球在哪', '我的玩具车在哪里', '帮我找一下袜子', '地上有什么东西', '有哪些玩具', '清点一下', '多了什么', '有什么变化',
    '把家里扫描一遍', '建个地图', '看看家里', '巡视一下', '扫描',
    '去沙发那边', '去桌子那边', '到筐那边', '去充电座', '走到架子那边', '过去沙发', '去阳台', '去那边',
    '去充电', '回去充电', '充电', '回充', '吃点电',
    '现在有谁', '有人吗', '孩子在哪', '谁在家',
    '记住我，我叫爸爸', '记住这个孩子叫小明', '忘掉小明', '忘掉所有人',
    '你在干什么', '还有多少电', '在忙什么',
    '你在录音吗', '静音', '你在听吗', '关掉麦克风',
    '你好', '早上好', '在吗', '你是谁', '你叫什么', '你会什么', '你能做什么', '过来', '到我身边来', '说英文', '用中文',
    '今天天气不错', '明天要下雨', '哈哈哈', '好', '对', '不是', '嗯',
    '巡视一下', '去巡视一下家里', '吃一下', '试一下', '这是谁的', '那个球是我的', '记一下我女儿', '认识一下我女儿', '别听了',
]
ROLES = ['owner', 'adult', 'child', 'unknown']
DIALOGUES = [
    dict(name='去沙发那边 → 对', turns=[('owner', '小拾小拾，去沙发那边', 0), ('owner', '对', 3)]),
    dict(name='去沙发那边 → 不是', turns=[('owner', '小拾小拾，去沙发那边', 0), ('owner', '不是', 3)]),
    dict(name='去沙发那边，隔了 20 秒才说对', turns=[('owner', '小拾小拾，去沙发那边', 0), ('owner', '对', 20)]),
    dict(name='孩子说去沙发那边', turns=[('child', '小拾小拾，去沙发那边', 0), ('child', '对', 3)]),
    dict(name='放进筐里 → 小球', turns=[('owner', '小拾小拾，放进筐里', 0), ('owner', '小球', 3)]),
    dict(name='收拾玩具之后不带唤醒词说停下', turns=[('owner', '小拾小拾，收拾一下玩具', 0), ('owner', '停下', 4)]),
    dict(name='收拾玩具之后 20 秒再说继续（没喊它）', turns=[('owner', '小拾小拾，收拾一下玩具', 0), ('owner', '继续', 20)]),
    dict(name='喊它之后不说话 → 我在', turns=[('owner', '小拾小拾', 0), ('owner', '去充电', 3)]),
    dict(name='没喊它的闲聊', turns=[('owner', '去充电', 0), ('owner', '收拾一下玩具', 2)]),
    dict(name='停止词不用唤醒', turns=[('child', '停下', 0), ('unknown', '别动', 2)]),
]


def voice_golden():
    import voice as V
    out = []
    for role in ROLES:
        for u in UTTS:
            vs = V.VoiceSystem(asr=V.TextASR(), seed=1)
            text = '小拾小拾，' + u if u else '小拾小拾'
            rec = vs.hear(text, role=role)
            reply = rec.get('reply') or (' / '.join(x[1] for x in vs.tts) if vs.tts else None)
            out.append(dict(role=role, text=text, intent=rec.get('intent'), action=rec.get('action'), conf=rec.get('conf'), reply=reply, slots=rec.get('slots')))
    dia = []
    for d in DIALOGUES:
        vs = V.VoiceSystem(asr=V.TextASR(), seed=1); turns = []
        for role, text, wait in d['turns']:
            vs.advance(wait); rec = vs.hear(text, role=role)
            turns.append(dict(role=role, text=text, wait=wait, intent=rec.get('intent'), action=rec.get('action'), conf=rec.get('conf'), reply=rec.get('reply')))
        dia.append(dict(name=d['name'], turns=turns))
    return dict(single=out, dialogues=dia)


# ---------------------------------------------------------------------------------------------------------------- golden: the safety layer
class _Cam:
    def __init__(s, t, x, y, vx, vy, cls):
        s.x, s.y, s.vx, s.vy, s.cls, s.coast = x, y, vx, vy, cls, False
    def predict(s, t): return s.x, s.y


def guard_golden():
    """Put a person in front of the real Guard in a grid of situations and write down what it does."""
    import people as P
    import robot as R

    class Det:
        def __init__(s): s.k = []
        def confirmed(s, n): return s.k
    class Rb:
        def __init__(s): s.t = 100.0; s.hold_arm = False; s.wheel_target = None
        def arm_done(s, tol): return True
        def set_twist(s, *a): s.wheel_target = R.twist_to_wheels(0, 0, 0)
    class Ag:
        def __init__(s): s.state = '前往'; s.arm_busy = False; s.zone_skip_t = -1e9; s.person_zones = []; s.dyn_zones = []
        def pose(s): return (2.0, 1.5, 0.0)
    class Sup:
        def __init__(s): s.mode = 'TASK'; s.r = Rb(); s.a = Ag(); s.events = []
        def _event(s, t): s.events.append(t)

    cfg = P.Guard.CFG; rr = cfg['robot_r'] + cfg['person_r']
    out = []
    for cls in ('adult', 'child', 'unsure'):
        for bearing in (0, 45, 90, 180):
            for gap in (0.05, 0.15, 0.25, 0.30, 0.40, 0.50, 0.60, 0.80, 1.0, 1.2, 1.5, 2.0):
                for closing in (0.0, 0.4, 0.8, 1.4):
                    for v in (0.06, 0.18):
                        for arm in (False, True):
                            sup = Sup(); det = Det(); g = P.Guard(det); d = rr + gap; b = math.radians(bearing)
                            px, py = 2.0 + d * math.cos(b), 1.5 + d * math.sin(b)
                            vx, vy = -closing * math.cos(b), -closing * math.sin(b)
                            det.k = [_Cam(0, px, py, vx, vy, cls)]
                            sup.a.arm_busy = arm
                            halted = bool(g.pre(sup))
                            sup.r.wheel_target = R.twist_to_wheels(v, 0.0, 0.0)
                            g.post(sup)
                            out.append(dict(cls=cls, bearing=bearing, gap=gap, closing=closing, v=v, arm=arm, halted=halted, scale=round(float(g.scale), 6)))
    return out


def battery_golden():
    import power as PW
    out = []
    for s0 in (0.05, 0.10, 0.20, 0.50, 0.80, 0.90):
        b = PW.Battery(soc=s0, est_bias=0.0); secs = b.charge_until(0.90)
        out.append(dict(from_soc=s0, to_soc=0.90, minutes=round(secs / 60, 2), final=round(b.soc, 4)))
    volt = [dict(soc=s / 20, volts_run=round(PW.Battery(soc=s / 20, est_bias=0.0).voltage(PW.P_RUN_W), 3)) for s in range(0, 21)]
    return dict(charge=out, voltage=volt)


def main():
    model = constants()
    model['layouts'] = layouts(); model['grasp'] = grasp_rates(); model['accept'] = accept_summary(); model['pinyin'] = pinyin_table()
    model['golden'] = dict(voice=voice_golden(), guard=guard_golden(), battery=battery_golden())
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as f: json.dump(model, f, ensure_ascii=False, separators=(',', ':'))
    g = model['golden']
    print(f'wrote {OUT} ({os.path.getsize(OUT) / 1024:.0f} KB): {len(model["layouts"])} layouts, {len(g["voice"]["single"])} sentences, {len(g["voice"]["dialogues"])} dialogues, {len(g["guard"])} guard situations, {len(model["accept"].get("items", {}))} acceptance items')


if __name__ == '__main__':
    main()
