# 가상 사용자 N명 × 하루치 생활 데이터 (실험실 과제가 아니라 "하루")
# 배울 개념: 과제 프로토콜(synth.py)은 30초짜리 정답 구간이라 쉽다. 실제 오알림은 하루 중
# 긴 정적 구간, 짧은 들기, 센서 밀림에서 나온다. 여기서 그 세 가지를 일부러 넣는다.
import numpy as np

from ml.config import FS
from ml.domain.types import Session
from ml.sim.synth import (TASKS, random_body, _smooth_noise, _ramp, _base, _task_targets,
                          tilt_to_imu, to_raw_df)
import pandas as pd

# 하루 블록: (task_id, 최소 초, 최대 초). task_id는 TASK_MAP과 같아 evaluator를 그대로 쓴다.
BLOCKS = {
    "sit_good":  ("T01", 120, 900),
    "sit_slump": ("T02", 8, 240),     # 정답 이벤트: 5초 이상이면 알려야 함
    "sit_hinge": ("T04", 20, 180),    # 하드 네거티브
    "stand":     ("T07", 30, 300),
    "hinge":     ("T06", 3, 10),      # 하드 네거티브 (물건 집기 등 권장 동작)
    "flex_lift": ("T05", 3, 12),      # 정답 이벤트
    "walk":      ("T08", 20, 240),
    "sitstand":  ("T09", 9, 30),
}


def random_user(rng):
    """사람마다 습관이 다르다. slump_rate = 앉아 있는 동안 구부정해지는 빈도."""
    return dict(body=random_body(rng),
                slump_rate=rng.uniform(0.15, 0.6),
                lift_bad=rng.uniform(0.2, 0.7),      # 들 때 허리를 마는 비율
                slide_deg=rng.uniform(2, 6),         # 하루 중 센서가 밀리는 크기(°)
                fatigue=rng.uniform(0.0, 0.5))       # 시간이 갈수록 구부정 증가


def _next_block(prev, t_frac, user, rng):
    w = {"sit_good": 3, "sit_slump": 0.5 + 6 * user["slump_rate"] * (1 + user["fatigue"] * t_frac),
         "sit_hinge": 0.7, "stand": 1.2, "hinge": 0.8, "flex_lift": 0.8 * user["lift_bad"] + 0.05,
         "walk": 1.5, "sitstand": 1.0}
    w["hinge"] *= (1 - user["lift_bad"]) + 0.3
    w.pop(prev, None)                      # 같은 블록 연속 방지
    keys = list(w)
    p = np.array([w[k] for k in keys]); p /= p.sum()
    return str(rng.choice(keys, p=p))


def make_user_day(sid, day, user, rng, wear_min=60):
    """보정 4단계로 시작(실제 사용도 매일 아침 보정한다고 가정) + wear_min분의 하루.
    센서는 하루 중 한 번 밀린다(slide). 보정은 밀리기 전 상태 기준이다."""
    body = user["body"]
    pel, lum, lin, rows = [], [], [], []
    st = {"p": 0.0, "l": 0.0, "t": 0}

    def add(tid, p, l, la):
        n = len(p)
        pel.append(p); lum.append(l); lin.append(la)
        rows.append((st["t"] * 1_000_000 // FS, (st["t"] + n) * 1_000_000 // FS, tid))
        st["t"] += n; st["p"], st["l"] = p[-1], l[-1]

    def hold(tid, tp, tl, n, sway=0.5):
        add(tid, _ramp(st["p"], tp, n) + _smooth_noise(n, sway, rng),
            _ramp(st["l"], tl, n) + _smooth_noise(n, sway, rng), np.zeros((n, 3)))

    hold("C_STAND", 0.0, 0.0, 30 * FS)
    n = 16 * FS
    tri = 40 * np.interp(np.arange(n), [0, 4 * FS, 8 * FS, 12 * FS, 16 * FS], [0, 1, 0, 1, 0])
    add("C_HINGE", st["p"] + tri + _smooth_noise(n, 0.5, rng),
        st["l"] + body.hinge_leak * tri + _smooth_noise(n, 0.5, rng), np.zeros((n, 3)))
    sp, sl = _base(body, "sit")
    hold("C_SIT", sp, sl, 20 * FS)
    hold("C_SLUMP", sp - 8, sl + rng.uniform(15, 28) * body.flex_scale, 10 * FS)
    cal_end = st["t"]

    total = wear_min * 60 * FS
    prev = "sit_good"
    while st["t"] - cal_end < total:
        name = _next_block(prev, (st["t"] - cal_end) / total, user, rng)
        tid, lo, hi = BLOCKS[name]
        n = int(rng.uniform(lo, hi) * FS)
        hold("REST", *_base(body, TASKS[tid][0]), 2 * FS)
        kind = TASKS[tid][3]
        if kind == "static":
            hold(tid, *_task_targets(body, tid, rng), n)
        elif kind == "moving":
            ph = 2 * np.pi * 2.0 * np.arange(n) / FS
            la = np.zeros((n, 3)); la[:, 0] = 0.15 * body.wobble * np.sin(2 * ph); la[:, 2] = 0.05 * body.wobble * np.sin(ph)
            add(tid, _ramp(st["p"], 0.0, n) + 4 * body.wobble * np.sin(ph),
                _ramp(st["l"], 0.0, n) + 2 * body.wobble * np.sin(ph + 0.5),
                la + rng.normal(0, 0.02 * body.wobble, (n, 3)))
        else:
            p, l = np.empty(n), np.empty(n)
            cp, cl, sitting = st["p"], st["l"], True
            for s in range(0, n, 3 * FS):
                e = min(s + 3 * FS, n)
                tp, tl = _base(body, "sit" if sitting else "stand")
                p[s:e] = _ramp(cp, tp, e - s); l[s:e] = _ramp(cl, tl, e - s)
                cp, cl, sitting = p[e - 1], l[e - 1], not sitting
            k = np.arange(n) % (3 * FS)
            p += np.where(k < FS, 25 * np.sin(np.pi * k / FS), 0)
            add(tid, p, l, rng.normal(0, 0.05 * body.wobble, (n, 3)))
        prev = name

    pelvis, lumbar, lin_acc = np.concatenate(pel), np.concatenate(lum), np.concatenate(lin)
    N = len(pelvis)
    # 부착 오차: 보정 구간까지는 고정, 감시 구간 중간에 한 번 밀린다 (상부 센서가 slide_deg만큼)
    slide_at = cal_end + int(rng.uniform(0.3, 0.7) * (N - cal_end))
    mu, ml_ = rng.normal(0, 5, 2)
    shift = np.where(np.arange(N) >= slide_at, user["slide_deg"] * rng.choice([-1, 1]), 0.0)
    drift = np.linspace(0, rng.normal(0, 1.5), N)    # 자이로 바이어스 느린 변화(°/s 아님, 각도 드리프트로 단순화)
    lower = pelvis + ml_
    upper = pelvis + body.gain * lumbar + body.crosstalk * pelvis + mu + shift + drift
    soft = _smooth_noise(N, 0.3 * body.wobble, rng, width=5)
    l_acc, l_gyr = tilt_to_imu(lower, lin_acc, rng, bias_dps=rng.normal(0, 1))
    u_acc, u_gyr = tilt_to_imu(upper + soft, lin_acc * 1.2, rng, bias_dps=rng.normal(0, 1))
    tasks = pd.DataFrame(rows, columns=["start_us", "end_us", "task_id"])
    return Session(sid, f"day{day:02d}", to_raw_df(u_acc, u_gyr, l_acc, l_gyr), tasks, body), cal_end


def make_cohort(n_users=10, n_days=5, seed=0, wear_min=60, slide=None):
    """U01..Un. 같은 (n, seed)면 항상 같은 사람·같은 하루. 반환: {sid: (user, [(Session, cal_end), ...])}"""
    out = {}
    for i in range(1, n_users + 1):
        rng = np.random.default_rng(seed * 1000 + i)
        user = random_user(rng)
        if slide is not None:
            user["slide_deg"] = slide          # 스트레스 실험: 센서가 크게 밀리는 사람들
        out[f"U{i:02d}"] = (user, [make_user_day(f"U{i:02d}", d + 1, user, rng, wear_min) for d in range(n_days)])
    return out


# ======================================================================
# 변수·함수 설명
# ======================================================================
#
# [BLOCKS]  하루를 이루는 행동. task_id를 TASK_MAP과 같게 해서 evaluator.replay/score를 그대로 쓴다.
#   정답 이벤트(알려야 함): sit_slump(T02), flex_lift(T05). 하드 네거티브: sit_hinge(T04), hinge(T06).
# [random_user]  slump_rate(구부정 빈도), lift_bad(들 때 허리를 마는 비율), slide_deg(센서 밀림),
#   fatigue(후반부 구부정 증가). 모두 가정값 — 본인 실측으로 맞출 것.
# [make_user_day]  하루 시작 보정 → wear_min분 생활. 센서 밀림은 감시 구간 중 한 번.
# [한계]  블록 전환과 구간 길이는 마르코프 가정이지 실제 분포가 아니다. 이 시뮬은 "파이프라인이
#   하루 단위 입력·밀림에서 안 깨지는가"를 보는 용도이며, 정확도 주장에는 쓸 수 없다.
