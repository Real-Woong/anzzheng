# 개인차가 있는 가짜 참가자 생성 (domain randomization)
# 배울 개념: 참가자 6명으로는 "체형 → 오차" 관계를 학습할 수 없다. 대신 시뮬레이터에서
# 신체 파라미터를 넓게 흔들어, 모델과 보정이 그 변화에 강건한지 미리 확인한다.
import numpy as np
import pandas as pd

from ml.config import FS
from ml.domain.types import Body, Session
from ml.signal.process import ACC_LSB, GYR_LSB

# 과제ID: (기본 자세, 골반 변화°, 요추 변화°, 종류)
# 앉기 과제는 Body.sit_pelvis / sit_loss 위에 더한다. 요추 변화 > 0은 flex_scale로 사람마다 키운다.
TASKS = {
    "T01": ("sit", 0, 0, "static"),       # 바르게 앉기
    "T02": ("sit", -8, 22, "static"),     # 구부정하게 앉기
    "T03": ("sit", 12, 28, "static"),     # 말아서 책상 쪽으로 숙여 앉기
    "T04": ("sit", 35, -2, "static"),     # 고관절만 접어 기울여 앉기 ← 하드 네거티브
    "T05": ("stand", 15, 35, "static"),   # 서서 허리 말아 숙이기
    "T06": ("stand", 45, 3, "static"),    # 힙 힌지 ← 하드 네거티브
    "T07": ("stand", 0, 0, "static"),     # 바르게 서기
    "T08": ("stand", 0, 0, "moving"),     # 걷기
    "T09": ("stand", 0, 0, "transition"), # 앉기 ↔ 서기
}
ROUND2 = ["T01", "T02", "T04", "T06", "T08"]


def random_body(rng):
    return Body(
        gain=rng.uniform(0.7, 1.1),
        crosstalk=rng.uniform(0.0, 0.15),
        sit_pelvis=rng.uniform(-20, -8),
        sit_loss=rng.uniform(10, 25),
        flex_scale=rng.uniform(0.75, 1.25),
        wobble=rng.uniform(0.5, 2.0),
        hinge_leak=rng.uniform(0.0, 0.05),
    )


def _smooth_noise(n, sd, rng, width=25):
    """사람의 미세한 흔들림. 샘플마다 독립인 노이즈는 자이로를 비현실적으로 키운다."""
    x = np.convolve(rng.normal(0, 1, n + width), np.ones(width) / width, "same")[:n]
    return x * (sd / (x.std() + 1e-9))


def _ramp(prev, target, n, ramp_s=1.0):
    w = np.clip(np.arange(n) / (ramp_s * FS), 0, 1)
    return prev + (target - prev) * w


def _base(body, posture):
    return (body.sit_pelvis, body.sit_loss) if posture == "sit" else (0.0, 0.0)


def _task_targets(body, tid, rng):
    posture, dp, dl, _ = TASKS[tid]
    bp, bl = _base(body, posture)
    jitter = rng.uniform(0.85, 1.15)
    dl = dl * body.flex_scale * jitter if dl > 0 else dl
    return bp + dp * jitter, bl + dl


def tilt_to_imu(theta_deg, lin_acc, rng, bias_dps):
    """기울기(°) 시계열 + 움직임 가속도(g) → 가속도(g), 자이로(°/s)
    센서 축: x = 척추 위쪽, z = 등 바깥쪽. 앞으로 숙이면 theta 증가."""
    th = np.deg2rad(theta_deg)
    acc = np.stack([np.cos(th), np.zeros_like(th), np.sin(th)], 1) + lin_acc
    gyr = np.zeros((len(th), 3))
    gyr[:, 1] = np.gradient(theta_deg) * FS + bias_dps
    acc += rng.normal(0, 0.01, acc.shape)
    gyr += rng.normal(0, 0.3, gyr.shape)
    return acc, gyr


def make_session(body, task_order, rng, dur_s=30):
    """보정 4단계 + 과제들. 반환: (원시 df, 과제 구간 df)"""
    pel, lum, lin, rows = [], [], [], []
    state = {"p": 0.0, "l": 0.0, "t": 0}

    def add(tid, p, l, lin_acc):
        n = len(p)
        pel.append(p); lum.append(l); lin.append(lin_acc)
        rows.append((state["t"] * 1_000_000 // FS, (state["t"] + n) * 1_000_000 // FS, tid))
        state["t"] += n
        state["p"], state["l"] = p[-1], l[-1]

    def hold(tid, tp, tl, n, sway=0.5):
        p = _ramp(state["p"], tp, n) + _smooth_noise(n, sway, rng)
        l = _ramp(state["l"], tl, n) + _smooth_noise(n, sway, rng)
        add(tid, p, l, np.zeros((n, 3)))

    # 보정: 서기 30초 → 힙 힌지 2회 → 바르게 앉기 20초 → 편하게 구부정 앉기 10초
    hold("C_STAND", 0.0, 0.0, 30 * FS)
    n = 16 * FS
    tri = 40 * np.interp(np.arange(n), [0, 4 * FS, 8 * FS, 12 * FS, 16 * FS], [0, 1, 0, 1, 0])
    add("C_HINGE", state["p"] + tri + _smooth_noise(n, 0.5, rng),
        state["l"] + body.hinge_leak * tri + _smooth_noise(n, 0.5, rng), np.zeros((n, 3)))
    sp, sl = _base(body, "sit")
    hold("C_SIT", sp, sl, 20 * FS)
    hold("C_SLUMP", sp - 8, sl + rng.uniform(15, 28) * body.flex_scale, 10 * FS)

    for tid in task_order:
        hold("REST", *_base(body, TASKS[tid][0]), 5 * FS)   # 과제 사이 5초 휴식 (ML 문서 4.4)
        n = dur_s * FS
        kind = TASKS[tid][3]
        if kind == "static":
            hold(tid, *_task_targets(body, tid, rng), n)
        elif kind == "moving":
            ph = 2 * np.pi * 2.0 * np.arange(n) / FS
            p = _ramp(state["p"], 0.0, n) + 4 * body.wobble * np.sin(ph)
            l = _ramp(state["l"], 0.0, n) + 2 * body.wobble * np.sin(ph + 0.5)
            la = np.zeros((n, 3))
            la[:, 0] = 0.15 * body.wobble * np.sin(2 * ph)
            la[:, 2] = 0.05 * body.wobble * np.sin(ph)
            add(tid, p, l, la + rng.normal(0, 0.02 * body.wobble, (n, 3)))
        else:  # transition: 3초마다 앉기 ↔ 서기
            p, l = np.empty(n), np.empty(n)
            cur_p, cur_l, sitting = state["p"], state["l"], True
            for s in range(0, n, 3 * FS):
                e = min(s + 3 * FS, n)
                tp, tl = _base(body, "sit" if sitting else "stand")
                p[s:e] = _ramp(cur_p, tp, e - s)
                l[s:e] = _ramp(cur_l, tl, e - s)
                cur_p, cur_l, sitting = p[e - 1], l[e - 1], not sitting
            k = np.arange(n) % (3 * FS)
            p += np.where(k < FS, 25 * np.sin(np.pi * k / FS), 0)  # 전환 1초 동안 상체 앞쏠림
            add(tid, p, l, rng.normal(0, 0.05 * body.wobble, (n, 3)))

    pelvis, lumbar, lin_acc = np.concatenate(pel), np.concatenate(lum), np.concatenate(lin)

    mount_u, mount_l = rng.normal(0, 5, 2)   # 재부착마다 달라진다
    lower = pelvis + mount_l
    upper = pelvis + body.gain * lumbar + body.crosstalk * pelvis + mount_u
    soft = _smooth_noise(len(pelvis), 0.3 * body.wobble, rng, width=5)
    l_acc, l_gyr = tilt_to_imu(lower, lin_acc, rng, bias_dps=rng.normal(0, 1))
    u_acc, u_gyr = tilt_to_imu(upper + soft, lin_acc * 1.2, rng, bias_dps=rng.normal(0, 1))

    return to_raw_df(u_acc, u_gyr, l_acc, l_gyr), pd.DataFrame(rows, columns=["start_us", "end_us", "task_id"])


def to_raw_df(u_acc, u_gyr, l_acc, l_gyr):
    """g, °/s → ESP32가 보내는 것과 같은 정수(LSB) CSV 모양. MuJoCo 시뮬레이터도 이걸 쓴다."""
    N = len(u_acc)
    df = pd.DataFrame({"seq": np.arange(N), "t_us": np.arange(N) * 1_000_000 // FS})
    for pre, acc, gyr in (("u", u_acc, u_gyr), ("l", l_acc, l_gyr)):
        for i, ax in enumerate("xyz"):
            df[f"{pre}_a{ax}"] = np.round(acc[:, i] * ACC_LSB).astype(int)
        for i, ax in enumerate("xyz"):
            df[f"{pre}_g{ax}"] = np.round(gyr[:, i] * GYR_LSB).astype(int)
    return df


def make_subject(sid, seed):
    """참가자 1명 = 1회차(전체 과제) + 재부착 후 2회차(단축). ML 문서 4.4 프로토콜과 같다."""
    rng = np.random.default_rng(seed)
    body = random_body(rng)
    out = []
    for name, order, dur in (("sess01", list(TASKS), 30), ("sess02", ROUND2, 20)):
        order = [str(t) for t in rng.permutation(order)]
        df, tasks = make_session(body, order, rng, dur_s=dur)
        out.append(Session(sid, name, df, tasks, body))
    return out


# ======================================================================
# 변수·함수 설명
# ======================================================================
#
# [Body 파라미터 범위]  (각 의미는 ml/domain/types.py)
#   gain 0.7–1.1, crosstalk 0–0.15, sit_pelvis −20~−8°, sit_loss 10–25°,
#   flex_scale 0.75–1.25, wobble 0.5–2.0, hinge_leak 0–5%
#   임상 감각에 맞춘 가정값이다. 실측 데이터가 생기면 여기를 맞춘다.
#
# [센서에 찍히는 각도]
#   하부(S1)     = 골반 + mount_l
#   상부(T12/L1) = 골반 + gain×요추 + crosstalk×골반 + mount_u + 연부조직 흔들림
#   mount_u/l    부착 각도. 재부착(2회차)마다 새로 뽑힌다.
#
# [make_session]
#   보정 구간(C_STAND, C_HINGE, C_SIT, C_SLUMP)을 먼저 넣고 과제를 무작위 순서로 붙인다.
#   REST         과제 사이 5초 휴식 (다음 과제의 바른 기본 자세). 라벨이 없어 학습에선 빠지고,
#                여기서 난 알림은 오알림으로 센다.
#   hold()       직전 각도에서 목표 각도로 1초 램프 후 유지 + 미세 흔들림
#   C_HINGE      골반 0→40°→0 두 번. 요추는 hinge_leak만큼만 따라간다 (완벽한 힙 힌지는 없다).
#   moving       2 Hz 걸음 주기로 각도·가속도 출렁임. wobble이 크면 더 흔들린다.
#   transition   3초마다 앉기↔서기, 전환 1초 동안 상체 앞쏠림.
#
# [make_subject(sid, seed)]  같은 seed면 항상 같은 사람이 나온다 (재현성).
