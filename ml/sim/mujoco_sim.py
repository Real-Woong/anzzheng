# 물리 시뮬레이터: MyoBack(MyoSuite) 요추 모델에 가상 IMU 2개를 붙여 "측정"한다.
# synth.py는 각도를 공식으로 IMU 값으로 바꿨다. 여기서는 골반·요추 5레벨을 실제로 움직이고
# MuJoCo 내장 가속도계·자이로가 읽은 값을 쓴다. 출력 모양은 synth.py와 같아서 아래 계층은 그대로다.
from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd
import mujoco
import myo_sim

from ml.config import FS
from ml.domain.types import Session
from ml.sim.synth import TASKS, ROUND2, random_body, _task_targets, _base, to_raw_df

MODEL_DIR = Path(myo_sim.__file__).parent / "models"
LEVELS = ("flex_extension", "L4_L5_FE", "L3_L4_FE", "L2_L3_FE", "L1_L2_FE")   # L5/S1 … L1/L2
RHYTHM = np.array([0.22, 0.24, 0.21, 0.18, 0.15])   # 레벨별 굴곡 분담 (문헌 대략값, 사람마다 흔든다)
LUMBAR = ("lumbar5", "lumbar4", "lumbar3", "lumbar2", "lumbar1")
HIP = [-0.056, -0.078, 0.0]   # 고관절 중심 (sacrum 좌표, m). 골반은 여기를 축으로 기운다
SIT_DROP = 0.45               # 앉으면 골반이 내려가는 높이 (m)
ROOT = ("root_lat", "root_fwd", "root_up", "yaw", "pelvis_tilt")

# sacrum 좌표계: x = 앞, y = 위, z = 오른쪽(대략). 우리 센서 축: x = 척추 위, z = 등 바깥, y = x×z 방향
IMU_AXES = [0, 1, 0, 0, 0, -1]   # site의 x축 = 위(0,1,0), y축 = (0,0,-1) → z축 = 뒤(-1,0,0)
TILT_AXIS = [0, 0, -1]           # 이 축으로 + 회전 = 앞으로 숙임 (센서 자이로 y와 같은 방향)


def random_mj_body(rng):
    """synth의 7개 파라미터 + MuJoCo에서만 의미 있는 3개 (척추 길이, 연부조직 두께, 분담 비율)"""
    return replace(random_body(rng),
                   spine_scale=rng.uniform(0.9, 1.1),
                   fat_cm=rng.uniform(1.0, 4.0),
                   rhythm=tuple(rng.dirichlet(RHYTHM * 40)))


def build_model(body):
    """MyoTorso 뼈대 + 루트 관절 + 피부 위 가상 IMU 2개. 근육·힘줄은 안 쓴다 (운동학 재생만)."""
    xml = f"""<mujoco><compiler meshdir="{MODEL_DIR}" texturedir="{MODEL_DIR}" angle="radian"/>
      <include file="{MODEL_DIR}/torso/assets/myotorso_assets.xml"/>
      <include file="{MODEL_DIR}/head/assets/myohead_simple_assets.xml"/>
      <worldbody><body name="Torso" pos="0 0 1">
        <include file="{MODEL_DIR}/torso/assets/myotorso_chain.xml"/>
      </body></worldbody></mujoco>"""
    spec = mujoco.MjSpec.from_string(xml)
    free = mujoco.mjtLimited.mjLIMITED_FALSE
    slide, hinge = mujoco.mjtJoint.mjJNT_SLIDE, mujoco.mjtJoint.mjJNT_HINGE

    for name in LUMBAR:                      # 척추 길이 개인차: 척추뼈 사이 간격을 늘리고 줄인다
        b = spec.body(name)
        b.pos = np.asarray(b.pos) * body.spine_scale

    sac = spec.body("sacrum")                # 원래는 공중에 고정. 몸 전체가 움직이도록 관절을 단다
    for name, axis in (("root_lat", [0, 0, 1]), ("root_fwd", [1, 0, 0]), ("root_up", [0, 1, 0])):
        sac.add_joint(name=name, type=slide, axis=axis, limited=free)
    sac.add_joint(name="yaw", type=hinge, axis=[0, 1, 0], pos=HIP, limited=free)
    sac.add_joint(name="pelvis_tilt", type=hinge, axis=TILT_AXIS, pos=HIP, limited=free)

    fat = body.fat_cm / 100
    for pre, parent, bone in (("u", "lumbar1", [-0.06, 0.0, 0]), ("l", "sacrum", [-0.11, 0.05, 0])):
        # 피부 = 뼈 위에 연부조직 두께만큼 떨어진 작은 몸체. 뼈에 대해 미끄러지는 각도(slip)를 갖는다.
        skin = spec.body(parent).add_body(name=f"{pre}_skin", pos=[bone[0] - fat, bone[1], 0],
                                          mass=1e-3, inertia=[1e-6] * 3, explicitinertial=True)
        skin.add_joint(name=f"{pre}_slip", type=hinge, axis=TILT_AXIS, limited=free)
        skin.add_site(name=f"{pre}_imu", xyaxes=IMU_AXES, type=mujoco.mjtGeom.mjGEOM_BOX,
                      size=[0.02, 0.012, 0.006], rgba=[1, 0.2, 0.2, 1], group=0)   # 화면에 빨간 상자로 보인다
        for kind, t in (("acc", mujoco.mjtSensor.mjSENS_ACCELEROMETER), ("gyr", mujoco.mjtSensor.mjSENS_GYRO)):
            spec.add_sensor(name=f"{pre}_{kind}", type=t, objtype=mujoco.mjtObj.mjOBJ_SITE, objname=f"{pre}_imu")

    m = spec.compile()
    m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONSTRAINT   # 관절은 우리가 직접 정한다
    return m


def _sway(n, sd, rng, sigma_s=0.4):
    """가우시안으로 거른 흔들림. synth의 사각 창 노이즈는 각도는 매끄러워 보여도
    두 번 미분하면 고주파가 남아, 가만히 서 있는데 가속도가 0.05 g나 출렁였다 (실제는 ~0.01 g)."""
    r = int(3 * sigma_s * FS)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / (sigma_s * FS)) ** 2)
    x = np.convolve(rng.normal(0, 1, n + 2 * r), k / k.sum(), "valid")
    return x * (sd / (x.std() + 1e-9))


def _smooth_ramp(prev, target, n, ramp_s=1.0):
    """synth._ramp의 매끄러운 버전. 직선 램프는 꺾이는 순간 가속도가 무한대가 된다."""
    w = np.clip(np.arange(n) / (ramp_s * FS), 0, 1)
    w = w * w * (3 - 2 * w)
    return prev + (target - prev) * w


def motion(body, task_order, rng, dur_s=30):
    """과제 순서 → 관절 궤적. 반환: {pelvis°, lumbar°, up m, fwd m, lat m, yaw°}, 과제 구간 df"""
    keys = ("pelvis", "lumbar", "up", "fwd", "lat", "yaw")
    parts = {k: [] for k in keys}
    rows = []
    st = {"pelvis": 0.0, "lumbar": 0.0, "up": 0.0, "fwd": 0.0, "t": 0}

    def add(tid, **tr):
        n = len(tr["pelvis"])
        for k in keys:
            parts[k].append(tr.get(k, np.full(n, st.get(k, 0.0))))
        rows.append((st["t"] * 1_000_000 // FS, (st["t"] + n) * 1_000_000 // FS, tid))
        st["t"] += n
        for k in ("pelvis", "lumbar", "up", "fwd"):
            st[k] = parts[k][-1][-1]

    def hold(tid, posture, tp, tl, n, sway=0.5):
        up = -SIT_DROP if posture == "sit" else 0.0
        add(tid, pelvis=_smooth_ramp(st["pelvis"], tp, n) + _sway(n, sway, rng),
            lumbar=_smooth_ramp(st["lumbar"], tl, n) + _sway(n, sway, rng),
            up=_smooth_ramp(st["up"], up, n, ramp_s=1.5))

    hold("C_STAND", "stand", 0.0, 0.0, 30 * FS)
    n = 16 * FS
    tri = 40 * np.interp(np.arange(n), [0, 4 * FS, 8 * FS, 12 * FS, 16 * FS], [0, 1, 0, 1, 0])
    tri = np.convolve(tri, np.ones(FS // 2) / (FS // 2), "same")      # 꼭짓점을 둥글게
    add("C_HINGE", pelvis=st["pelvis"] + tri + _sway(n, 0.5, rng),
        lumbar=st["lumbar"] + body.hinge_leak * tri + _sway(n, 0.5, rng))
    sp, sl = _base(body, "sit")
    hold("C_SIT", "sit", sp, sl, 20 * FS)
    hold("C_SLUMP", "sit", sp - 8, sl + rng.uniform(15, 28) * body.flex_scale, 10 * FS)

    for tid in task_order:
        posture, _, _, kind = TASKS[tid]
        hold("REST", posture, *_base(body, posture), 5 * FS)
        n = dur_s * FS
        t = np.arange(n) / FS
        if kind == "static":
            hold(tid, posture, *_task_targets(body, tid, rng), n)
        elif kind == "moving":   # 걷기: 실제로 앞으로 1.2 m/s 이동 + 2 Hz 상하 출렁임 + 1 Hz 좌우·회전
            w, ph = body.wobble, 2 * np.pi * 2.0 * t
            env = _smooth_ramp(0.0, 1.0, n) * _smooth_ramp(0.0, 1.0, n)[::-1]   # 출발·정지 1초씩
            add(tid, pelvis=_smooth_ramp(st["pelvis"], 0.0, n) + 4 * w * env * np.sin(ph),
                lumbar=_smooth_ramp(st["lumbar"], 0.0, n) + 2 * w * env * np.sin(ph + 0.5),
                up=_smooth_ramp(st["up"], 0.0, n) + 0.02 * w * env * np.sin(ph),
                fwd=st["fwd"] + np.cumsum(1.2 * env) / FS,
                lat=0.02 * w * env * np.sin(ph / 2), yaw=4 * w * env * np.sin(ph / 2))
        else:                    # 전환: 3초마다 앉기 ↔ 서기. 높이가 실제로 0.45 m 오르내린다
            p, l, up = np.empty(n), np.empty(n), np.empty(n)
            cur = (st["pelvis"], st["lumbar"], st["up"])
            sitting = True
            for s in range(0, n, 3 * FS):
                e = min(s + 3 * FS, n)
                tp, tl = _base(body, "sit" if sitting else "stand")
                p[s:e] = _smooth_ramp(cur[0], tp, e - s)
                l[s:e] = _smooth_ramp(cur[1], tl, e - s)
                up[s:e] = _smooth_ramp(cur[2], -SIT_DROP if sitting else 0.0, e - s)
                cur, sitting = (p[e - 1], l[e - 1], up[e - 1]), not sitting
            k = np.arange(n) % (3 * FS)
            p += np.where(k < FS, 25 * np.sin(np.pi * k / FS) ** 2, 0)   # 일어서고 앉을 때 상체 앞쏠림
            add(tid, pelvis=p + _sway(n, 0.5, rng), lumbar=l, up=up)

    traj = {k: np.concatenate(v) for k, v in parts.items()}
    return traj, pd.DataFrame(rows, columns=["start_us", "end_us", "task_id"])


def playback(m, qpos):
    """관절 궤적 (N, nq) → 센서값. 위치를 두 번 미분해 qvel·qacc를 넣고 mj_inverse로 센서를 계산한다."""
    dt = 1 / FS
    qvel = np.gradient(qpos, dt, axis=0)
    qacc = np.gradient(qvel, dt, axis=0)
    d = mujoco.MjData(m)
    out = np.empty((len(qpos), m.nsensordata))
    for i in range(len(qpos)):
        d.qpos[:], d.qvel[:], d.qacc[:] = qpos[i], qvel[i], qacc[i]
        mujoco.mj_inverse(m, d)      # 역동역학: 이 움직임에 필요한 힘 + 센서값(가속도계·자이로)
        out[i] = d.sensordata
    return out


def build_qpos(m, body, traj, rng):
    """관절 궤적(°, m) → MuJoCo qpos 행렬 (N, nq). 화면 재생(ml/app/view.py)도 이걸 쓴다."""
    N = len(traj["pelvis"])
    mount_u, mount_l = rng.normal(0, 5, 2)   # 재부착마다 달라지는 부착 각도
    soft = _sway(N, 0.3 * body.wobble, rng, sigma_s=0.1)   # 연부조직 떨림 (더 빠르다)

    qpos = np.zeros((N, m.nq))
    col = lambda j: m.joint(j).qposadr[0]
    qpos[:, col("root_lat")] = traj["lat"]
    qpos[:, col("root_fwd")] = traj["fwd"]
    qpos[:, col("root_up")] = traj["up"]
    qpos[:, col("yaw")] = np.deg2rad(traj["yaw"])
    qpos[:, col("pelvis_tilt")] = np.deg2rad(traj["pelvis"])
    for name, share in zip(LEVELS, body.rhythm):          # 요추 굴곡을 5레벨에 나눠 준다 (모델은 − = 굴곡)
        qpos[:, col(name)] = -np.deg2rad(traj["lumbar"] * share)
    # 피부 미끄러짐: gain < 1이면 센서가 굴곡을 덜 읽는다. 골반 크로스토크·부착 각도·흔들림도 여기서
    u_slip = (body.gain - 1) * traj["lumbar"] + body.crosstalk * traj["pelvis"] + mount_u + soft
    qpos[:, col("u_slip")] = np.deg2rad(u_slip)
    qpos[:, col("l_slip")] = np.deg2rad(mount_l)
    return qpos


def make_session(body, task_order, rng, dur_s=30):
    traj, tasks = motion(body, task_order, rng, dur_s)
    m = build_model(body)
    qpos = build_qpos(m, body, traj, rng)
    N = len(qpos)
    s = playback(m, qpos)
    adr = lambda name: slice(m.sensor(name).adr[0], m.sensor(name).adr[0] + 3)
    g = 9.81
    imu = []
    for pre in ("u", "l"):
        acc = s[:, adr(f"{pre}_acc")] / g + rng.normal(0, 0.01, (N, 3))
        gyr = np.rad2deg(s[:, adr(f"{pre}_gyr")]) + rng.normal(0, 1, 3) + rng.normal(0, 0.3, (N, 3))
        imu += [acc, gyr]
    return to_raw_df(*imu), tasks


def make_subject(sid, seed):
    """synth.make_subject와 같은 프로토콜: 1회차(전체 과제) + 재부착 후 2회차(단축)"""
    rng = np.random.default_rng(seed)
    body = random_mj_body(rng)
    out = []
    for name, order, dur in (("sess01", list(TASKS), 30), ("sess02", ROUND2, 20)):
        order = [str(t) for t in rng.permutation(order)]
        df, tasks = make_session(body, order, rng, dur_s=dur)
        out.append(Session(sid, name, df, tasks, body))
    return out


# ======================================================================
# 설명
# ======================================================================
#
# [synth.py와 무엇이 다른가]
#   synth   각도(°)를 만들고 cos/sin 공식으로 가속도를 계산. 선가속도는 손으로 넣은 사인파.
#   여기    MyoTorso 뼈대(천골 → L5 → … → L1 → 흉곽)를 실제로 움직이고, 피부 위 가상 IMU를 읽는다.
#           앉고 서는 0.45 m 이동, 걷기 전진, 뼈 바깥 몇 cm에 붙은 센서의 원심·접선 가속도가
#           물리적으로 맞게 나온다. 같은 TASKS·Body 파라미터를 써서 두 시뮬레이터를 비교할 수 있다.
#
# [모델 조립 build_model]
#   MjSpec   XML을 파이썬 객체로 불러와 코드로 고친 뒤 compile한다 (관절·센서 추가).
#   루트 관절 원래 천골이 공중에 고정돼 있어, 좌우·앞뒤·상하 이동 + 수직축 회전 + 골반 기울기를 단다.
#            골반 기울기 축은 고관절 중심(HIP) → 힙 힌지가 "고관절에서 접기"가 된다.
#   피부     뼈 위 fat_cm만큼 떨어진 작은 몸체 + slip 관절. 센서는 여기 붙는다.
#            연부조직이 두꺼우면 회전축에서 멀어져 같은 움직임에도 가속도가 커진다.
#   근육 210개는 안 쓴다. 근육은 "어떤 힘으로 움직이나"이고, 우리는 "움직이면 센서가 뭘 읽나"만 필요하다.
#
# [재생 playback]  운동학 재생 (kinematic playback)
#   qpos     각 시점의 관절값. 우리가 정한다.
#   qvel/qacc 수치 미분. 그래서 램프를 매끄럽게(_smooth_ramp) 만들어야 가속도가 튀지 않는다.
#   mj_inverse 역동역학. 주어진 운동에 필요한 힘을 계산하면서 센서값도 채운다.
#              (mj_step은 힘 → 운동, 시뮬레이션 방향이 반대다)
#
# [개인차 파라미터]  Body (ml/domain/types.py)
#   spine_scale  척추뼈 사이 간격 0.9–1.1배 → 상부 센서 높이·회전 반경이 달라진다
#   fat_cm       센서 아래 연부조직 1–4 cm
#   rhythm       굴곡을 L5/S1…L1/L2에 나누는 비율. 센서 2개는 합만 보므로 기울기는 같고,
#                L1 위치(가속도)만 조금 바뀐다 → "레벨은 IMU로 알 수 없다"를 시뮬레이션으로 확인할 수 있다.
#   gain/crosstalk/mount  피부 미끄러짐(u_slip)으로 넣는다. MuJoCo도 연부조직 변형은 모른다.
