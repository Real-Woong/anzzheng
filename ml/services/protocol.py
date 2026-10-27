# Service 계층: 수집 프로토콜 (ML 문서 4.4). 어떤 순서로, 몇 초씩, 뭐라고 안내할지.
# 시뮬레이터(synth.py, mujoco_sim.py)도 같은 길이를 쓴다. 바꾸면 양쪽을 같이 바꿔야 가짜 모드 라벨이 맞는다.
from dataclasses import dataclass
import numpy as np

from ml.config import FS

REST_S = 5

CALIB_STEPS = (
    ("C_STAND", 30, "똑바로 서서 가만히 있기"),
    ("C_HINGE", 16, "허리는 곧게, 고관절만 접어 숙였다 펴기 2번 (4초 내려가고 4초 올라오기)"),
    ("C_SIT", 20, "의자에 바르게 앉기"),
    ("C_SLUMP", 10, "편하게 구부정하게 앉기 (아프지 않은 만큼만)"),
)

GUIDE = {
    "T01": ("sit", "바르게 앉기"),
    "T02": ("sit", "구부정하게 앉기 (허리를 말아서)"),
    "T03": ("sit", "허리를 말고 책상 쪽으로 숙여 앉기"),
    "T04": ("sit", "허리는 곧게, 고관절만 접어 앞으로 기울여 앉기"),
    "T05": ("stand", "서서 허리를 말아 숙이기 (통증이 있으면 바로 멈추기)"),
    "T06": ("stand", "서서 힙 힌지 (허리 곧게, 엉덩이를 뒤로 빼며 숙이기)"),
    "T07": ("stand", "바르게 서기"),
    "T08": ("stand", "제자리 또는 복도 걷기"),
    "T09": ("stand", "3초마다 앉기 ↔ 일어서기 반복"),
}
ROUNDS = {1: (tuple(GUIDE), 30), 2: (("T01", "T02", "T04", "T06", "T08"), 20)}   # (과제, 초)


@dataclass
class Step:
    task_id: str
    n: int            # 샘플 수 (= 초 × FS)
    text: str


def task_order(round_no, seed):
    """과제 순서는 무작위 (순서 효과 방지). seed를 기록해 두면 나중에 재현할 수 있다."""
    tasks, _ = ROUNDS[round_no]
    return [str(t) for t in np.random.default_rng(seed).permutation(tasks)]


def build_steps(order, dur_s):
    steps = [Step(t, s * FS, txt) for t, s, txt in CALIB_STEPS]
    for tid in order:
        posture, txt = GUIDE[tid]
        rest = "편하게 앉아 쉬기" if posture == "sit" else "편하게 서서 쉬기"
        steps.append(Step("REST", REST_S * FS, f"{rest} — 다음: {tid} {txt}"))
        steps.append(Step(tid, dur_s * FS, txt))
    return steps
