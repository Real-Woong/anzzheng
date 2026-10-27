# 설정 계층 (Spring의 application.yml, FE의 constants 역할)
# 숫자를 코드 곳곳에 흩지 않고 여기 한 곳에서 바꾼다.
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"

FS = 50                 # Hz
WIN = 100               # 2초 윈도우
STRIDE = 25             # 0.5초마다 판정
MARGIN = 2 * FS         # 과제 시작·끝 2초는 학습에서 제외

# 수집 라벨(과제 ID) → 학습 타깃. -1 = 마스킹(정지 아님)
# (activity, lumbar_flexion, trunk_lean)
TASK_MAP = {
    "T01": ("static", 0, 0),
    "T02": ("static", 1, 0),
    "T03": ("static", 1, 1),
    "T04": ("static", 0, 1),
    "T05": ("static", 1, 1),
    "T06": ("static", 0, 1),
    "T07": ("static", 0, 0),
    "T08": ("moving", -1, -1),
    "T09": ("transition", -1, -1),
}
CALIB_TASKS = ("C_STAND", "C_HINGE", "C_SIT", "C_SLUMP")

ACTIVITIES = ("static", "moving", "transition")
POSTURES = ("lumbar_flexion", "trunk_lean")

# 모델 입력 채널. ml/signal/process.py의 derive_channels가 만든다.
CHANNELS = ("pelvis_rel", "lumbar_c", "lumbar_norm", "trunk_rel",
            "u_gyr_n", "l_gyr_n", "u_acc_n", "l_acc_n")

DEFAULT_SLUMP_RANGE = 20.0   # C_SLUMP 보정을 못 했을 때 쓰는 개인 굴곡 범위(°)
MIN_SLUMP_RANGE = 5.0

TRAIN = dict(epochs=25, batch_size=128, lr=3e-3, weight_decay=1e-4, seed=0)
PRETRAIN = dict(epochs=8, batch_size=512, lr=3e-3, weight_decay=1e-4, seed=0)   # HARTH (데이터가 많다)
FINETUNE = dict(epochs=15, batch_size=128, lr=1e-3, weight_decay=1e-4, seed=0)  # 작은 lr로 덜 잊게

# 엔진별 (enter, exit). 모두 같은 상태 머신에 넣는다.
THRESH = {
    ("v0a", "lumbar_flexion"): (20.0, 12.0),        # 센서 1개 기울기(°), 기존 교정기
    ("v0b_legacy", "lumbar_flexion"): (20.0, 12.0), # 서기 보정만 + 절대 각도(°)
    ("v0b", "lumbar_flexion"): (0.6, 0.35),         # 개인 범위 대비 비율
    ("cnn", "lumbar_flexion"): (0.6, 0.4),          # 확률
    ("v0a", "trunk_lean"): (25.0, 15.0),
    ("v0b_legacy", "trunk_lean"): (25.0, 15.0),
    ("v0b", "trunk_lean"): (25.0, 15.0),
    ("cnn", "trunk_lean"): (0.6, 0.4),
}
ENGINES = ("v0a", "v0b_legacy", "v0b", "cnn")

RULE_MOVING_GYR = 6.0   # 규칙 활동 판정: 노이즈 바닥의 몇 배면 "이동"
RULE_MOVING_ACC = 6.0

PROFILES = {
    "lumbar_flexion_guard": "lumbar_flexion",   # 허리 굽힘 주의
    "trunk_lean_guard": "trunk_lean",           # 상체 숙임 주의
}

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"
