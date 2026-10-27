# 도메인 타입 계층 (백엔드의 Entity/DTO, FE의 types.ts 역할)
# 계층 사이를 오가는 데이터 모양을 여기서만 정의한다. 로직은 넣지 않는다.
from dataclasses import dataclass, asdict
import pandas as pd


@dataclass
class Body:
    """시뮬레이터용 '정답' 신체 파라미터. 실제 데이터에서는 알 수 없다."""
    gain: float          # 피부 센서가 실제 요추 굴곡을 몇 배로 읽는가 (연부조직·부착 위치)
    crosstalk: float     # 골반 기울기가 굴곡각으로 새어 드는 비율
    sit_pelvis: float    # 앉으면 골반이 뒤로 눕는 각도(°, 음수)
    sit_loss: float      # 앉으면 줄어드는 요추 전만(°)
    flex_scale: float    # 같은 지시에도 사람마다 다른 굽힘 크기
    wobble: float        # 연부조직 흔들림 배수 (체지방·근육)
    hinge_leak: float    # 힙 힌지를 해도 실제로 새는 요추 굴곡 비율
    # 아래 3개는 MuJoCo 시뮬레이터만 쓴다 (기본값이 있어 synth.py는 그대로 동작)
    spine_scale: float = 1.0   # 요추 길이 배수 (척추 길이 개인차)
    fat_cm: float = 2.0        # 센서 아래 연부조직 두께 (체지방·근육)
    rhythm: tuple = ()         # 레벨별 굴곡 분담 비율 L5/S1 … L1/L2 (합 1)


@dataclass
class Calibration:
    """3단 보정 결과. 각도는 °."""
    u_bias: float
    l_bias: float
    pelvis_stand: float
    trunk_stand: float
    stand_offset: float
    crosstalk: float
    sit_offset: float
    slump_ref: float
    acc_noise: float
    gyr_noise: float

    def to_dict(self):
        return asdict(self)


@dataclass
class Session:
    sid: str                  # 참가자 ID (LOSO 분할 단위)
    name: str                 # 세션 이름
    df: pd.DataFrame          # 원시 IMU (LSB)
    tasks: pd.DataFrame       # start_us, end_us, task_id
    body: Body | None = None  # 시뮬레이션일 때만 존재
