# Repository 계층: HARTH 공개 데이터셋 로드 (UCI #779, CC BY 4.0)
# 22명, 허리 아래쪽+허벅지 가속도계, 일상생활 약 2시간, 영상 기반 라벨.
# 주의: UCI 설명은 50 Hz지만 S006은 실제로 100 Hz다 (타임스탬프 간격으로 확인).
# 다운로드: https://archive.ics.uci.edu/static/public/779/harth.zip → data/harth/S006.csv ...
from pathlib import Path
import numpy as np
import pandas as pd

from ml.config import ROOT, FS

HARTH_DIR = ROOT / "data" / "harth"

# HARTH 라벨 코드 → 우리 활동. None = 사용 안 함 (자전거는 우리 과제에 없다)
LABEL_TO_ACT = {
    1: "moving",   # walking
    2: "moving",   # running
    3: "moving",   # shuffling
    4: "moving",   # stairs up
    5: "moving",   # stairs down
    6: "static",   # standing
    7: "static",   # sitting
    8: "static",   # lying
}
STANDING, SITTING = 6, 7


def list_subjects(root=HARTH_DIR):
    return sorted(p.stem for p in Path(root).glob("S*.csv"))


def load_subject(sid, root=HARTH_DIR):
    """반환: acc (N, 3) [g] 우리 축(x=척추 위, z=등 바깥쪽)으로 변환, label (N,) int. 50 Hz."""
    df = pd.read_csv(Path(root) / f"{sid}.csv",
                     usecols=["timestamp", "back_x", "back_y", "back_z", "label"])
    # HARTH 등 센서 축: x=아래, y=왼쪽, z=앞 → 우리 축: x=위, z=뒤(등 바깥쪽)
    acc = np.stack([-df.back_x, df.back_y, -df.back_z], 1).astype(np.float32)
    label = df.label.to_numpy()

    # 대부분 50 Hz지만 S006은 100 Hz다. k개씩 평균 내서 50 Hz로 맞춘다.
    dt = pd.to_datetime(df.timestamp).diff().dt.total_seconds().median()
    k = max(int(round((1 / FS) / dt)), 1)
    if k > 1:
        n = len(acc) // k * k
        acc = acc[:n].reshape(-1, k, 3).mean(1)
        label = label[:n:k]
    return acc, label
