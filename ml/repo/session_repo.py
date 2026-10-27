# Repository 계층: 세션 데이터를 "어디서" 가져오는지만 안다 (FE의 api/, 백엔드의 Repo)
# 위 계층(Service)은 데이터가 시뮬레이터에서 왔는지 실제 CSV에서 왔는지 모른다.
from pathlib import Path
import numpy as np
import pandas as pd

from ml.domain.types import Session
from ml.sim.synth import make_subject


SIMS = ("synth", "mujoco")


def synth_subjects(n, seed=0, sim="synth"):
    """S01..Sn 가상 참가자. 같은 (n, seed)면 항상 같은 사람들.
    sim="mujoco"면 MyoBack 물리 모델로 측정한다 (mujoco 패키지가 필요해 그때만 불러온다)."""
    if sim == "mujoco":
        from ml.sim.mujoco_sim import make_subject as make
    else:
        make = make_subject
    out = []
    for i in range(1, n + 1):
        out += make(f"S{i:02d}", seed * 1000 + i)
    return out


def load_csv_session(root, sid, name):
    """실제 수집 데이터: data/raw/S01/sess01_imu.csv + sess01_tasks.csv (ML 문서 4.7)"""
    d = Path(root) / sid
    df = pd.read_csv(d / f"{name}_imu.csv", comment="#")
    tasks = pd.read_csv(d / f"{name}_tasks.csv")
    if "quality" in tasks:
        tasks = tasks[tasks["quality"] != "bad"]
    return Session(sid, name, df, tasks[["start_us", "end_us", "task_id"]].reset_index(drop=True))


def segments(session):
    """task_id → [(시작 샘플, 끝 샘플), ...]. 시각(µs)을 샘플 인덱스로 바꾼다."""
    t = session.df["t_us"].to_numpy()
    out = {}
    for r in session.tasks.itertuples():
        i0, i1 = np.searchsorted(t, [r.start_us, r.end_us])
        out.setdefault(r.task_id, []).append((int(i0), int(i1)))
    return out
