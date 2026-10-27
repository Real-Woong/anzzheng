# Service 계층: 세션 → (보정, 채널, 윈도우, 라벨)
# 배울 개념: 학습용 윈도우는 "라벨이 확실한 구간"만, 평가용 윈도우는 세션 전체를 빠짐없이.
import numpy as np

from ml.config import WIN, STRIDE, MARGIN, TASK_MAP, CHANNELS, ACTIVITIES, CALIB_TASKS
from ml.repo.session_repo import segments
from ml.signal.process import calibrate, derive_channels


def prepare(session):
    """보정 → 채널. 보정 구간은 각 보정 과제의 첫 번째 구간을 쓴다."""
    seg = segments(session)
    calib_seg = {k: v[0] for k, v in seg.items() if k in CALIB_TASKS}
    cal = calibrate(session.df, calib_seg)
    return cal, derive_channels(session.df, cal), seg


def sliding(ch, start=0):
    """채널 dict → X (n, C, WIN), ends (n,). ends = 각 윈도우의 끝 샘플(미포함)."""
    data = np.stack([ch[c] for c in CHANNELS]).astype(np.float32)   # (C, N)
    ends = np.arange(start + WIN, data.shape[1] + 1, STRIDE)
    X = np.stack([data[:, e - WIN:e] for e in ends])
    return X, ends


def label_windows(ends, seg):
    """윈도우마다 (activity id, [lumbar_flexion, trunk_lean], 유효 여부)."""
    y_act = np.full(len(ends), -1)
    y_post = np.full((len(ends), 2), -1)
    for tid, spans in seg.items():
        if tid not in TASK_MAP:
            continue
        act, lf, tl = TASK_MAP[tid]
        for i0, i1 in spans:
            ok = (ends - WIN >= i0 + MARGIN) & (ends <= i1 - MARGIN)
            y_act[ok] = ACTIVITIES.index(act)
            y_post[ok] = (lf, tl)
    return y_act, y_post, y_act >= 0


def training_windows(sessions):
    Xs, A, P = [], [], []
    for s in sessions:
        _, ch, seg = prepare(s)
        X, ends = sliding(ch)
        a, p, ok = label_windows(ends, seg)
        Xs.append(X[ok]); A.append(a[ok]); P.append(p[ok])
    return np.concatenate(Xs), np.concatenate(A), np.concatenate(P)
