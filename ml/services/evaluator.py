# Service 계층: 세션을 시간 순서대로 재생해 알림을 만들고 정답 이벤트와 비교 (ML 문서 5.3)
import numpy as np

from ml.config import FS, WIN, STRIDE, THRESH, TASK_MAP, POSTURES, CALIB_TASKS, PROFILES, ENGINES
from ml.judge.state_machine import RuleStateMachine
from ml.services import engines
from ml.services.trainer import predict, train
from ml.services.windowing import prepare, sliding, training_windows

GRACE = 2 * FS   # 과제 끝난 뒤 2초 안의 알림도 정탐으로 인정


def monitor_start(seg):
    """보정이 끝난 시점부터 감시한다."""
    return max(i1 for k, v in seg.items() if k in CALIB_TASKS for _, i1 in v)


def replay(session, engine, watch, bundle=None):
    """반환: 알림이 나간 샘플 인덱스 리스트, seg"""
    _, ch, seg = prepare(session)
    X, ends = sliding(ch, start=monitor_start(seg))
    values, acts = engines.run(engine, watch, ch, X, ends, WIN, bundle)
    enter, exit_ = THRESH[(engine, watch)]
    sm = RuleStateMachine(enter, exit_)
    alerts = [int(e) for e, v, a in zip(ends, values, acts) if sm.step(v, a, STRIDE / FS)]
    return alerts, seg


def score(alerts, seg, watch):
    """정답 이벤트 = 감시 대상 과제 구간. 구간 안 첫 알림 = TP, 나머지 구간 밖 알림 = FA."""
    k = 1 + POSTURES.index(watch)
    events = [(i0, i1) for tid, spans in seg.items()
              if tid in TASK_MAP and TASK_MAP[tid][k] == 1 for i0, i1 in spans]
    tp = sum(any(i0 <= a <= i1 + GRACE for a in alerts) for i0, i1 in events)
    fa_tasks = []
    for a in alerts:
        if any(i0 <= a <= i1 + GRACE for i0, i1 in events):
            continue
        fa_tasks += [tid for tid, spans in seg.items() for i0, i1 in spans if i0 <= a < i1] or ["?"]
    return {"TP": tp, "FN": len(events) - tp, "FA": len(fa_tasks), "FA_tasks": fa_tasks}


def window_metrics(model, meta, sessions, device):
    """윈도우 단위: 활동 정확도, 자세 요소별 F1 (정지 윈도우만)"""
    X, ya, yp = training_windows(sessions)
    pa, pp = predict(model, meta, X, device)
    out = {"act_acc": float((pa.argmax(1) == ya).mean())}
    for j, name in enumerate(POSTURES):
        m = yp[:, j] >= 0
        pred, true = pp[m, j] >= 0.5, yp[m, j] == 1
        tp, fp, fn = (pred & true).sum(), (pred & ~true).sum(), (~pred & true).sum()
        out[f"{name}_f1"] = float(2 * tp / max(2 * tp + fp + fn, 1))
    return out


def run_fold(train_sessions, test_sessions, device, log=print):
    """LOSO 한 번: 학습 → 테스트 참가자에서 모든 엔진 × 모든 프로필 재생"""
    model, meta = train(*training_windows(train_sessions), device, log=log)
    bundle = (model, meta, device)
    rows = []
    for s in test_sessions:
        for profile, watch in PROFILES.items():
            for eng in ENGINES:
                alerts, seg = replay(s, eng, watch, bundle if eng == "cnn" else None)
                rows.append({"sid": s.sid, "session": s.name, "profile": profile,
                             "engine": eng, **score(alerts, seg, watch)})
    return rows, window_metrics(model, meta, test_sessions, device)
