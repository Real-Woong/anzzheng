# Service 계층: 실시간 판정. 샘플이 들어올 때마다 쌓아 두고, 0.5초마다 최근 2초 윈도우를 판정한다.
# 오프라인 재생(evaluator.replay)과 같은 엔진·같은 상태 머신을 쓴다. 다른 점은 "미래를 못 본다"는 것뿐.
from collections import deque
import numpy as np
import pandas as pd

from ml.config import FS, WIN, STRIDE, THRESH, CHANNELS
from ml.judge.state_machine import RuleStateMachine
from ml.repo.stream_repo import COLUMNS
from ml.services import engines
from ml.signal.process import calibrate, derive_channels

BUFFER_S = 20   # 상보 필터를 다시 돌릴 구간. 필터가 안정되려면 몇 초의 앞부분이 필요하다


def calibrate_rows(rows, calib_seg):
    """녹화 중 쌓인 샘플 + 보정 구간(샘플 인덱스) → Calibration"""
    return calibrate(pd.DataFrame(rows, columns=COLUMNS), calib_seg)


class LiveMonitor:
    def __init__(self, cal, engine, watch, bundle=None):
        self.cal, self.engine, self.watch, self.bundle = cal, engine, watch, bundle
        self.sm = RuleStateMachine(*THRESH[(engine, watch)])
        self.buf = deque(maxlen=BUFFER_S * FS)   # 꽉 차면 가장 오래된 샘플이 자동으로 빠진다
        self.count = 0
        self.last = None   # 마지막 판정 (값, 활동, 상태) — 화면 표시용

    def feed(self, row):
        """샘플 1개 추가. 알림을 보내야 하면 True."""
        self.buf.append(row)
        self.count += 1
        if len(self.buf) < WIN or self.count % STRIDE:
            return False
        ch = derive_channels(pd.DataFrame(list(self.buf), columns=COLUMNS), self.cal)
        X = np.stack([ch[c][-WIN:] for c in CHANNELS]).astype(np.float32)[None]   # (1, C, WIN)
        values, acts = engines.run(self.engine, self.watch, ch, X, [len(self.buf)], WIN, self.bundle)
        alert = self.sm.step(float(values[0]), acts[0], STRIDE / FS)
        self.last = (float(values[0]), acts[0], self.sm.state)
        return alert
