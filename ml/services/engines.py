# Service 계층: 판정 엔진. 윈도우마다 (감시 값, 활동)을 낸다.
# 엔진만 바뀌고 상태 머신·프로필은 모두 같다 (설계 결정 3).
import numpy as np

from ml.config import ACTIVITIES, POSTURES, RULE_MOVING_GYR, RULE_MOVING_ACC
from ml.services.trainer import predict

# 규칙 엔진이 감시 값으로 쓰는 채널
RULE_CHANNEL = {
    ("v0a", "lumbar_flexion"): "trunk_rel",       # 센서 1개(상부)만: 상체 기울기
    ("v0b_legacy", "lumbar_flexion"): "legacy",   # 서기 보정만 한 굴곡각
    ("v0b", "lumbar_flexion"): "lumbar_norm",     # 3단 보정 + 개인 비율
    ("v0a", "trunk_lean"): "trunk_rel",
    ("v0b_legacy", "trunk_lean"): "trunk_rel",
    ("v0b", "trunk_lean"): "trunk_rel",
}


def rule_activity(ch, ends, win):
    acts = []
    for e in ends:
        g = np.abs(ch["u_gyr_n"][e - win:e]).mean()
        a = ch["u_acc_n"][e - win:e].std()
        acts.append("moving" if g > RULE_MOVING_GYR or a > RULE_MOVING_ACC else "static")
    return acts


def run(engine, watch, ch, X, ends, win, bundle=None):
    """반환: values (n,), activities [str] * n"""
    if engine == "cnn":
        model, meta, device = bundle
        p_act, p_post = predict(model, meta, X, device)
        return p_post[:, POSTURES.index(watch)], [ACTIVITIES[i] for i in p_act.argmax(1)]
    c = ch[RULE_CHANNEL[(engine, watch)]]
    values = np.array([c[e - win:e].mean() for e in ends])
    return values, rule_activity(ch, ends, win)
