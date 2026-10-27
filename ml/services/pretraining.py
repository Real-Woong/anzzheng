# Service 계층: HARTH 사전학습 → 우리 데이터 미세조정, 그리고 "처음부터 학습"과 비교
# 배울 개념: 전이학습은 "효과가 있다고 가정"하지 말고, 같은 조건의 비교군과 함께 잰다.
import numpy as np

from ml.config import ACTIVITIES, PRETRAIN, FINETUNE, TRAIN
from ml.services.evaluator import window_metrics
from ml.services.trainer import train, predict, fit_norm
from ml.services.windowing import training_windows


def activity_report(model, meta, X, y, device):
    """활동 정확도, 클래스별 재현율, macro-F1"""
    pred = predict(model, meta, X, device)[0].argmax(1)
    out = {"acc": float((pred == y).mean())}
    f1s = []
    for i, a in enumerate(ACTIVITIES):
        tp = ((pred == i) & (y == i)).sum()
        fp = ((pred == i) & (y != i)).sum()
        fn = ((pred != i) & (y == i)).sum()
        if (y == i).any():
            out[f"rec_{a}"] = float(tp / (tp + fn))
            f1s.append(2 * tp / max(2 * tp + fp + fn, 1))
    out["macro_f1"] = float(np.mean(f1s))
    return out


def pretrain(Xh, yh, norm, device, log=print):
    """활동 헤드만 학습한다. 자세 라벨은 전부 -1(마스킹)이라 자세 손실은 0."""
    no_posture = np.full((len(yh), 2), -1)
    return train(Xh, yh, no_posture, device, cfg=PRETRAIN, log=log, norm=norm, balance=True)


def compare(harth_train, harth_test, synth_train, synth_test, device, log=print):
    """harth_*: (X, y) 튜플, synth_*: Session 리스트"""
    Xs, ya, yp = training_windows(synth_train)
    norm = fit_norm(Xs)                     # 목표 도메인(우리 데이터) 기준으로 고정

    log("[1/3] HARTH 사전학습")
    pre, _ = pretrain(*harth_train, norm, device, log)
    log("[2/3] 우리 데이터 미세조정 (사전학습 가중치에서 시작)")
    ft, meta = train(Xs, ya, yp, device, cfg=FINETUNE, log=log, init=pre.state_dict(), norm=norm)
    log("[3/3] 비교군: 우리 데이터만으로 처음부터")
    sc, _ = train(Xs, ya, yp, device, cfg=TRAIN, log=log, norm=norm)

    rows = {}
    for name, m in (("pretrained_only", pre), ("finetuned", ft), ("scratch", sc)):
        r = {f"harth_{k}": v for k, v in activity_report(m, meta, *harth_test, device).items()}
        r.update({f"synth_{k}": v for k, v in window_metrics(m, meta, synth_test, device).items()})
        rows[name] = r
    return rows, ft, meta, pre


# ======================================================================
# 설명
# ======================================================================
#
# [세 모델]
#   pretrained_only  HARTH로 활동만 배운 모델. 자세 헤드는 학습 안 됨 (참고용).
#   finetuned        위 가중치에서 출발해 우리 데이터로 다시 학습 (전이학습).
#   scratch          랜덤 가중치에서 우리 데이터로만 학습 (비교군).
#
# [평가 두 곳]
#   harth_*  학습에 안 쓴 HARTH 참가자. "일상생활에서 활동 게이트가 버티는가".
#            우리 데이터는 스크립트 과제뿐이라, 이 점수가 일반화의 유일한 단서다.
#   synth_*  학습에 안 쓴 가상 참가자. 미세조정 후에도 원래 과제를 잘하는가 (잊지 않았나).
#
# [norm 고정 이유]  사전학습과 미세조정이 다른 정규화를 쓰면, 같은 숫자가 다른 의미가 되어
#   사전학습 가중치가 쓸모없어진다. 최종 사용처인 우리 데이터 기준으로 통일한다.
