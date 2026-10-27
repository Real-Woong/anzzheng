# Controller 계층: 장시간 실험 — 학습량·초기화·epoch를 바꿔 가며 "잘못 울린 알람"이 줄어드는지 재서 CSV에 쌓는다
# 실행(맥미니 하루): caffeinate -i nohup .venv/bin/python -u -m ml.app.longrun --hours 24 > longrun.log 2>&1 &
# 중간에 꺼도 된다. 끝난 실험은 artifacts/longrun_results.csv에 남고, 다시 실행하면 이어서 한다.
import argparse
import itertools
import time
from pathlib import Path
import numpy as np
import pandas as pd

from ml.config import ARTIFACTS, TRAIN, FINETUNE
from ml.repo import model_repo
from ml.repo.harth_repo import list_subjects
from ml.repo.session_repo import synth_subjects
from ml.services.evaluator import replay, score
from ml.services.harth_windowing import harth_windows
from ml.services.pretraining import pretrain
from ml.services.trainer import pick_device, train, fit_norm
from ml.services.windowing import training_windows
from ml.sim.daily import make_cohort

RESULTS = ARTIFACTS / "longrun_results.csv"
TEST_SEED = 7777          # 평가용 사람들. 학습용(아래 seed들)과 겹치지 않는다
STRESS_SLIDE = 12.0       # 스트레스: 센서가 12° 밀리는 사람들 (기본 시뮬은 2~6°)


def eval_sets(users, days, minutes):
    """고정 평가 데이터 2개. 보통 조건 / 센서가 크게 밀리는 조건."""
    return {name: [s for _, ds in make_cohort(users, days, TEST_SEED, minutes, slide=sl).values() for s in ds]
            for name, sl in (("normal", None), ("stress", STRESS_SLIDE))}


def evaluate(engine, sets, bundle=None):
    out = {}
    for name, sessions in sets.items():
        tp = fn = fa = 0
        hours = 0.0
        for s, cal_end in sessions:
            al, seg = replay(s, engine, "lumbar_flexion", bundle)
            r = score(al, seg, "lumbar_flexion")
            tp += r["TP"]; fn += r["FN"]; fa += r["FA"]
            hours += (len(s.df) - cal_end) / 50 / 3600
        out[f"{name}_recall"] = tp / max(tp + fn, 1)
        out[f"{name}_FA_h"] = fa / hours
    return out


def train_windows(n_lab, n_daily, seed):
    """실험실 프로토콜 사람 n_lab명 + 하루 생활 사람 n_daily명. 하루 데이터는 메모리 때문에 1/3만 쓴다."""
    parts = [training_windows(synth_subjects(n_lab, seed=seed))]
    if n_daily:
        for _, ds in make_cohort(n_daily, 3, seed=500000 + seed, wear_min=15).values():
            for s, _ in ds:
                X, a, p = training_windows([s])
                parts.append((X[::3], a[::3], p[::3]))
    return tuple(np.concatenate(z) for z in zip(*parts))


def run_trial(cfg, harth, device):
    t_cfg = dict(TRAIN, epochs=cfg["epochs"], seed=cfg["seed"])
    X, ya, yp = train_windows(cfg["n_lab"], cfg["n_daily"], cfg["seed"])
    if cfg["init"] == "harth":
        norm = fit_norm(X)
        pre, _ = pretrain(*harth, norm, device, log=lambda *a: None)
        ft_cfg = dict(FINETUNE, epochs=max(cfg["epochs"] // 2, 1), seed=cfg["seed"])
        model, meta = train(X, ya, yp, device, cfg=ft_cfg, log=lambda *a: None, init=pre.state_dict(), norm=norm)
    else:
        model, meta = train(X, ya, yp, device, cfg=t_cfg, log=lambda *a: None)
    return model, meta, len(X)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=24, help="이 시간이 지나면 다음 실험을 시작하지 않고 끝낸다")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--quick", action="store_true", help="동작 확인용 초소형 실험")
    for k in ("n_lab", "n_daily", "epochs"):
        ap.add_argument(f"--{k.replace('_', '-')}", type=int, nargs="+", help=f"grid의 {k} 값을 덮어쓴다")
    ap.add_argument("--init", nargs="+", choices=["scratch", "harth"], help="grid의 init 값을 덮어쓴다")
    ap.add_argument("--results", help="결과 CSV 경로(기본: artifacts/longrun_results.csv). 실험을 동시에 돌릴 때는 서로 다른 파일을 쓴다")
    a = ap.parse_args()
    global RESULTS
    if a.results:
        RESULTS = Path(a.results)

    device = pick_device()
    print(f"device: {device}", flush=True)
    grid = dict(n_lab=[8, 32, 128], n_daily=[0, 20], epochs=[25, 60], init=["scratch", "harth"])
    grid.update({k: getattr(a, k) for k in grid if getattr(a, k)})
    users, days, minutes = 10, 5, 60
    if a.quick:
        grid, users, days, minutes = dict(n_lab=[8], n_daily=[0], epochs=[3], init=["scratch"]), 3, 1, 10
    sets = eval_sets(users, days, minutes)

    done = pd.read_csv(RESULTS) if RESULTS.exists() else pd.DataFrame()
    rows = []
    if done.empty:
        for eng in ("v0a", "v0b"):
            rows.append(dict(engine=eng, **evaluate(eng, sets)))
    harth = None
    if "harth" in grid["init"]:
        sids = list_subjects()
        harth = harth_windows(sids[:-5])[:2]

    start, best = time.time(), float("inf")
    trials = [dict(zip(grid, v), seed=sd) for sd in range(1, a.seeds + 1) for v in itertools.product(*grid.values())]
    for cfg in trials:
        if time.time() - start > a.hours * 3600:
            print("시간 예산 소진, 종료", flush=True)
            break
        key = {k: cfg[k] for k in ("n_lab", "n_daily", "epochs", "init", "seed")}
        if not done.empty and (done[list(key)].eq(pd.Series(key)).all(axis=1)).any():
            continue
        t0 = time.time()
        model, meta, nwin = run_trial(cfg, harth, device)
        res = evaluate("cnn", sets, (model, meta, device))
        rows.append(dict(engine="cnn", **key, windows=nwin, minutes=round((time.time() - t0) / 60, 1), **res))
        print(f"{key} windows={nwin} {res} ({rows[-1]['minutes']}분)", flush=True)
        pd.concat([done, pd.DataFrame(rows)], ignore_index=True).to_csv(RESULTS, index=False)   # 매 실험마다 저장
        score_ = res["normal_FA_h"] + res["stress_FA_h"]
        if min(res["normal_recall"], res["stress_recall"]) >= 0.8 and score_ < best:
            best = score_
            model_repo.save(ARTIFACTS / "posture_cnn_best.pt", model.cpu(), meta)
    print(pd.read_csv(RESULTS).round(3).to_string(), flush=True)


if __name__ == "__main__":
    main()


# ======================================================================
# 변수·함수 설명
# ======================================================================
#
# [왜 이렇게 하나]  모델은 약 1만 파라미터라 학습 자체는 몇 분이면 끝난다. 하루가 필요한 이유는
#   "학습량(사람 수)·초기화(사전학습)·epoch·시드"를 바꿔 가며 반복 측정하기 위해서다.
#   시드를 3개 두는 이유: 한 번의 결과가 운인지 알려면 같은 설정을 여러 번 돌려야 한다.
# [grid]  n_lab 실험실형 가상인 수, n_daily 하루 생활형 가상인 수, epochs, init(처음부터/HARTH 사전학습).
# [평가 2종]  normal = 기본 시뮬(센서 밀림 2~6°), stress = 12° 밀림. FA_h = 시간당 잘못 울린 알람.
# [baseline]  v0a, v0b 규칙 결과를 첫 줄에 같이 기록 → CNN이 규칙보다 나은지 같은 표에서 비교.
# [best 저장]  재현율 0.8 이상 중 잘못 울린 알람(normal+stress 합)이 가장 적은 모델 → posture_cnn_best.pt
# [한계]  전부 가상 사람 기준이다. 이 표는 "학습량을 늘리면 어느 조건에서 개선되는가"를 보는 것이지
#   실제 정확도를 말해 주지 않는다.
