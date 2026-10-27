# Controller 계층: LOSO 평가 실행 → 결과 표 출력
# 실행: python -m ml.app.loso --subjects 6
# Controller는 입력을 받고 Service를 호출하고 결과를 보여 주기만 한다. 계산 로직은 넣지 않는다.
import argparse
import time
import pandas as pd

from ml.repo.session_repo import SIMS, synth_subjects
from ml.services.evaluator import run_fold
from ml.services.trainer import pick_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sim", default="synth", choices=SIMS, help="synth=공식, mujoco=MyoBack 물리 모델")
    args = ap.parse_args()

    device = pick_device()
    print(f"device: {device}  sim: {args.sim}")
    sessions = synth_subjects(args.subjects, args.seed, args.sim)
    sids = sorted({s.sid for s in sessions})

    rows, wins = [], []
    for sid in sids:
        t0 = time.time()
        print(f"[fold] test={sid}")
        tr = [s for s in sessions if s.sid != sid]
        te = [s for s in sessions if s.sid == sid]
        r, w = run_fold(tr, te, device)
        rows += r
        wins.append({"sid": sid, **w})
        print(f"  {time.time() - t0:.1f}s  " + "  ".join(f"{k}={v:.3f}" for k, v in w.items()))

    df = pd.DataFrame(rows)
    df["FA_hard"] = df["FA_tasks"].apply(lambda t: sum(x in ("T01", "T04", "T06") for x in t))
    df["FA_move"] = df["FA_tasks"].apply(lambda t: sum(x in ("T08", "T09") for x in t))

    print("\n=== 이벤트 수준 (전체 참가자 합) ===")
    summary = df.groupby(["profile", "engine"], sort=False)[["TP", "FN", "FA", "FA_hard", "FA_move"]].sum()
    summary["recall"] = summary["TP"] / (summary["TP"] + summary["FN"])
    print(summary.round(2).to_string())

    print("\n=== 참가자별 FA (허리 굽힘 주의) ===")
    lf = df[df.profile == "lumbar_flexion_guard"]
    print(lf.pivot_table(index="sid", columns="engine", values="FA", aggfunc="sum")[
        ["v0a", "v0b_legacy", "v0b", "cnn"]].to_string())

    print("\n=== CNN 윈도우 수준 ===")
    print(pd.DataFrame(wins).set_index("sid").round(3).to_string())


if __name__ == "__main__":
    main()
