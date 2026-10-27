# Controller 계층: 가상 사용자 N명 × D일을 생성해 규칙 엔진으로 재생, 사용자·일자별 표 출력
# 실행: python -m ml.app.simday --users 10 --days 5 --minutes 60 [--save data/sim_daily]
import argparse
from pathlib import Path
import pandas as pd

from ml.repo.session_repo import synth_subjects
from ml.services.trainer import pick_device, train
from ml.services.windowing import training_windows
from ml.services.evaluator import replay, score
from ml.sim.daily import make_cohort

ENG = ("v0a", "v0b")
TRAIN_SEED = 100   # 학습용 사람들은 하루 데이터의 사람들과 다른 seed → 같은 사람이 섞이지 않는다


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=10)
    ap.add_argument("--days", type=int, default=5)
    ap.add_argument("--minutes", type=int, default=60, help="하루 착용 시간(분)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cnn", action="store_true", help="실험실 프로토콜 가상인 8명으로 CNN을 새로 학습해 같이 평가")
    ap.add_argument("--save", default="", help="지정하면 uXX/dayYY_imu.csv, _tasks.csv로 저장 (load_csv_session 호환)")
    a = ap.parse_args()

    engines, bundle = ENG, None
    if a.cnn:
        dev = pick_device()
        model, meta = train(*training_windows(synth_subjects(8, TRAIN_SEED)), dev, log=lambda *x: None)
        bundle, engines = (model, meta, dev), ENG + ("cnn",)
    cohort = make_cohort(a.users, a.days, a.seed, a.minutes)
    rows = []
    for sid, (_, days) in cohort.items():
        for s, cal_end in days:
            if a.save:
                d = Path(a.save) / sid; d.mkdir(parents=True, exist_ok=True)
                s.df.to_csv(d / f"{s.name}_imu.csv", index=False)
                s.tasks.to_csv(d / f"{s.name}_tasks.csv", index=False)
            hours = (len(s.df) - cal_end) / 50 / 3600
            for eng in engines:
                al, seg = replay(s, eng, "lumbar_flexion", bundle if eng == "cnn" else None)
                r = score(al, seg, "lumbar_flexion")
                rows.append(dict(sid=sid, day=s.name, engine=eng, hours=hours, TP=r["TP"], FN=r["FN"], FA=r["FA"],
                                 FA_hard=sum(t in ("T01", "T04", "T06") for t in r["FA_tasks"])))
    df = pd.DataFrame(rows)
    g = df.groupby("engine")[["hours", "TP", "FN", "FA", "FA_hard"]].sum()
    g["recall"] = g.TP / (g.TP + g.FN); g["FA_per_h"] = g.FA / g.hours
    print("=== 전체 (허리 굽힘 주의) ===\n", g.round(2).to_string())
    p = df.pivot_table(index="sid", columns="engine", values=["FA", "TP", "FN"], aggfunc="sum")
    print("\n=== 사용자별 ===\n", p.to_string())
    d = df.assign(n=lambda x: x.day).pivot_table(index="day", columns="engine", values="FA", aggfunc="sum")
    print("\n=== 일자별 FA 합 ===\n", d.to_string())


if __name__ == "__main__":
    main()
