# Controller 계층: 데모용 최종 모델 학습 → artifacts/posture_cnn.pt 저장
# 실행: python -m ml.app.train --subjects 8
import argparse
from pathlib import Path

from ml.config import ARTIFACTS
from ml.repo import model_repo
from ml.repo.session_repo import SIMS, synth_subjects
from ml.services.trainer import pick_device, train
from ml.services.windowing import training_windows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sim", default="synth", choices=SIMS, help="synth=공식, mujoco=MyoBack 물리 모델")
    ap.add_argument("--out", default=str(ARTIFACTS / "posture_cnn.pt"))
    args = ap.parse_args()

    device = pick_device()
    X, ya, yp = training_windows(synth_subjects(args.subjects, args.seed, args.sim))
    print(f"device: {device}  windows: {len(X)}  shape: {X.shape}")
    model, meta = train(X, ya, yp, device)
    meta["data"] = {"source": args.sim, "subjects": args.subjects, "seed": args.seed}
    model_repo.save(Path(args.out), model.cpu(), meta)
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
