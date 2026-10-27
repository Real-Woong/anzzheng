# Controller 계층: HARTH 사전학습 실험 실행 → 비교표 + 모델 저장
# 실행: python -m ml.app.pretrain --harth-test 5 --subjects 8
import argparse
from pathlib import Path
import pandas as pd

from ml.config import ARTIFACTS
from ml.repo import model_repo
from ml.repo.harth_repo import list_subjects
from ml.repo.session_repo import SIMS, synth_subjects
from ml.services.harth_windowing import harth_windows
from ml.services.pretraining import compare
from ml.services.trainer import pick_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--harth-test", type=int, default=5, help="테스트로 뺄 HARTH 참가자 수 (뒤에서부터)")
    ap.add_argument("--subjects", type=int, default=8, help="가상 학습 참가자 수")
    ap.add_argument("--sim", default="synth", choices=SIMS, help="synth=공식, mujoco=MyoBack 물리 모델")
    args = ap.parse_args()

    device = pick_device()
    print(f"device: {device}")
    sids = list_subjects()
    tr_sids, te_sids = sids[:-args.harth_test], sids[-args.harth_test:]
    print(f"HARTH train {len(tr_sids)}명, test {te_sids}")
    Xtr, ytr, _ = harth_windows(tr_sids)
    Xte, yte, _ = harth_windows(te_sids)

    synth_train = synth_subjects(args.subjects, seed=0, sim=args.sim)
    synth_test = synth_subjects(3, seed=99, sim=args.sim)     # 다른 seed = 처음 보는 사람들

    rows, ft, meta, pre = compare((Xtr, ytr), (Xte, yte), synth_train, synth_test, device)

    print("\n=== 비교 (높을수록 좋음) ===")
    print(pd.DataFrame(rows).round(3).to_string())

    meta["data"] = {"pretrain": "HARTH", "harth_train": tr_sids, "synth_subjects": args.subjects, "sim": args.sim}
    model_repo.save(ARTIFACTS / "harth_pretrained.pt", pre.cpu(), meta)
    model_repo.save(ARTIFACTS / "posture_cnn_pt.pt", ft.cpu(), meta)
    print(f"\nsaved: {ARTIFACTS / 'harth_pretrained.pt'}, {ARTIFACTS / 'posture_cnn_pt.pt'}")


if __name__ == "__main__":
    main()
