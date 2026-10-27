# Controller 계층: 학습에 안 쓴 새 참가자 1명을 재생하며 알림을 출력 (데모 리허설)
# 실행: python -m ml.app.replay --profile lumbar_flexion_guard --engine cnn --llm
import argparse
from pathlib import Path

from ml.config import ARTIFACTS, FS, PROFILES, ENGINES
from ml.repo import model_repo
from ml.repo.session_repo import SIMS, synth_subjects
from ml.services.evaluator import replay, score
from ml.services.notifier import compose
from ml.services.trainer import pick_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ARTIFACTS / "posture_cnn.pt"))
    ap.add_argument("--profile", default="lumbar_flexion_guard", choices=list(PROFILES))
    ap.add_argument("--engine", default="cnn", choices=ENGINES)
    ap.add_argument("--seed", type=int, default=99)   # 학습 seed와 달라야 처음 보는 사람
    ap.add_argument("--llm", action="store_true")
    ap.add_argument("--sim", default="synth", choices=SIMS, help="synth=공식, mujoco=MyoBack 물리 모델")
    args = ap.parse_args()

    watch = PROFILES[args.profile]
    bundle = None
    if args.engine == "cnn":
        device = pick_device()
        model, meta = model_repo.load(Path(args.model), device)
        bundle = (model, meta, device)

    session = synth_subjects(1, args.seed, args.sim)[0]
    alerts, seg = replay(session, args.engine, watch, bundle)
    task_at = {i: tid for tid, spans in seg.items() for i0, i1 in spans for i in range(i0, i1)}

    print(f"[{args.engine} / {args.profile}] 참가자 {session.sid} {session.name}")
    print("과제 순서:", " → ".join(session.tasks.task_id))
    for a in alerts:
        text, src = compose(watch, use_llm=args.llm)
        print(f"  {a / FS:6.1f}s  {task_at.get(a, '?'):7s}  [{src}] {text}")
    print(score(alerts, seg, watch))


if __name__ == "__main__":
    main()
