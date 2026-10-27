# Controller 계층: 데이터 수집. 프로토콜을 화면에 안내하면서 IMU를 CSV로 저장하고, 과제 구간을 기록한다.
# 하드웨어 없이:  python -m ml.app.record --fake --speed 10
# 실시간 알림:    python -m ml.app.record --fake --live --engine v0b
# ESP32 연결:     python -m ml.app.record --port /dev/cu.usbserial-0001 --sid S01 --round 1
# 저장 위치는 ML 문서 4.7 형식: data/raw/S01/sess01_imu.csv, sess01_tasks.csv
import argparse
from pathlib import Path
import pandas as pd

from ml.config import ROOT, FS, ENGINES, PROFILES, ARTIFACTS, CALIB_TASKS
from ml.repo.session_repo import load_csv_session
from ml.repo.stream_repo import COLUMNS, FakeStream, SerialStream
from ml.services.evaluator import replay, score
from ml.services.live import LiveMonitor, calibrate_rows
from ml.services.notifier import compose
from ml.services.protocol import ROUNDS, build_steps, task_order


def parse():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--fake", action="store_true", help="하드웨어 없이 MuJoCo 가상 참가자")
    src.add_argument("--port", help="ESP32 시리얼 포트 (ls /dev/cu.* 로 확인)")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--sid", default=None, help="참가자 ID (가짜 모드 기본 FAKE)")
    ap.add_argument("--round", type=int, default=1, choices=[1, 2], help="1=전체 과제, 2=재부착 후 단축")
    ap.add_argument("--seed", type=int, default=0, help="과제 순서 seed (기록해 두면 재현 가능)")
    ap.add_argument("--speed", type=float, default=1.0, help="가짜 모드 재생 배속")
    ap.add_argument("--out", default=str(ROOT / "data" / "raw"))
    ap.add_argument("--live", action="store_true", help="보정이 끝나면 실시간 판정·알림")
    ap.add_argument("--engine", default="v0b", choices=ENGINES)
    ap.add_argument("--profile", default="lumbar_flexion_guard", choices=list(PROFILES))
    ap.add_argument("--model", default=str(ARTIFACTS / "posture_cnn.pt"), help="--engine cnn일 때")
    ap.add_argument("--llm", action="store_true", help="알림 문구를 로컬 Llama로")
    return ap.parse_args()


def load_bundle(args):
    if args.engine != "cnn":
        return None
    from ml.repo import model_repo
    from ml.services.trainer import pick_device
    device = pick_device()
    model, meta = model_repo.load(Path(args.model), device)
    return model, meta, device


def main():
    args = parse()
    sid = args.sid or ("FAKE" if args.fake else None)
    if sid is None:
        raise SystemExit("--sid 를 지정하세요 (예: --sid S01)")
    name = f"sess{args.round:02d}"
    order = task_order(args.round, args.seed)
    dur = ROUNDS[args.round][1]
    steps = build_steps(order, dur)
    total = sum(s.n for s in steps) / FS
    print(f"[{sid} {name}] 과제 순서 {' → '.join(order)}  (seed {args.seed}, 약 {total / 60:.1f}분)")

    if args.fake:
        stream = FakeStream(order, dur, seed=args.seed + 7, speed=args.speed)
    else:
        stream = SerialStream(args.port, args.baud)
        input("센서를 붙이고 참가자가 똑바로 서면 Enter ▶ ")

    out = Path(args.out) / sid
    out.mkdir(parents=True, exist_ok=True)
    imu_path, task_path = out / f"{name}_imu.csv", out / f"{name}_tasks.csv"
    f = open(imu_path, "w")
    f.write(f"# sid={sid} round={args.round} seed={args.seed} order={','.join(order)} "
            f"source={'fake' if args.fake else args.port}\n")
    f.write(",".join(COLUMNS) + "\n")

    rows, tasks = [], []
    calib_seg = {}
    k, n_in, start_us, start_i = 0, 0, None, 0
    gaps, last_seq = 0, None
    watch = PROFILES[args.profile]
    monitor = None
    bundle = load_bundle(args) if args.live else None

    try:
        for row in stream:
            if last_seq is not None and row[0] != last_seq + 1:
                gaps += row[0] - last_seq - 1          # 펌웨어가 못 보낸 샘플 수
            last_seq = row[0]
            f.write(",".join(map(str, row)) + "\n")
            rows.append(row)

            step = steps[k]
            if n_in == 0:                              # 새 과제 시작
                start_us, start_i = row[1], len(rows) - 1
                print(f"\n▶ [{k + 1}/{len(steps)}] {step.task_id:7s} {step.n // FS:3d}초  {step.text}")
            n_in += 1
            left = (step.n - n_in) // FS
            if n_in % FS == 0 and 0 < left <= 3:
                print(f"   {left}…")

            if monitor is not None and monitor.feed(row):
                text, src = compose(watch, use_llm=args.llm)
                print(f"\n   🔔 [{src}] {text}")
            if monitor is not None and n_in % FS == 0 and monitor.last:
                v, act, state = monitor.last
                print(f"   값 {v:6.2f}  {act:10s} {state}", end="\r")

            if n_in == step.n:                         # 과제 끝
                tasks.append((start_us, row[1] + 1_000_000 // FS, step.task_id))   # 끝 = 다음 샘플 시각
                if step.task_id in CALIB_TASKS:
                    calib_seg[step.task_id] = (start_i, len(rows))
                    if step.task_id == CALIB_TASKS[-1]:
                        cal = calibrate_rows(rows, calib_seg)
                        print(f"\n✔ 보정 완료: 크로스토크 {cal.crosstalk:.2f}, 앉은 중립 {cal.sit_offset:.1f}°, "
                              f"개인 굴곡 범위 {cal.slump_ref - cal.sit_offset:.1f}°")
                        if args.live:
                            monitor = LiveMonitor(cal, args.engine, watch, bundle)
                            print(f"  실시간 판정 시작: {args.engine} / {args.profile}")
                k, n_in = k + 1, 0
                if k == len(steps):
                    break
    except KeyboardInterrupt:
        print("\n중단됨. 끝난 과제까지만 저장합니다.")
    finally:
        f.close()
        stream.close()
        pd.DataFrame([(s, e, t, args.round, "ok") for s, e, t in tasks],
                     columns=["start_us", "end_us", "task_id", "round", "quality"]).to_csv(task_path, index=False)

    print(f"\n저장: {imu_path} ({len(rows)}행, 빠진 샘플 {gaps}개)\n      {task_path} ({len(tasks)}구간)")
    check(out.parent, sid, name, stream, args.fake, len(tasks) == len(steps))


def check(root, sid, name, stream, fake, complete):
    """방금 저장한 CSV를 오프라인 파이프라인으로 다시 읽어 본다 = 실제 데이터 경로의 통합 테스트"""
    if not complete:
        return
    session = load_csv_session(root, sid, name)
    print("\n=== 저장한 파일로 오프라인 재생 (v0b) ===")
    for profile, watch in PROFILES.items():
        alerts, seg = replay(session, "v0b", watch)
        r = score(alerts, seg, watch)
        print(f"  {profile:22s} TP {r['TP']}  FN {r['FN']}  FA {r['FA']} {r['FA_tasks'] or ''}")
    if fake:   # 가짜 모드: 기록한 구간이 시뮬레이터의 정답 구간과 맞는지
        mine = pd.read_csv(Path(root) / sid / f"{name}_tasks.csv")
        diff = (mine[["start_us", "end_us"]].to_numpy() - stream.truth[["start_us", "end_us"]].to_numpy())
        print(f"  라벨 구간 오차: 최대 {abs(diff).max() / 1000:.0f} ms (시뮬레이터 정답 대비)")


if __name__ == "__main__":
    main()
