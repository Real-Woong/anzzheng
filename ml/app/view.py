# Controller 계층: MuJoCo 시뮬레이션을 화면으로 본다 (옆에서 본 시상면, 빨간 상자 = IMU)
# Mac:   .venv/bin/mjpython -m ml.app.view --tasks T02,T04,T06
#        (Mac에서 창을 띄우려면 python 대신 mjpython으로 실행해야 한다)
# 동영상: python -m ml.app.view --tasks T02,T04,T06 --video artifacts/sim.mp4
import argparse
import time
import numpy as np
import mujoco

from ml.config import FS
from ml.sim.mujoco_sim import TASKS, build_model, build_qpos, motion, random_mj_body


def side_camera(m):
    """골반(sacrum)을 따라가며 옆에서 보는 카메라. 걷기 때 몸이 앞으로 가도 화면 밖으로 안 나간다."""
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = m.body("sacrum").id
    cam.distance, cam.azimuth, cam.elevation = 1.6, 0.0, -10.0
    return cam


def task_at(tasks):
    """프레임 번호 → 과제 ID"""
    out = np.full(int(tasks.end_us.max() * FS // 1_000_000) + 1, "", dtype=object)
    for r in tasks.itertuples():
        out[r.start_us * FS // 1_000_000: r.end_us * FS // 1_000_000] = r.task_id
    return out


def play_window(m, qpos, labels, speed):
    import mujoco.viewer
    d = mujoco.MjData(m)
    with mujoco.viewer.launch_passive(m, d) as v:
        cam = side_camera(m)
        v.cam.type, v.cam.trackbodyid = cam.type, cam.trackbodyid
        v.cam.distance, v.cam.azimuth, v.cam.elevation = cam.distance, cam.azimuth, cam.elevation
        last = None
        for i in range(len(qpos)):
            if not v.is_running():
                break
            t0 = time.time()
            d.qpos[:] = qpos[i]
            mujoco.mj_forward(m, d)          # 위치만 바꿨으니 운동학만 다시 계산해서 그린다
            v.sync()
            if labels[i] != last:
                last = labels[i]
                print(f"{i / FS:6.1f}s  {last}  {TASKS[last][0] + ' ' + TASKS[last][3] if last in TASKS else ''}")
            time.sleep(max(0.0, 1 / (FS * speed) - (time.time() - t0)))


def save_video(m, qpos, path, speed, step=2):
    import imageio
    d = mujoco.MjData(m)
    cam = side_camera(m)
    with mujoco.Renderer(m, height=480, width=640) as r:
        frames = []
        for i in range(0, len(qpos), step):
            d.qpos[:] = qpos[i]
            mujoco.mj_forward(m, d)
            r.update_scene(d, camera=cam)
            frames.append(r.render())
    imageio.mimsave(path, frames, fps=int(FS / step * speed))
    print(f"saved: {path}  ({len(frames)} frames)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=99, help="가상 참가자 (같은 seed = 같은 체형)")
    ap.add_argument("--tasks", default="T01,T02,T04,T06,T08,T09", help="쉼표로 구분한 과제 ID")
    ap.add_argument("--dur", type=int, default=8, help="과제당 초")
    ap.add_argument("--speed", type=float, default=1.0, help="재생 배속")
    ap.add_argument("--video", help="창 대신 mp4로 저장할 경로")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    body = random_mj_body(rng)
    print(f"body: 척추길이 x{body.spine_scale:.2f}, 연부조직 {body.fat_cm:.1f}cm, "
          f"분담 L5/S1…L1/L2 {np.round(body.rhythm, 2).tolist()}")
    traj, tasks = motion(body, args.tasks.split(","), rng, dur_s=args.dur)
    m = build_model(body)
    qpos = build_qpos(m, body, traj, rng)
    labels = task_at(tasks)

    if args.video:
        save_video(m, qpos, args.video, args.speed)
    else:
        play_window(m, qpos, labels, args.speed)


if __name__ == "__main__":
    main()
