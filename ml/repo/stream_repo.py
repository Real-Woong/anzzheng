# Repository 계층: 실시간 IMU 스트림. 진짜 ESP32(시리얼)와 가짜(MuJoCo 시뮬레이터)가 같은 모양을 낸다.
# 위 계층(record.py)은 어느 쪽인지 모른다. 하드웨어가 오면 --fake만 --port로 바꾸면 된다.
import time
import numpy as np

COLUMNS = ("seq", "t_us", "u_ax", "u_ay", "u_az", "u_gx", "u_gy", "u_gz",
           "l_ax", "l_ay", "l_az", "l_gx", "l_gy", "l_gz")   # 펌웨어 출력 순서와 같다


class SerialStream:
    """ESP32 펌웨어(firmware/anzzheng_logger)가 보내는 CSV 줄을 읽는다."""

    def __init__(self, port, baud=115200, on_message=print):
        import serial                      # pyserial. 가짜 모드에서는 필요 없어서 여기서 불러온다
        self.ser = serial.Serial(port, baud, timeout=1)
        self.ser.reset_input_buffer()      # 연결 전에 쌓인 반쪽짜리 줄 버리기
        self.on_message = on_message

    def __iter__(self):
        """한 샘플씩 (14개 int 튜플)을 낸다. '#' 줄은 on_message로만 보낸다."""
        while True:
            line = self.ser.readline().decode("ascii", errors="ignore").strip()
            if not line:
                continue
            if line.startswith("#"):
                self.on_message(f"[ESP32] {line[1:].strip()}")
                continue
            parts = line.split(",")
            if len(parts) != len(COLUMNS):
                continue                   # 깨진 줄 (연결 직후 등)
            try:
                yield tuple(int(p) for p in parts)
            except ValueError:
                continue

    def close(self):
        self.ser.close()


class FakeStream:
    """MuJoCo 가상 참가자가 프로토콜을 그대로 따라 한다고 치고, 그 IMU 값을 실시간 속도로 흘려보낸다."""

    def __init__(self, order, dur_s, seed=7, speed=1.0, fs=50):
        from ml.sim.mujoco_sim import make_session, random_mj_body
        rng = np.random.default_rng(seed)
        self.body = random_mj_body(rng)
        df, self.truth = make_session(self.body, order, rng, dur_s=dur_s)   # truth = 시뮬레이터가 아는 정답 구간
        self.rows = df[list(COLUMNS)].to_numpy()
        self.period = 1 / (fs * speed)

    def __iter__(self):
        t0 = time.perf_counter()
        for i, r in enumerate(self.rows):
            wait = t0 + i * self.period - time.perf_counter()
            if wait > 0:
                time.sleep(wait)           # 진짜 센서처럼 1/50초마다 한 줄
            yield tuple(int(v) for v in r)

    def close(self):
        pass
