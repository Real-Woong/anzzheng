# Service 계층: HARTH(등 가속도계 1개) → 우리 모델 입력 형식(8채널 윈도우)
# 배울 개념: 다른 데이터셋을 쓰려면 "입력 모양"과 "라벨 정의"를 내 모델에 맞춰 번역해야 한다.
# 번역이 안 되는 채널(요추 굴곡)은 0으로 두고, 그 한계를 알고 쓴다.
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from ml.config import FS, WIN, CHANNELS, ACTIVITIES
from ml.repo.harth_repo import load_subject, LABEL_TO_ACT, STANDING, SITTING
from ml.signal.process import accel_tilt

HARTH_STRIDE = 50   # 데이터가 많아서 1초 간격이면 충분


def _lowpass(x, width=FS // 2):
    k = np.ones(width) / width
    return np.stack([np.convolve(x[:, i], k, "same") for i in range(x.shape[1])], 1)


def harth_channels(acc, label):
    """우리 calibrate/derive_channels 대신 HARTH용 근사. 기준은 그 사람의 서기 구간."""
    tilt = accel_tilt(_lowpass(acc))
    pseudo_gyr = np.gradient(tilt) * FS                 # 자이로가 없어 기울기 미분으로 대신
    mag = np.linalg.norm(acc, axis=1) - 1.0
    stand = label == STANDING
    if stand.sum() < FS * 10:                           # 서기가 거의 없으면 정지 전체로
        stand = np.isin(label, (STANDING, SITTING))
    tilt_rel = tilt - np.median(tilt[stand])
    acc_noise = max(np.percentile(np.abs(mag[stand]), 50), 1e-3)
    gyr_noise = max(np.percentile(np.abs(pseudo_gyr[stand]), 50), 0.05)
    zero = np.zeros_like(tilt)
    ch = {
        "pelvis_rel": tilt_rel, "trunk_rel": tilt_rel,  # 센서 1개라 골반·상체가 같은 값
        "lumbar_c": zero, "lumbar_norm": zero,          # 요추 굴곡은 측정 불가
        "u_gyr_n": pseudo_gyr / gyr_noise, "l_gyr_n": pseudo_gyr / gyr_noise,
        "u_acc_n": mag / acc_noise, "l_acc_n": mag / acc_noise,
    }
    return np.stack([ch[c] for c in CHANNELS]).astype(np.float32)


def window_labels(label, stride=HARTH_STRIDE):
    """윈도우 라벨: 한 활동만 있으면 그 활동, 서기와 앉기가 섞이면 transition, 그 외 버림."""
    lw = sliding_window_view(label, WIN)[::stride]
    lo, hi = lw.min(1), lw.max(1)
    y = np.full(len(lw), -1)
    for code, act in LABEL_TO_ACT.items():
        y[(lo == code) & (hi == code)] = ACTIVITIES.index(act)
    sit_stand = np.isin(lw, (STANDING, SITTING)).all(1) & (lo != hi)
    y[sit_stand] = ACTIVITIES.index("transition")
    return y


def harth_windows(sids, stride=HARTH_STRIDE, log=print):
    Xs, ys, groups = [], [], []
    for sid in sids:
        acc, label = load_subject(sid)
        data = harth_channels(acc, label)
        X = sliding_window_view(data, WIN, axis=1)[:, ::stride].transpose(1, 0, 2)
        y = window_labels(label, stride)
        ok = y >= 0
        Xs.append(np.ascontiguousarray(X[ok])); ys.append(y[ok]); groups += [sid] * int(ok.sum())
        log(f"  {sid}: {ok.sum():6d} windows  " +
            " ".join(f"{a}={int((y[ok] == i).sum())}" for i, a in enumerate(ACTIVITIES)))
    return np.concatenate(Xs), np.concatenate(ys), np.array(groups)


# ======================================================================
# 설명
# ======================================================================
#
# [채널 번역표]  우리 채널 ← HARTH에서 만든 값
#   pelvis_rel, trunk_rel ← 등 가속도 기울기 − 그 사람 서기 중앙값 (센서가 1개라 같은 값)
#   lumbar_c, lumbar_norm ← 0 (센서 2개의 차이라서 만들 수 없다)
#   *_gyr_n               ← 기울기의 시간 미분 / 노이즈 바닥 (유사 자이로)
#   *_acc_n               ← (가속도 크기 − 1 g) / 노이즈 바닥
#
# [_lowpass]  0.5초 이동평균. 가속도로 기울기를 구할 때 걸음 충격을 줄인다.
#             (우리 데이터는 상보 필터가 이 역할을 하지만 HARTH엔 자이로가 없다)
# [노이즈 바닥]  서기 구간 값의 중앙값. 일상생활 데이터라 표준편차보다 중앙값이 튼튼하다.
# [window_labels]
#   sliding_window_view(label, WIN)  모든 시작점의 길이 100짜리 조각을 복사 없이 만든다
#   lo == hi == code                 윈도우 안의 라벨이 전부 같으면 그 활동
#   서기·앉기만 있고 둘이 섞임      → transition (HARTH엔 전환 라벨이 없어서 만든다)
# [groups]  윈도우마다 참가자 ID. 참가자 단위로 학습/테스트를 나눌 때 쓴다.
