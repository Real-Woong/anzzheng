# 단위 변환, 상보 필터, 3단 보정, 모델 입력 채널
# 배울 개념: 개인차(전만·연부조직·부착 위치)는 모델이 외우게 하지 말고, 보정으로 먼저 지운다.
import numpy as np

from ml.config import FS, DEFAULT_SLUMP_RANGE, MIN_SLUMP_RANGE
from ml.domain.types import Calibration

ACC_LSB = 8192.0
GYR_LSB = 65.5


def to_physical(df, pre):
    """pre = 'u' 또는 'l'. LSB → (acc[g], gyr[°/s]) numpy 배열."""
    acc = df[[f"{pre}_ax", f"{pre}_ay", f"{pre}_az"]].to_numpy() / ACC_LSB
    gyr = df[[f"{pre}_gx", f"{pre}_gy", f"{pre}_gz"]].to_numpy() / GYR_LSB
    return acc, gyr


def accel_tilt(acc):
    """가속도만으로 구한 시상면 기울기(°). 정지 상태에서만 정확."""
    return np.rad2deg(np.arctan2(acc[:, 2], acc[:, 0]))


def complementary(acc, gyr, gyro_bias=0.0, alpha=0.98):
    """상보 필터: 자이로(단기) + 가속도(장기)."""
    dt = 1.0 / FS
    a_tilt = accel_tilt(acc)
    out = np.empty(len(acc))
    out[0] = a_tilt[0]
    for i in range(1, len(acc)):
        gyro_pred = out[i - 1] + (gyr[i, 1] - gyro_bias) * dt
        out[i] = alpha * gyro_pred + (1 - alpha) * a_tilt[i]
    return out


def calibrate(df, seg):
    """seg = {"C_STAND": (i0, i1), "C_HINGE": ..., "C_SIT": ..., "C_SLUMP"(선택): ...}"""
    acc_u, gyr_u = to_physical(df, "u")
    acc_l, gyr_l = to_physical(df, "l")
    st = slice(*seg["C_STAND"])
    u_bias, l_bias = gyr_u[st, 1].mean(), gyr_l[st, 1].mean()
    u = complementary(acc_u, gyr_u, u_bias)
    l = complementary(acc_l, gyr_l, l_bias)

    # 1단: 서기 → 부착 각도 차이
    pelvis_stand, trunk_stand = l[st].mean(), u[st].mean()
    stand_offset = (u[st] - l[st]).mean()

    # 2단: 힙 힌지 → 실제 요추 굴곡 ≈ 0이므로, 굴곡각이 골반을 따라 오르는 기울기 = crosstalk
    hg = slice(*seg["C_HINGE"])
    crosstalk = np.polyfit(l[hg] - pelvis_stand, u[hg] - l[hg] - stand_offset, 1)[0]

    # 3단: 앉기(+선택 구부정 앉기) → 앉은 상태의 중립과 개인 굴곡 범위
    lumbar_c = (u - l) - stand_offset - crosstalk * (l - pelvis_stand)
    sit_offset = lumbar_c[slice(*seg["C_SIT"])].mean()
    if "C_SLUMP" in seg:
        slump_ref = lumbar_c[slice(*seg["C_SLUMP"])].mean()
    else:
        slump_ref = sit_offset + DEFAULT_SLUMP_RANGE
    slump_ref = max(slump_ref, sit_offset + MIN_SLUMP_RANGE)

    acc_noise = max(np.linalg.norm(acc_u[st], axis=1).std(), 1e-3)
    gyr_noise = max((gyr_u[st, 1] - u_bias).std(), 0.05)

    return Calibration(*(float(v) for v in (
        u_bias, l_bias, pelvis_stand, trunk_stand, stand_offset,
        crosstalk, sit_offset, slump_ref, acc_noise, gyr_noise)))


def derive_channels(df, cal):
    """원시 IMU + 보정 → 샘플별 채널 dict. 값은 모두 (N,) 배열."""
    acc_u, gyr_u = to_physical(df, "u")
    acc_l, gyr_l = to_physical(df, "l")
    u = complementary(acc_u, gyr_u, cal.u_bias)
    l = complementary(acc_l, gyr_l, cal.l_bias)
    pelvis_rel = l - cal.pelvis_stand
    lumbar_c = (u - l) - cal.stand_offset - cal.crosstalk * pelvis_rel
    return {
        "pelvis_rel": pelvis_rel,
        "lumbar_c": lumbar_c,
        "lumbar_norm": (lumbar_c - cal.sit_offset) / (cal.slump_ref - cal.sit_offset),
        "trunk_rel": u - cal.trunk_stand,
        "u_gyr_n": (gyr_u[:, 1] - cal.u_bias) / cal.gyr_noise,
        "l_gyr_n": (gyr_l[:, 1] - cal.l_bias) / cal.gyr_noise,
        "u_acc_n": (np.linalg.norm(acc_u, axis=1) - 1.0) / cal.acc_noise,
        "l_acc_n": (np.linalg.norm(acc_l, axis=1) - 1.0) / cal.acc_noise,
        "legacy": (u - l) - cal.stand_offset,
    }


# ======================================================================
# 변수·함수 설명
# ======================================================================
#
# [측정 모델]  사람마다 다른 것을 식 하나로 묶으면
#   측정 굴곡각 = gain × 실제 굴곡 + crosstalk × 골반 기울기 + 부착 offset
#   offset    ← 서기 보정 (stand_offset)
#   crosstalk ← 힙 힌지 보정 (실제 굴곡≈0인데 굴곡각이 오르는 만큼)
#   gain      ← 직접 못 구함. 대신 "개인 범위 대비 비율"(lumbar_norm)로 나누면 상쇄된다.
#
# [calibrate(df, seg)]
#   seg          보정 구간의 샘플 인덱스 (시작, 끝). 과제 안내 스크립트가 기록한다.
#   u_bias/l_bias  서기 중 자이로 평균 = 바이어스
#   pelvis_stand 서기 중 하부(S1) 기울기. 이후 골반 기울기는 이 값 기준.
#   trunk_stand  서기 중 상부(T12/L1) 기울기. 상체 숙임 기준.
#   stand_offset 서기 중 (상부 − 하부). 두 센서를 붙인 각도 차이.
#   crosstalk    힙 힌지 구간에서 x=골반 기울기, y=굴곡각으로 직선을 맞춘 기울기.
#                np.polyfit(x, y, 1)[0] = 1차 직선의 기울기.
#   sit_offset   바르게 앉기 중 보정 굴곡각. 앉으면 전만이 줄어 0이 아니다.
#   slump_ref    편하게 구부정하게 앉기 중 보정 굴곡각 = 개인 굴곡 기준점.
#                최대 굴곡이 아니다 (굴곡 불내성 환자에게 최대 굴곡을 시키지 않는다).
#   acc_noise    서기 중 가속도 크기의 표준편차 = 그 사람·그 부착의 노이즈 바닥
#   gyr_noise    서기 중 자이로(시상면) 표준편차
#
# [derive_channels(df, cal)]  모델과 규칙 엔진이 쓰는 채널
#   pelvis_rel   골반 기울기 (서기 기준, °). 앉기·숙이기 맥락을 알려 준다.
#   lumbar_c     crosstalk까지 지운 굴곡각 (서기 기준, °)
#   lumbar_norm  0 = 바르게 앉기, 1 = 개인 구부정 기준. gain이 상쇄된 개인 정규화 값.
#   trunk_rel    상체 기울기 (서기 기준, °). "상체 숙임 주의" 프로필용.
#   *_gyr_n      자이로를 노이즈 바닥으로 나눈 값 (사람 간 흔들림 차이 상쇄)
#   *_acc_n      (가속도 크기 − 1 g) / 노이즈 바닥. 걷기·전환에서 커진다.
#   legacy       서기 보정만 한 기존 굴곡각. 비교군(v0b_legacy)용.
