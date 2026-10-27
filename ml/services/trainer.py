# Service 계층: PyTorch 학습 루프
# 배울 개념: device 선택, DataLoader, 순전파 → 손실 → 역전파 → optimizer.step, 마스킹 손실, AMP.
from contextlib import nullcontext
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ml.config import TRAIN, CHANNELS
from ml.models.posture_cnn import PostureCNN


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")      # NVIDIA GPU
    if torch.backends.mps.is_available():
        return torch.device("mps")       # Apple Silicon GPU
    return torch.device("cpu")


def fit_norm(X):
    """채널별 평균·표준편차. 학습 세트에서만 계산해야 테스트 정보가 새지 않는다."""
    return X.mean(axis=(0, 2)), X.std(axis=(0, 2)) + 1e-6


def apply_norm(X, mean, std):
    return (X - np.asarray(mean)[None, :, None]) / np.asarray(std)[None, :, None]


def masked_bce(logits, target):
    """target = -1인 칸(이동·전환 중 자세 라벨)은 손실에서 뺀다."""
    mask = target >= 0
    if not mask.any():
        return logits.sum() * 0.0
    return nn.functional.binary_cross_entropy_with_logits(logits[mask], target[mask].float())


def class_weights(y, n):
    """빈도의 역수. 드문 클래스(transition)의 손실을 키운다."""
    counts = np.bincount(y, minlength=n).astype(np.float32)
    w = counts.sum() / np.maximum(counts, 1) / n
    return torch.tensor(w)


def train(X, y_act, y_post, device, cfg=TRAIN, log=print, init=None, norm=None, balance=False):
    """init: 시작 가중치(state_dict, 사전학습). norm: (mean, std)를 고정해서 쓸 때."""
    torch.manual_seed(cfg["seed"])
    mean, std = norm if norm is not None else fit_norm(X)
    ds = TensorDataset(
        torch.from_numpy(apply_norm(X, mean, std).astype(np.float32)),
        torch.from_numpy(y_act).long(),
        torch.from_numpy(y_post).long(),
    )
    loader = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True,
                        pin_memory=device.type == "cuda")

    model = PostureCNN(len(CHANNELS))
    if init is not None:
        model.load_state_dict(init)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    weight = class_weights(y_act, 3).to(device) if balance else None
    ce = nn.CrossEntropyLoss(weight=weight)

    for epoch in range(cfg["epochs"]):
        model.train()
        total = 0.0
        for xb, ab, pb in loader:
            xb, ab, pb = xb.to(device), ab.to(device), pb.to(device)
            amp = torch.autocast("cuda", dtype=torch.float16) if use_amp else nullcontext()
            with amp:
                act_logit, post_logit = model(xb)
                loss = ce(act_logit, ab) + masked_bce(post_logit, pb)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += loss.item() * len(xb)
        sched.step()
        if (epoch + 1) % 5 == 0 or epoch == 0:
            log(f"  epoch {epoch + 1:3d}  loss {total / len(ds):.4f}")

    meta = {"channels": list(CHANNELS), "mean": np.asarray(mean).tolist(),
            "std": np.asarray(std).tolist()}
    return model.eval(), meta


@torch.no_grad()
def predict(model, meta, X, device):
    """X (n, C, WIN) → (활동 확률 (n,3), 자세 확률 (n,2)) numpy"""
    xb = torch.from_numpy(apply_norm(X, meta["mean"], meta["std"]).astype(np.float32)).to(device)
    act, post = model(xb)
    return act.softmax(-1).cpu().numpy(), post.sigmoid().cpu().numpy()


# ======================================================================
# 설명
# ======================================================================
#
# [학습 한 스텝]  모든 PyTorch 학습은 이 5줄의 반복이다
#   1. act_logit, post_logit = model(xb)   순전파
#   2. loss = ...                          정답과 얼마나 다른가
#   3. opt.zero_grad()                     지난 스텝의 기울기 지우기 (안 지우면 누적된다)
#   4. loss.backward()                     역전파: 모든 가중치의 기울기 계산
#   5. opt.step()                          기울기 방향으로 가중치 갱신
#
# [device]  텐서와 모델을 같은 장치에 올려야 한다 (.to(device)). 섞이면 에러.
#   cuda  NVIDIA GPU. pin_memory=True로 CPU→GPU 복사를 빠르게 한다.
#   mps   Mac GPU. 이 정도 작은 모델은 CPU와 속도 차이가 크지 않다.
#
# [AMP (자동 혼합 정밀도)]  CUDA에서만 켠다.
#   autocast   행렬곱 등을 float16으로 계산해 빠르고 메모리를 덜 쓴다.
#   GradScaler float16은 작은 기울기가 0으로 사라지기 쉬워, 손실을 크게 키웠다가 되돌린다.
#              enabled=False면 아무것도 안 하는 통과 장치가 되어 CPU/MPS에서도 같은 코드가 돈다.
#
# [손실]
#   CrossEntropyLoss   활동 3클래스 (softmax 포함)
#   masked_bce         자세 2개 멀티라벨. 이동·전환 윈도우(-1)는 빼고 계산.
#
# [전이학습 옵션]
#   init     다른 데이터(HARTH)로 학습한 가중치에서 출발한다. 처음부터(랜덤) 시작하지 않는다.
#   norm     정규화 통계를 고정한다. 사전학습과 미세조정이 같은 스케일을 봐야 가중치가 통한다.
#   balance  클래스 빈도의 역수로 손실 가중치. HARTH는 앉기가 대부분이고 전환은 아주 드물다.
#
# [AdamW + CosineAnnealingLR]  가장 무난한 기본 조합. 학습률을 코사인 곡선으로 서서히 줄인다.
# [@torch.no_grad()]  추론 때는 기울기 기록을 끈다 → 빠르고 메모리 절약.
