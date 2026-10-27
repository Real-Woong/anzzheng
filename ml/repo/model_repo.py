# Repository 계층: 학습된 모델 저장/로드
# 모델 가중치만 저장하면 안 된다. 정규화 통계·채널 목록이 다르면 같은 가중치도 엉뚱한 답을 낸다.
import torch

from ml.models.posture_cnn import PostureCNN


def save(path, model, meta):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "meta": meta}, path)


def load(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    meta = ckpt["meta"]
    model = PostureCNN(len(meta["channels"]))
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), meta
