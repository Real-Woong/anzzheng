# Component 계층: 신경망 구조만 정의한다 (학습 방법은 services/trainer.py)
# 배울 개념: nn.Module = "층을 조립한 부품". __init__에서 층을 만들고 forward에서 데이터가 흐르는 길을 쓴다.
import torch
from torch import nn


def _block(c_in, c_out):
    return nn.Sequential(
        nn.Conv1d(c_in, c_out, kernel_size=5, padding=2),
        nn.BatchNorm1d(c_out),
        nn.ReLU(),
    )


class PostureCNN(nn.Module):
    def __init__(self, in_ch, n_act=3, n_post=2):
        super().__init__()
        self.backbone = nn.Sequential(
            _block(in_ch, 16), nn.MaxPool1d(2),
            _block(16, 32), nn.MaxPool1d(2),
            _block(32, 32),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
        )
        self.act_head = nn.Linear(32, n_act)    # 활동: static / moving / transition
        self.post_head = nn.Linear(32, n_post)  # 자세 요소: lumbar_flexion, trunk_lean

    def forward(self, x):
        z = self.backbone(x)
        return self.act_head(z), self.post_head(z)


# ======================================================================
# 설명
# ======================================================================
#
# [텐서 모양]  B = 배치 크기, C = 채널 8개, T = 100 샘플(2초)
#   입력 x              (B, 8, 100)
#   _block(8→16)        (B, 16, 100)   Conv1d: 시간축을 따라 길이 5짜리 필터를 미끄러뜨린다
#   MaxPool1d(2)        (B, 16, 50)    시간축을 절반으로 줄인다
#   _block(16→32)+Pool  (B, 32, 25)
#   _block(32→32)       (B, 32, 25)
#   AdaptiveAvgPool1d   (B, 32, 1)     시간축 평균 → 윈도우 전체 요약
#   Flatten             (B, 32)
#   act_head            (B, 3)  logit. softmax를 거치면 활동 확률.
#   post_head           (B, 2)  logit. sigmoid를 거치면 자세 요소별 확률 (멀티라벨).
#
# [멀티헤드]  몸통(backbone) 하나를 두 헤드가 공유한다. 활동과 자세는 같은 신호에서 나온다.
# [BatchNorm1d]  층마다 값의 분포를 정규화해 학습을 안정시킨다. 학습/추론 동작이 달라서
#                추론 때는 반드시 model.eval()을 호출한다.
# [padding=2]  kernel 5의 양옆을 2칸씩 채워 길이를 유지한다.
# 파라미터 수: 약 9천 개 (ML 문서 3.2의 "약 1만 개" 목표)
