# Service 계층: 알림 문구. 로컬 Llama(ollama)가 있으면 문구를 다양하게, 없으면 템플릿.
# 배울 개념: LLM 출력은 그대로 믿지 않는다. 길이·금지어를 검사하고 실패하면 템플릿으로 돌아간다.
import json
import urllib.request

from ml.config import OLLAMA_URL, OLLAMA_MODEL

BANNED = ("진단", "치료", "재활", "처방", "완치", "질환", "디스크")

TEMPLATES = {
    "lumbar_flexion": "허리가 {s}초 넘게 말려 있어요. 허리를 펴고 고관절로 숙여 보세요.",
    "trunk_lean": "상체가 {s}초 넘게 앞으로 기울어 있어요. 잠깐 바르게 앉아 볼까요?",
}

PROMPT = """너는 자세 습관 알림 앱의 문구 작성기다.
상황: 사용자가 '{what}' 자세를 {s}초 넘게 유지했다.
규칙: 한국어 한 문장, 50자 이내, 부드러운 존댓말, 바꿀 동작 하나를 제안.
의학 용어(진단, 치료, 재활, 질환 등)와 이모지는 쓰지 않는다. 문장만 출력한다."""

WHAT = {"lumbar_flexion": "허리를 말아 굽힌", "trunk_lean": "상체를 앞으로 숙인"}


def _ask_llama(watch, held_s, timeout=5.0):
    body = json.dumps({
        "model": OLLAMA_MODEL,
        "prompt": PROMPT.format(what=WHAT[watch], s=held_s),
        "stream": False,
        "options": {"temperature": 0.7, "num_predict": 80},
    }).encode()
    req = urllib.request.Request(OLLAMA_URL, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())["response"].strip().strip('"')


def _valid(text):
    return 0 < len(text) <= 60 and "\n" not in text and not any(w in text for w in BANNED)


def compose(watch, held_s=5, use_llm=False):
    """반환: (문구, 출처 'llama' | 'template')"""
    if use_llm:
        try:
            text = _ask_llama(watch, held_s)
            if _valid(text):
                return text, "llama"
        except (OSError, ValueError, KeyError):
            pass
    return TEMPLATES[watch].format(s=held_s), "template"


# ======================================================================
# 설명
# ======================================================================
#
# [왜 분류에 Llama를 안 쓰나]  IMU 숫자 시계열은 LLM이 잘하는 입력이 아니다. 느리고(수백 ms~초),
#   정확도도 1D-CNN보다 낮다. LLM은 "말"을 만드는 곳(알림 문구)에만 쓴다.
# [준비]  터미널에서  ollama pull llama3.2  →  ollama serve (앱이 켜져 있으면 생략)
# [_ask_llama]  ollama HTTP API(/api/generate)에 POST. stream=False면 한 번에 응답을 받는다.
#   temperature  0.7 → 매번 조금씩 다른 문구
#   num_predict  최대 생성 토큰 수
# [_valid]  길이, 줄바꿈, 금지어(의료 표현 가이드) 검사. 하나라도 걸리면 템플릿 사용.
# [timeout]  알림이 LLM 때문에 늦어지면 안 된다. 5초 안에 못 받으면 템플릿.
