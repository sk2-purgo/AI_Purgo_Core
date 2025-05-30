from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import BertModel, AutoTokenizer
import torch.nn as nn
import os
import re
from typing import List
from asyncio import to_thread
import logging
from dotenv import load_dotenv
from openai import OpenAI

# ✅ 로그 레벨 낮추기 (transformers 불필요한 경고 숨기기)
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

# ✅ 환경변수 로드
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ✅ 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ KoBERT 모델 정의
class CussClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained("monologg/kobert", trust_remote_code=True)
        self.classifier = nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(outputs.pooler_output)

# ✅ KoBERT 모델 로딩
model = CussClassifier().to(device)
model_path = os.path.join("purgo_kobert", "model", "purgo.pth")
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ✅ KoBERT 토크나이저 로딩
kobert_tokenizer = AutoTokenizer.from_pretrained("monologg/kobert", trust_remote_code=True)

# ✅ 욕설 단어 사전 로딩
def load_badwords_from_txt():
    path = os.path.join("purgo_kobert", "app", "befasttext_filter", "befasttext_cuss_train_full.txt")
    badwords = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            word = line.strip().split(",")[0]
            if word:
                badwords.add(word)
    return badwords

badword_set = load_badwords_from_txt()  # ✅ 반드시 활성화

# ✅ 텍스트 전처리 (특수문자 제거)
def clean_text(text):
    text = re.sub(r"[^\u4E00-\u9FFF\u3400-\u4DBF가-힣ㄱ-ㅎㅏ-ㅣ0-9\s]", "", text)
    return text

# ✅ FastText 감지
def detect_fasttext(text: str, badwords: set) -> List[str]:
    cleaned_text = clean_text(text)
    return [word for word in badwords if word in cleaned_text]

# ✅ KoBERT 감지
def detect_kobert(text):
    inputs = kobert_tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=64)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        probs = torch.softmax(outputs, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        conf = probs[0][pred].item()
    return pred, round(conf, 4)

# ✅ GPT-3.5 순화
def rewrite_text_gpt3_5(text):
    prompt = f"""
당신은 사람들의 발화를 정중하고 긍정적인 표현으로 순화하는 AI입니다.

다음 문장을 아래 조건에 맞춰 순화해주세요:

[💬 순화 조건]
1. 문장에 욕설, 비속어, 혐오, 성적인 표현이 있을 경우 → 문맥을 고려하여 **정중하고 바른 표현**으로 자연스럽게 바꾸세요.
2. 노골적이지 않아도 공격적이거나 부정적인 뉘앙스를 가진 단어는 **긍정적이고 포용적인 표현**으로 순화하세요.
3. **문장의 구조와 말투는 유지**하면서, 문제가 되는 단어만 바꾸는 것이 핵심입니다.
4. 예의 바르고 부드러운 어투로 작성하세요.
5. **욕설이 없는 경우에는 문장을 수정하지 않고 그대로 반환**하세요.
6. 출력은 반드시 **정제된 문장 한 줄만**, 설명이나 따옴표 없이 출력하세요.

[🔴 반드시 순화해야 할 표현 예시]
- 초성 욕설 (예: ㅅㅂ, ㅈㄹ, ㅄ 등)
- 감정 과격 표현 (예: 존나, 개같은, 지랄 등)
- 인신 공격 표현 (예: 미친놈, 병신, 새끼, 븅신 등)

[🧠 순화 예시]
- "씨발 오늘 왜 이래" → "아 진짜 오늘 왜 이래"
- "개같은 새끼" → "정말 못된 사람"
- "존나 짜증나" → "정말 짜증나"
- "지랄하지 마" → "화내지 마"
- "병신같이 하네" → "어수룩하게 하네"
- "ㅅㅂ" → "아 진짜"
- "저새끼 뭐야" → "저 사람 뭐지"

문장: "{text}"

"""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "너는 문장을 정제하는 순화 전용 편집기야. 설명 없이 바른 말로 바꿔줘."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=100,
            top_p=0.9,
        )
        result = response.choices[0].message.content.strip()
        if result.startswith('"') and result.endswith('"'):
            result = result[1:-1]
        return result
    except Exception as e:
        print(f"❌ GPT 호출 실패: {e}")
        return text

# ✅ FastAPI 서버 시작
app = FastAPI()

class TextRequest(BaseModel):
    text: str

@app.get("/")
async def root():
    return {"message": "✅ FastText + KoBERT + GPT-3.5 비동기 서버 실행 중!"}

@app.post("/analyze")  # ❗ 앞에 "/" 필수
async def analyze(req: TextRequest):
    text = req.text.strip()
    fasttext_result = detect_fasttext(text, badword_set)
    fasttext_hit = 1 if fasttext_result else 0

    response = {
        "fasttext": {
            "is_bad": fasttext_hit,
            "detected_words": fasttext_result
        },
        "kobert": {
            "is_bad": None,
            "confidence": None
        },
        "result": {
            "original_text": text,
            "rewritten_text": text
        },
        "final_decision": 0
    }

    if fasttext_hit:
        rewritten = await to_thread(rewrite_text_gpt3_5, text)
        response["result"]["rewritten_text"] = rewritten  # 🔧
        response["final_decision"] = 1
    else:
        kobert_pred, kobert_conf = await to_thread(detect_kobert, text)
        kobert_hit = 1 if kobert_pred == 1 else 0
        response["kobert"]["is_bad"] = kobert_hit
        response["kobert"]["confidence"] = kobert_conf
        if kobert_hit:
            rewritten = await to_thread(rewrite_text_gpt3_5, text)
            response["result"]["rewritten_text"] = rewritten
            response["final_decision"] = 1

    return response

# uvicorn gpt_be_run:app --host 0.0.0.0 --port 5000 --reload
