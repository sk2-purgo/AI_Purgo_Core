# 💬 Purgo - AI 욕설 탐지 및 정화 시스템

FastAPI 기반 웹 서버에서 **FastText + KoBERT + GPT**를 활용하여  
댓글/채팅 등 텍스트 내 **욕설을 탐지하고 정제하는 AI 서비스**입니다.

---

## 👥 AI 팀원
| 이름 | 역할 |
|------|------|
| 이현영 | PM |
| 김태기 | PL |
| 백욱진 | 팀원 |
| 김소현 | 팀원 |
| 표상혁 | 팀원 |

---

## 🌿 주요 브랜치
- `main`: 최종 완성 브랜치

---

## 🛠 사용 기술 및 라이브러리

### ✅ 딥러닝 및 NLP
- PyTorch, transformers, gluonnlp
- KoBERT, KoGPT, sentencepiece, fasttext

### ✅ 서버 및 시각화
- FastAPI, Flask, uvicorn
- jinja2, matplotlib, pdfkit

### ✅ 데이터 처리 및 기타
- pandas, scikit-learn, tqdm, requests, os
- python-dotenv

---

## 🧠 욕설 탐지 파이프라인

```
1단계 FastText → 2단계 KoBERT → 3단계 GPT (직렬 조건부 구조)
```

- **FastText**: 단어 기반 필터링
- **KoBERT**: 문장/문맥 기반 욕설 감지
- **GPT**: 감지된 욕설을 정중하게 재작성

---

## 📁 프로젝트 디렉터리 구조



```
purgo_kobert/
├── app/
│   ├── bfasttext_filter/
│   │   └── befasttext_cuss_train_full.txt       ← FastText 단어 사전
│   ├── └── betrain_fasttext.py                  ← 사전 업데이트 
│   ├── befasttext_kobert/
│   ├── └── betrain_fasttext.py                  ← 테스트 코드 
├── data/
│   ├── 학습용 욕설 전체 데이터.csv                 ← 학습 데이터
│   └── 학습용 비욕설 전체 데이터.csv                ← 학습 데이터
├── model/
│   └── purgo.pth                                ← 학습 모델
├── run.py                                       ← Fast API 서버 실행
```

---

## ⚙️ 주요 라이브러리 설치 
# ✅ 웹 서버 관련 
pip install flask
pip install fastapi
pip install uvicorn

# ✅ AI 모델 (딥러닝) / NLP 관련
pip install torch torchvision torchaudio
pip install transformers==4.10.0
pip install gluonnlp==0.10.0
pip install sentencepiece
pip install kobert-tokenizer
pip install fasttext

# ✅ 데이터 처리 및 평가 
pip install pandas
pip install tqdm
pip install scikit-learn

# ✅ HTTP 통신
pip install requests

# ✅ 보고서 및 시각화
pip install jinja2
pip install matplotlib
pip install pdfkit
pip install tqdm

# pdf 변환  
📌 pdfkit 사용 시, 시스템에 wkhtmltopdf가 설치되어 있어야 합니다.
설치 링크: https://wkhtmltopdf.org/downloads.html
wkhtmltopdf

### 1단계 FastText
### 2단계 KoBERT
### 3단계 GPT


### KoBERT 관련
```bash
pip install torch torchvision torchaudio
pip install transformers==4.10.0
pip install gluonnlp==0.10.0
pip install sentencepiece pandas tqdm kobert-tokenizer scikit-learn
```

### GPT 관련
```bash
pip install transformers torch sentencepiece
pip install dotenv
pip install openai==0.28.1

```

### FastText 관련
```bash
pip install fasttext
```

### 보고서 및 시각화
```bash
pip install jinja2 matplotlib pandas pdfkit
```

---

## 🚀 실행 방법

### 1. KoBERT 모델 학습
```bash
python purgo_kobert/train.py
```

### 2. Fast API 서버 실행
```bash
uvicorn gpt_be_run:app --host 0.0.0.0 --port 5000 --reload
```

### 3. 자동화 테스트 실행
```bash
python purgo_kobert/app/befasttext_kobert/test.py
```

⚠️ `uvicorn gpt_be_run:app --host 0.0.0.0 --port 5000 --reload`로 서버를 먼저 실행한 후, 새 터미널에서 테스트를 진행하세요.
---

## 📌 모델 성능 참고 메모

| 모델 이름                             | 사용 가능 여부 | 비고     |
| ------------------------------------ | -------- | ------ |
| nlpai-lab/korean-paraphrase-t5-small | ❌        | 사용 불가  |
| paust/pko-t5-base                    | ⭕        | 성능 아쉬움 |
| beomi/KoParrot                       | ❌        | 사용 불가  |
| digit82/kobart-summarization         | ⭕        | 성능 미흡  |
| kluebert                             | ⭕        | 성능 미흡  |
| **KoGPT** (사용 중)                    | ⭕        | 성능 양호  |

---

## 📞 문의
이슈나 버그는 GitHub Issues 또는 Pull Request를 통해 알려주세요. 감사합니다! 🙇





