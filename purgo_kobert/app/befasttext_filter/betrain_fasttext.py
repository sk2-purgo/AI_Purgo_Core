# FastText 비학습 모델
import os
import pandas as pd
from datetime import datetime
import time  # ⏱️ 시간 측정용 추가

# 경로 설정
base_dir = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(base_dir, "test_input.csv")
output_path = os.path.join(base_dir, "test_results.csv")
fasttext_data_path = os.path.join(base_dir, "befasttext_cuss_train_full.txt")

# 로그 함수
def log(msg):
    now = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    print(f"{now} {msg}")

# 욕설 단어 사전 로딩
def load_badwords_from_csv():
    try:
        df = pd.read_csv(fasttext_data_path, encoding="utf-8-sig")
        if "label" not in df.columns or "text" not in df.columns:
            df.columns = ["text", "label"]
        badwords = df[df["label"] == 1]["text"].dropna().unique()
        return list(map(str.strip, badwords))
    except Exception as e:
        log(f"❌ 욕설 사전 로딩 중 오류 발생: {e}")
        return []

# 문장에서 욕설 포함 단어 추출
def detect_badwords_in_text(text, badwords):
    return [word for word in badwords if word in text]

# 전체 실행 로직
def filter_texts_from_file():
    log("⏳ FastText 비학습 방식 욕설 필터링 시작")
    start_time = time.time()  # 시작 시간

    badwords = load_badwords_from_csv()
    log(f"✅ 총 {len(badwords)}개의 욕설 단어 로드됨.")

    df = pd.read_csv(input_path, encoding="utf-8-sig")
    results = []

    for _, row in df.iterrows():
        text = row["text"]
        detected = detect_badwords_in_text(text, badwords)
        results.append({
            "문장": text,
            "단어욕설_감지_수": len(detected),
            "감지된_단어": ", ".join(detected),
            "욕설_여부": "욕설" if detected else "중립"
        })

    pd.DataFrame(results).to_csv(output_path, index=False, encoding="utf-8-sig")

    end_time = time.time()  # 종료 시간
    elapsed = round(end_time - start_time, 4)

    log("✅ 결과 저장 완료")
    log(f"⏱️ 총 처리 시간: {elapsed}초")

# 실행
if __name__ == "__main__":
    filter_texts_from_file()
