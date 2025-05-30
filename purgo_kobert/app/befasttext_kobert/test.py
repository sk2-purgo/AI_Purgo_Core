import pandas as pd
import requests
import time
from jinja2 import Template
import pdfkit
import os
from tqdm import tqdm  # ✅ 진행률 표시

# ✅ API URL 설정
API_URL = "http://127.0.0.1:5000/analyze"


# ✅ 파일 경로 설정
input_path = "전체.csv"
html_output_path = "test_result_report.html"
pdf_output_path = "test_result_report.pdf"
csv_output_path = "test_result_report.csv"

# ✅ CSV 파일 불러오기
df = pd.read_csv(input_path)

print("🚀 테스트 실행 중...")

results = []
start_time = time.time()

# ✅ tqdm으로 진행률 표시
for idx, row in tqdm(df.iterrows(), total=len(df), desc="진행 상황", ncols=100):
    text = row["text"]
    try:
        response = requests.post(API_URL, json={"text": text})
        if response.status_code == 200:
            data = response.json()
            results.append({
                "문장": data.get("result", {}).get("original_text", ""),
                "단어욕설_감지_수": len(data.get("fasttext", {}).get("detected_words", [])),
                "문맥욕설_감지": data.get("kobert", {}).get("is_bad", ""),
                "문맥_신뢰도": data.get("kobert", {}).get("confidence", 0),
                "욕설_여부": data.get("final_decision", "")
            })
        else:
            results.append({
                "문장": text,
                "단어욕설_감지_수": "요청 실패",
                "문맥욕설_감지": "요청 실패",
                "문맥_신뢰도": 0,
                "욕설_여부": "요청 실패"
            })
    except Exception as e:
        results.append({
            "문장": text,
            "단어욕설_감지_수": "에러",
            "문맥욕설_감지": "에러",
            "문맥_신뢰도": 0,
            "욕설_여부": "에러"
        })

end_time = time.time()
elapsed_time = end_time - start_time

# ✅ 결과 DataFrame 생성
result_df = pd.DataFrame(results)

# ✅ 통계 정보 계산
total = len(result_df)
abusive_count = pd.to_numeric(result_df["욕설_여부"], errors='coerce').fillna(0).astype(int).sum()
normal_count = total - abusive_count

confidence_values = pd.to_numeric(result_df["문맥_신뢰도"], errors='coerce')
average_confidence = round(confidence_values.mean(), 4)

# ✅ CSV 저장 (문장, 욕설_여부만)
csv_result_df = result_df[["문장", "욕설_여부"]]
csv_result_df.to_csv(csv_output_path, index=False, encoding="utf-8-sig")
print(f"✅ CSV 리포트 생성 완료 (문장 + 욕설 여부): {csv_output_path}")

# ✅ HTML 템플릿 작성
template_str = """
<html>
<head>
    <meta charset="UTF-8">
    <title>Purgo - 욕설 탐지 리포트</title>
    <style>
        body { font-family: Arial, sans-serif; }
        table { border-collapse: collapse; width: 100%; margin-top: 20px; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: center; }
        th { background-color: #f2f2f2; }
        .summary { margin-bottom: 20px; padding: 10px; background-color: #f9f9f9; border: 1px solid #ddd; }
    </style>
</head>
<body>
    <h2>Purgo - 욕설 탐지 리포트</h2>
    <div class="summary">
        <p>총 문장 수: {{ total }}</p>
        <p>욕설 감지 문장 수: {{ abusive_count }}</p>
        <p>정상 문장 수: {{ normal_count }}</p>
        <p>평균 문맥 신뢰도: {{ average_confidence }}</p>
        <p>총 처리 시간: {{ elapsed_time }} 초</p>
    </div>
    <table>
        <thead>
            <tr>
                <th>문장</th>
                <th>단어욕설 감지 수</th>
                <th>문맥욕설 감지</th>
                <th>문맥 신뢰도</th>
                <th>욕설 여부</th>
            </tr>
        </thead>
        <tbody>
        {% for row in rows %}
            <tr>
                <td>{{ row.문장 }}</td>
                <td>{{ row.단어욕설_감지_수 }}</td>
                <td>{{ row.문맥욕설_감지 }}</td>
                <td>{{ row.문맥_신뢰도 }}</td>
                <td>{{ row.욕설_여부 }}</td>
            </tr>
        {% endfor %}
        </tbody>
    </table>
</body>
</html>
"""

# ✅ HTML 렌더링 및 저장
template = Template(template_str)
html_content = template.render(
    total=total,
    abusive_count=abusive_count,
    normal_count=normal_count,
    average_confidence=average_confidence,
    elapsed_time=round(elapsed_time, 2),
    rows=result_df.to_dict(orient="records")
)

with open(html_output_path, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"✅ HTML 리포트 생성 완료: {html_output_path}")

# ✅ PDF 리포트 생성
config = pdfkit.configuration(
    wkhtmltopdf=r"C:\\Program Files\\wkhtmltopdf\\bin\\wkhtmltopdf.exe"
)
options = {
    'enable-local-file-access': None
}

pdfkit.from_file(html_output_path, pdf_output_path, configuration=config, options=options)

print(f"✅ PDF 리포트 생성 완료: {pdf_output_path}")

# ✅ PDF 자동 열기 (Windows 전용)
os.startfile(pdf_output_path)
