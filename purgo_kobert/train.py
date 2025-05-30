# 라이브러리 및 디바이스 설정 
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, AutoTokenizer
import os
import time
from sklearn.model_selection import train_test_split

# ✅ 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Dataset 클래스 정의
class TextDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=64):
        self.texts = df['text'].tolist()
        self.labels = df['label'].astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __getitem__(self, idx):
        inputs = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'label': torch.tensor(self.labels[idx])
        }

    def __len__(self):
        return len(self.labels)

# ✅ 모델 정의
class CussClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained("monologg/kobert", trust_remote_code=True)
        self.classifier = nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(outputs.pooler_output)

# ✅ 데이터 로드 및 전처리
df1 = pd.read_csv("data/최종 학습용  데이터_3.csv")
df1 = df1[df1['label'].isin([0, 1])]
df1['label'] = df1['label'].astype(int)

# ✅ 학습/검증 데이터 나누기 
train_df, val_df = train_test_split(df1, test_size=0.2, stratify=df1['label'], random_state=42)

# ✅ Tokenizer 및 DataLoader 생성
tokenizer = AutoTokenizer.from_pretrained("monologg/kobert", trust_remote_code=True)
train_dataset = TextDataset(train_df, tokenizer)
val_dataset = TextDataset(val_df, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# ✅ 모델 초기화 및 기존 학습 모델 불러오기
model = CussClassifier().to(device)
model_path = "model/purgo_V3.pth"
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"📂 기존 모델 불러와서 이어서 학습합니다: {model_path}")
else:
    print("⚠️ 기존 모델이 없어 새로 학습을 시작합니다.")

optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
loss_fn = nn.CrossEntropyLoss()

# ✅ 학습 루프 시작
print("\n📚 학습 시작...")
num_batches = len(train_loader)

for epoch in range(4):
    start_time = time.time()
    total_loss = 0
    model.train()

    for i, batch in enumerate(train_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids, attention_mask)
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        progress = (i + 1) / num_batches * 100
        print(f"\r🌀 Epoch {epoch+1} 진행률: {progress:.2f}% | 현재 Loss: {loss.item():.4f}", end='')

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"\n📘 Epoch {epoch+1} 완료 - Train Loss: {total_loss:.4f} | 소요 시간: {elapsed:.2f}초")

    # ✅ 검증 루프
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask)
            val_loss += loss_fn(outputs, labels).item()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    print(f"✅ Validation Loss: {val_loss:.4f}, Accuracy: {acc:.4f}")

# ✅ 모델 저장 (덮어쓰기)
os.makedirs("model", exist_ok=True)
torch.save(model.state_dict(), model_path)
print(f"\n✅ 모델 저장 완료 (덮어쓰기): {model_path}")
print("🎉 전체 학습 완료! 이제 purgo_V3.pth 모델을 사용할 수 있어요.")
