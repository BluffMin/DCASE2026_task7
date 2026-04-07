import os
import pandas as pd

# -----------------------------
# 경로 설정
# -----------------------------
ROOT = "/workspace/DCASE/task7_data"   # 네 환경에 맞게 수정
META_DIR = os.path.join(ROOT, "metadata")
OUT_DIR = os.path.join(ROOT, "evaluation_setup")

os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------------
# 클래스 매핑 (baseline config와 동일)
# -----------------------------
CLASS_TO_ID = {
    "alarm": 0,
    "baby_cry": 1,
    "dog_bark": 2,
    "engine": 3,
    "fire": 4,
    "footsteps": 5,
    "knocking": 6,
    "telephone_ringing": 7,
    "piano": 8,
    "speech": 9,
}

# -----------------------------
# 어떤 CSV를 train/test에 넣을지 정의
# -----------------------------
SPECS = [
    ("d2-dev-train.csv", "d2-dev-train", "D2", "train"),
    ("d2-dev-test.csv",  "d2-dev-test",  "D2", "test"),
    ("d3-dev-train.csv", "d3-dev-train", "D3", "train"),
    ("d3-dev-test.csv",  "d3-dev-test",  "D3", "test"),
]

train_rows = []
test_rows = []

for csv_name, audio_subdir, domain, split in SPECS:
    csv_path = os.path.join(META_DIR, csv_name)
    audio_dir = os.path.join(ROOT, audio_subdir)

    if not os.path.exists(csv_path):
        print(f"[경고] CSV 없음: {csv_path}")
        continue
    if not os.path.isdir(audio_dir):
        print(f"[경고] 오디오 폴더 없음: {audio_dir}")
        continue

    df = pd.read_csv(csv_path)

    # 컬럼명 확인
    if "filename" not in df.columns or "class" not in df.columns:
        raise ValueError(f"{csv_name} 에 filename, class 컬럼이 없습니다. 현재 컬럼: {list(df.columns)}")

    for _, row in df.iterrows():
        fname = str(row["filename"]).strip()
        cls = str(row["class"]).strip()

        if cls not in CLASS_TO_ID:
            raise ValueError(f"{csv_name}: 알 수 없는 class '{cls}'")

        # baseline에서 읽을 상대경로
        rel_path = f"{audio_subdir}/{fname}"

        # 실제 파일 존재 확인
        abs_path = os.path.join(ROOT, rel_path)
        if not os.path.exists(abs_path):
            print(f"[누락] 파일 없음: {abs_path}")
            continue

        out_row = [rel_path, cls, domain, CLASS_TO_ID[cls]]

        if split == "train":
            train_rows.append(out_row)
        else:
            test_rows.append(out_row)

# 저장
train_out = os.path.join(OUT_DIR, "development_train.txt")
test_out = os.path.join(OUT_DIR, "development_test.txt")

pd.DataFrame(train_rows).to_csv(train_out, sep="\t", header=False, index=False)
pd.DataFrame(test_rows).to_csv(test_out, sep="\t", header=False, index=False)

print(f"저장 완료:")
print(f" - {train_out} ({len(train_rows)}개)")
print(f" - {test_out} ({len(test_rows)}개)")