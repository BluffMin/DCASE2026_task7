import os
import pandas as pd
import soundfile as sf
import numpy as np

ROOT = "/workspace/DCASE/task7_data"
META_DIR = os.path.join(ROOT, "metadata")
EVAL_DIR = os.path.join(ROOT, "evaluation_setup")

SR = 32000
CHUNK_SEC = 4
CHUNK_SAMPLES = SR * CHUNK_SEC

os.makedirs(EVAL_DIR, exist_ok=True)

def chunk_audio_file(in_wav, out_dir, base_name):
    audio, sr = sf.read(in_wav)
    if sr != SR:
        raise ValueError(f"Sample rate mismatch: {in_wav}, got {sr}, expected {SR}")

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    os.makedirs(out_dir, exist_ok=True)

    out_files = []
    n = len(audio)

    start = 0
    chunk_idx = 0
    while start < n:
        end = start + CHUNK_SAMPLES
        chunk = audio[start:end]

        # 마지막 chunk가 4초보다 짧으면 zero-pad
        if len(chunk) < CHUNK_SAMPLES:
            pad = CHUNK_SAMPLES - len(chunk)
            chunk = np.pad(chunk, (0, pad), mode="constant")

        out_name = f"{base_name}_chunk{chunk_idx:04d}.wav"
        out_path = os.path.join(out_dir, out_name)
        sf.write(out_path, chunk, SR)

        out_files.append(out_name)
        chunk_idx += 1
        start += CHUNK_SAMPLES   # non-overlap 4초 분할

    # 길이가 0인 이상한 파일 대비
    if n == 0:
        chunk = np.zeros(CHUNK_SAMPLES, dtype=np.float32)
        out_name = f"{base_name}_chunk0000.wav"
        out_path = os.path.join(out_dir, out_name)
        sf.write(out_path, chunk, SR)
        out_files.append(out_name)

    return out_files

train_specs = [
    {
        "csv": "d2-dev-train.csv",
        "audio_dir": "d2-dev-train",
        "chunk_dir": "d2-dev-train-chunked",
        "domain": "D2",
    },
    {
        "csv": "d3-dev-train.csv",
        "audio_dir": "d3-dev-train",
        "chunk_dir": "d3-dev-train-chunked",
        "domain": "D3",
    },
]

test_specs = [
    {
        "csv": "d2-dev-test.csv",
        "audio_dir": "d2-dev-test",
        "domain": "D2",
    },
    {
        "csv": "d3-dev-test.csv",
        "audio_dir": "d3-dev-test",
        "domain": "D3",
    },
]

# class -> id
# 네 metadata의 class 문자열에 맞게 꼭 수정해야 함
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

train_rows = []
test_rows = []

# -----------------
# train: chunking
# -----------------
for spec in train_specs:
    df = pd.read_csv(os.path.join(META_DIR, spec["csv"]))

    for _, row in df.iterrows():
        fname = str(row["filename"]).strip()
        cls = str(row["class"]).strip()

        if cls not in CLASS_TO_ID:
            raise ValueError(f"Unknown class: {cls}")

        in_wav = os.path.join(ROOT, spec["audio_dir"], fname)
        if not os.path.exists(in_wav):
            print(f"[missing train wav] {in_wav}")
            continue

        base = os.path.splitext(fname)[0]
        out_dir = os.path.join(ROOT, spec["chunk_dir"])
        out_files = chunk_audio_file(in_wav, out_dir, base)

        for out_name in out_files:
            rel_path = f"{spec['chunk_dir']}/{out_name}"
            train_rows.append([rel_path, cls, spec["domain"], CLASS_TO_ID[cls]])

# -----------------
# test: original
# -----------------
for spec in test_specs:
    df = pd.read_csv(os.path.join(META_DIR, spec["csv"]))

    for _, row in df.iterrows():
        fname = str(row["filename"]).strip()
        cls = str(row["class"]).strip()

        if cls not in CLASS_TO_ID:
            raise ValueError(f"Unknown class: {cls}")

        wav_path = os.path.join(ROOT, spec["audio_dir"], fname)
        if not os.path.exists(wav_path):
            print(f"[missing test wav] {wav_path}")
            continue

        rel_path = f"{spec['audio_dir']}/{fname}"
        test_rows.append([rel_path, cls, spec["domain"], CLASS_TO_ID[cls]])

# 저장
train_txt = os.path.join(EVAL_DIR, "development_train.txt")
test_txt = os.path.join(EVAL_DIR, "development_test.txt")

pd.DataFrame(train_rows).to_csv(train_txt, sep="\t", header=False, index=False)
pd.DataFrame(test_rows).to_csv(test_txt, sep="\t", header=False, index=False)

print(f"saved: {train_txt} ({len(train_rows)} rows)")
print(f"saved: {test_txt} ({len(test_rows)} rows)")