# DCASE 2026 Task 7 Auxiliary Study (Refactored)

이 폴더는 `exp/auxiliary-study` 브랜치를 **읽기 쉽게 다시 나눈 정리 버전**입니다.

원본 브랜치에서 확인한 핵심 구성은 다음과 같습니다.

- `config_task7.py`: 전역 오디오/데이터 설정
- `datasetfactory_task7.py`: waveform 로딩 + padding + one-hot label 생성
- `domain_net.py`: baseline CNN14 계열 backbone
- `domain_net_geometry.py`: geometry head + task-specific residual adapter
- `domain_net_task7_peft.py`: residual / LoRA / hybrid / BN strategy 비교
- `domain_net_task7_residual_focus.py`: 특정 block만 adapter/LoRA 활성화하는 ablation
- `run_task7_peft_ablation_tqdm.py`, `run_task7_residual_focus.py`: 실험 러너
- `experiment_logger.py`: 결과 저장 유틸

## 왜 다시 정리했는가

원본은 **모델 정의 / 실험 러너 / 로깅 / 설정**이 서로 섞여 있어,
처음 보는 사람이 흐름을 따라가기 어렵습니다.

이 버전은 아래 원칙으로 나눴습니다.

1. `config`: 전역 설정만 둔다.
2. `data`: 데이터셋 로딩만 둔다.
3. `models`: backbone / geometry / peft를 분리한다.
4. `runners`: 학습 루프와 평가만 둔다.
5. `scripts`: 실제 실행 entrypoint만 둔다.

## 디렉터리 구조

```text
dcase_task7_aux_refactor/
├─ README.md
└─ task7_aux_refactor/
   ├─ __init__.py
   ├─ config.py
   ├─ data/
   │  ├─ __init__.py
   │  └─ dataset.py
   ├─ models/
   │  ├─ __init__.py
   │  ├─ base_cnn.py
   │  ├─ geometry.py
   │  └─ peft.py
   ├─ runners/
   │  ├─ __init__.py
   │  ├─ common.py
   │  └─ geometry_runner.py
   └─ scripts/
      └─ run_geometry.py
```

## 읽는 순서

### 1) `config.py`
샘플레이트, mel 설정, split 경로 등 **전역 설정**을 모아둡니다.

### 2) `data/dataset.py`
원본 `datasetfactory_task7.py` 역할입니다.
- TSV split 읽기
- waveform 로딩
- pad / truncate
- one-hot target 생성

### 3) `models/base_cnn.py`
원본 `domain_net.py`의 baseline backbone을 옮긴 파일입니다.
공통 CNN 블록과 task-specific BN 구조를 이 레벨에 둡니다.

### 4) `models/geometry.py`
원본 `domain_net_geometry.py`를 더 읽기 쉽게 정리한 버전입니다.
- residual adapter
- geometry head
- class anchor
- partial checkpoint load
- task별 unfreeze 정책

### 5) `models/peft.py`
PEFT 계열 실험용 모듈을 별도 파일로 분리했습니다.
지금은 최소 구조만 남겨두고, residual/LoRA/hybrid 확장을 쉽게 붙일 수 있게 만들었습니다.

### 6) `runners/*`
학습 루프, optimizer 생성, epoch 단위 평가를 여기 둡니다.
모델 정의 파일에 학습 코드가 섞이지 않도록 분리했습니다.

## 원본 브랜치와의 대응표

| 원본 파일 | 정리 후 위치 | 역할 |
|---|---|---|
| `config_task7.py` | `config.py` | 전역 설정 |
| `datasetfactory_task7.py` | `data/dataset.py` | 데이터셋 로딩 |
| `domain_net.py` | `models/base_cnn.py` | baseline backbone |
| `domain_net_geometry.py` | `models/geometry.py` | geometry 확장 |
| `domain_net_task7_peft.py` | `models/peft.py` | PEFT 모듈 |
| `baseline_DIL_task7_geometry.py` | `runners/geometry_runner.py` + `scripts/run_geometry.py` | 러너/실행 분리 |

## 중요한 점

이 refactor는 **가독성과 유지보수성 개선**이 목적입니다.
즉, 원본 브랜치의 모든 실험 옵션을 완전히 1:1 복제한 최종본이라기보다,
**구조를 바로 이해하고 확장하기 쉬운 정리판**에 가깝습니다.

다음 단계로는 아래 순서를 추천합니다.

1. geometry 실험부터 이 구조에서 정상 실행 확인
2. 그다음 PEFT preset (`none`, `residual`, `lora`, `hybrid`) 이식
3. 마지막으로 residual-focus block subset ablation 이식

## 실행 예시

```bash
python -m scripts.run_geometry   --data-root /path/to/task7_data   --epochs 5   --batch-size 16   --lr 1e-3
```

## 메모

원본 브랜치에서 확인한 사실:
- 오디오는 `sample_rate = 32000`, `clip_samples = sample_rate * 4`, `mel_bins = 64`
- 클래스 수는 10개
- dataset loader는 `librosa`로 waveform을 읽고 zero-padding 후 one-hot target을 만듭니다
- geometry 모델은 shared CNN + task-specific BN + task-specific adapter + class anchor 구조입니다
