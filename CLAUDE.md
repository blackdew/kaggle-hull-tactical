# Repository Guidelines

이 문서는 본 저장소에서 일관된 협업을 하기 위한 실무 가이드를 제공합니다.

## 프로젝트 개요

이 디렉토리는 데이터 사이언스와 머신러닝 프로젝트를 위한 Kaggle 작업 공간입니다. 현재는 비어있으며 새로운 Kaggle 경진대회나 데이터셋을 위해 준비되어 있습니다.

## 일반적인 Kaggle 워크플로우 명령어

이것은 Kaggle 작업공간이므로, 일반적인 명령어들은 특정 프로젝트 타입에 따라 달라집니다:

### Python 기반 프로젝트

- `uv sync` - 의존성 설치 (uv 사용 시)
- `uv pip install -r requirements.txt` - pip 방식 의존성 설치 (fallback)
- `uv run python main.py` 또는 `uv run python train.py` - 메인 스크립트 실행
- `uv run jupyter notebook` 또는 `uv run jupyter lab` - Jupyter 환경 시작
- `uv run pytest tests/` - 테스트 실행 (테스트 프레임워크가 설정된 경우)

### 데이터 처리

- Kaggle CLI 명령어 (kaggle 패키지가 설치된 경우):
  - `kaggle competitions download -c <경진대회명>` - 경진대회 데이터 다운로드
  - `kaggle datasets download -d <데이터셋경로>` - 데이터셋 다운로드
  - `kaggle competitions submit -c <경진대회명> -f <파일> -m "메시지"` - 예측 결과 제출

## 일반적인 프로젝트 구조

Kaggle 프로젝트는 일반적으로 다음과 같은 패턴을 따릅니다:

- `data/` - 원시 데이터와 처리된 데이터셋
- `notebooks/` - 탐색과 실험을 위한 Jupyter 노트북
- `src/` - 소스 코드 모듈
- `models/` - 훈련된 모델 파일
- `submissions/` - 경진대회 제출 파일
- `pyproject.toml` 또는 `requirements.txt` - 의존성 (uv 프로젝트는 pyproject.toml 선호)

## 개발 참고사항

- 스크립트 실행 전 항상 데이터 파일 경로를 확인하세요
- 딥러닝 프레임워크 사용 시 GPU 가용성을 확인하세요
- 대용량 데이터셋 작업 시 메모리 사용량을 모니터링하세요
- 코드는 버전 관리를 사용하되 대용량 데이터 파일은 커밋하지 마세요

## 프로젝트 구조
- `main.py`: 로컬 실행 엔트리 포인트.
- `baseline_model.py`, `data_exploration.py`: 학습/EDA 스크립트.
- `kaggle_evaluation/`: 로컬 평가 프레임워크(벤더 성격, 원칙적 비수정).
- `notebooks/`: 탐색용 노트북. 무거운 EDA는 여기서 수행.
- `data/`: 로컬 데이터(giignore 대상). `results/`, `submissions/`는 산출물.

## 코딩 스타일
- Python 3.11+, PEP 8, 4칸 들여쓰기. 함수/모듈 `snake_case`, 클래스 `PascalCase`.
- 타입 힌트 권장, 작은 함수로 분리. 최상위 실행 금지: `if __name__ == "__main__":` 사용.
- 포매터: 가능하면 `black`/`ruff` 사용. `kaggle_evaluation/`는 대규모 리포맷 금지.

## 테스트 가이드
- 현재 공식 테스트 없음. 추가 시 `pytest` 사용 권장.
- 위치/이름: `tests/` 아래 `test_*.py`, 모듈 경로 미러링.
- 우선순위: 전처리(`preprocess_data`), 간단 유틸, 임계 로직의 스모크 테스트.
- 실행: `pytest -q`.

## 커밋/PR
- 커밋: 간결한 명령형 현재시제(한글/영문 모두 허용). 관련 이슈 참조(예: `Fixes #12`).
- PR: 변경 요약, 실행/재현 방법, 전/후 비교(스크린샷/플롯) 첨부.
- 범위 최소화: 벤더 디렉터리 수정/대규모 리포맷/데이터 파일 포함 금지.

## 보안/설정
- Kaggle API 키는 `.kaggle/kaggle.json`에 보관, 비공개 유지.
- 대용량 데이터는 `data/`에 두고 경로 하드코딩 지양(상대경로 사용).

## 에이전트 주의사항
- 기존 동작을 보존하며 최소 변경 원칙 준수.
- `kaggle_evaluation/`는 버그 수정 외 변경 금지. 변경 시 근거/테스트 포함.