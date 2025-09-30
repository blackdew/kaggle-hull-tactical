# Repository Guidelines

## 프로젝트 정보

- 프로젝트: Hull Tactical Market Prediction
- Kaggle 슬러그: `hull-tactical-market-prediction`
- Kaggle CLI 예시
  - 데이터 다운로드: `kaggle competitions download -c hull-tactical-market-prediction`
  - 결과 제출: `kaggle competitions submit -c hull-tactical-market-prediction -f <file> -m "msg"`

## 프로젝트 구조

- `main.py`: 로컬 실행 엔트리 포인트
- `baseline_model.py`, `data_exploration.py`, `kaggle_baseline_model.py`: 학습·EDA 스크립트
- `kaggle_evaluation/`: 로컬 평가 프레임워크(벤더 성격, 광범위 리팩토링 금지)
- `notebooks/`: 탐색형 노트북·리포트, 제출 산출물은 여기 또는 `submissions/`
- `data/`: 로컬 데이터 보관(대용량/원천 데이터 커밋 금지), `results/`, `docs/`
- `pyproject.toml`, `.python-version`: Python 3.11 환경 정의, `.kaggle/kaggle.json`: API 키(비공개)

## 빌드·실행·개발 명령

- 의존성 설치: `uv sync`(권장) 또는 `uv pip install -r requirements.txt`
- 스크립트 실행: `uv run python main.py` | `uv run python baseline_model.py`
- Jupyter: `uv run jupyter lab`(또는 `uv run jupyter notebook`)
- 테스트: `uv run pytest -q tests/`

## 코딩 스타일·네이밍

- PEP 8, 4칸 들여쓰기, 함수/모듈 `snake_case`, 클래스 `PascalCase`
- 타입 힌트 권장, 작은 순수 함수 지향, 진입점 가드 `if __name__ == "__main__":`
- 포매터/린터: `black`, `ruff` 권장. `kaggle_evaluation/` 대규모 리포맷 금지(필요 시 국소 수정+테스트)

## 테스트 가이드라인

- 프레임워크: `pytest` 사용, `tests/` 아래 `test_*.py`로 배치(모듈 경로 미러)
- 우선순위: 전처리, 특징 생성, 평가 유틸 스모크 테스트 중심
- 픽스처: 소형 CSV/Parquet를 `tests/data/`에 두고 결정적 실행 유지

## 커밋·PR 가이드

- 커밋: 명령형 현재시제, 작은 단위, 관련 이슈 참조(`Fixes #12` 등)
- PR: 변경 요약, 실행/재현 방법, 전·후 지표/플롯 첨부, 무관 리팩토링 병합 금지
- 데이터/모델 산출물은 커밋하지 않기, 벤더 디렉터리 변경 최소화

## 보안·환경

- Kaggle API 키는 `~/.kaggle/kaggle.json` 또는 프로젝트 `.kaggle/kaggle.json`에 보관(커밋 금지)
- 경로는 상대 경로 사용(`data/train.csv` 등), 결측/파일 없음 처리는 방어적으로
- 캐글 환경 제약 인지: 학습 8h·예측 9h, 인터넷 제한, 공개 외부데이터만 사용

## 에이전트 안내

- 기존 동작 보존, 최소 변경 원칙. 문서·코드 일관성 유지
- `kaggle_evaluation/`는 벤더 성격: 버그 수정 시 근거·테스트 포함하여 국소 변경만
