# CLAUDE.md

이 파일은 Claude Code (claude.ai/code)가 이 저장소에서 코드 작업을 할 때 필요한 가이드를 제공합니다.

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
