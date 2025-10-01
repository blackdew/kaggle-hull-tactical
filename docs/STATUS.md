# Hull Tactical — Current Status Snapshot

Last updated: 2025-10-01 (local repo snapshot)

## 목표 요약
- 예측 대상: `prediction` = S&P 500 포지션 비율 [0, 2]
- 제약/지표: 변동성 ≤ 1.2×시장, Modified Sharpe Ratio 최적화
- 데이터: train≈8,990×98, test=10×99 (`data/*.csv`)

## 현재 구성(완료 사항)
- 실험 트랙 완비: `experiments/000`(피처 분석) → `001`(베이스라인) → `002`(가설검증) → `003`(후보 패키징/서빙)
- 후보 모델
  - 후보 A(성능): Lasso(α=1e-4) + Top‑20(abs corr) + k=50, [0,2] 클립, vol‑cap 옵션 지원
  - 후보 B(단순/안정): OLS + base features + k=18, [0,2] 클립, vol‑cap 옵션 지원
- 서빙/평가
  - 로컬 게이트웨이: `kaggle_evaluation/`(벤더, 광범위 리팩토링 금지)
  - 제출 서버(로컬): `submissions/submission.py` → `submission.parquet` 생성
  - 캐글 커널(동일 로직): `kaggle_kernel/kaggle_inference.py`
- 문서화: `docs/PRD.md`, `docs/PLAN.md`, `docs/EXPERIMENTS-SUMMARY.md`

## 핵심 결과(요약)
- 베이스라인(OLS+k=50): Sharpe≈0.383, vol_ratio≈1.249
- k 튜닝: k=18~20 구간 유리(예: k=18 Sharpe≈0.444, vol≈1.045)
- vol‑aware(cap=1.2): Sharpe 유지, 변동성 안정화(≈1.05±)
- Lasso Top‑20(α=1e‑4): Sharpe≈0.57, vol≈1.05, 표준편차≈0.23 → 추천 1안
- GBR Top‑20(+cap): Sharpe≈0.566, vol≈1.165 → 대안

자세한 수치/근거: `experiments/002/REPORT.md`, `experiments/002/results/*.csv`

## 빠른 실행(로컬)
- 후보 제출 생성
  - 후보 A: `uv run python experiments/003/run_candidates.py --candidate A`
  - 후보 B: `uv run python experiments/003/run_candidates.py --candidate B`
- 서빙(검증/로그/제약 포함)
  - 후보 A: `uv run python experiments/003/serve.py --candidate A`
  - 후보 B: `uv run python experiments/003/serve.py --candidate B`
  - 로그 출력: `experiments/003/logs/serve_*.json`
- 제출 아티팩트 재현(게이트웨이+서버)
  - `uv run python submissions/run_local.py` → `submission.parquet` 생성
- 제출(Kaggle CLI)
  - `bash scripts/submit.sh -f <csv_or_parquet> -m "msg"`

## 파일 안내(핵심)
- 실험 파이프라인: `experiments/003/pipeline.py`
- 후보 서빙: `experiments/003/serve.py`, 후보 생성: `experiments/003/run_candidates.py`
- 제출 서버(로컬): `submissions/submission.py` / 캐글: `kaggle_kernel/kaggle_inference.py`
- 벤더 게이트웨이: `kaggle_evaluation/default_gateway.py`, 템플릿: `kaggle_evaluation/core/templates.py`

## 다음 단계(Checklist)
- [ ] 후보 확정: 기본 1안(후보 A) 채택, 후보 B 백업 유지
- [ ] 파라미터 고정: `top_n=20`, `alpha=1e-4`, `k=50`, 필요 시 `--vol_cap 1.2`
- [ ] 스모크 테스트 추가: 전처리/피처선택/출력 범위/vol‑cap 스케일 함수
- [ ] 루트 README 보강: 실행/제출 요약 링크(본 문서 포함)
- [ ] 제출 리허설: `submissions/run_local.py` → `scripts/submit.sh` 드라이런

## 유의사항
- `kaggle_evaluation/`는 벤더 디렉터리로, 국소 수정만(근거·테스트 포함)
- 경로는 상대경로 사용, 결측/파일 없음 방어적 처리
- 캐글 환경: 학습 8h·예측 9h, 인터넷 제한, 공개 외부데이터만

## 제출 이슈 기록(2025-10-01)

문제 증상
- Kaggle 제출이 "Kaggle Error"로 실패하고 상세 로그 확인 불가.
- 노트북 콘솔에는 nbconvert 로그만 보이거나, 초기에 TypeError만 출력.
- 일부 실행에서는 로그는 정상이나 `submission.parquet`가 Outputs에 보이지 않음.

원인 분석
- InferenceServer 엔드포인트 등록을 bound method로 넘겨 relay가 거부(FunctionType만 허용) → 초기화 TypeError.
- `InferenceServer._get_gateway_for_test` 미구현으로 추상 메서드 에러(TypeError) 발생.
- rerun(호스트 재실행) 환경에서 서버 컨테이너가 게이트웨이까지 구동하려 해 포트/호스트 충돌 가능 → 평가 단계에서 실패.
- 경로 탐색이 협소해 train/test를 못 찾는 케이스 존재(특히 캐글 환경 경로 차이).
- 배치별 피처 누락 시 스케일러 입력 차원 불일치 가능.

적용한 해결책
- `kaggle_kernel/kaggle_inference.py`
  - predict 등록 방식을 함수(FunctionType)로 래핑하여 전달.
  - `_get_gateway_for_test` 구현.
  - 학습/테스트 경로 폴백 탐색 강화(`/kaggle/input/...`, `.`/`./`, `/kaggle/working`, `data/`).
  - 배치 피처 누락 0.0 보정 + float64 캐스팅.
  - rerun 가드 추가: 호스트 재실행 시 `serve()`만, 로컬/노트북에선 게이트웨이 별도.
- `kaggle_kernel/kaggle_inference_debug.py`(신규)
  - 위 동일 개선 + 풍부한 [DEBUG] 로그(경로/피처/스케일/예측/제출 요약).
  - `KAGGLE_DEBUG_LOCAL=1`로 강제 로컬 실행(서버+게이트웨이 동시 구동) 지원.
  - 게이트웨이 실패 시 폴백 라이터로 `submission.parquet` 직접 생성.
  - 가시성 확보용 복제 아티팩트: `submission_debug.parquet`, `submission_debug.csv`와 `/kaggle/working` 목록 출력.

재현/검증 절차
- 로컬: `uv run python submissions/run_local.py` → `submission.parquet` 생성, 포맷 검사.
- Kaggle 디버그 실행: `kaggle_inference_debug.py` + `KAGGLE_DEBUG_LOCAL=1` → [DEBUG] 로그 확인, 제출 파일 3종 확인.
- 정상 제출: `kaggle_inference.py` 스크립트 커널로 실행 후 Submit.

운영 Runbook(에러 시)
- 증상: Kaggle Error/상세 로그 없음 → 디버그 커널로 재실행(`KAGGLE_DEBUG_LOCAL=1`)해 로그 수집.
- `submission.parquet` 미노출 → `submission_debug.parquet`/`csv`로 가시성 확인·응급 제출, UI 갱신 후 원인 추적.
- 경로 이슈 의심 → 로그의 경로 탐색 결과 확인, 입력 마운트 존재 여부 점검.
- 타입/차원 이슈 의심 → 로그의 피처 개수/입력 shape/스칼라 반환 여부 확인.
