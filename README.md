# Kaggle: Hull Tactical Market Prediction

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Status](https://img.shields.io/badge/Status-Active-success)
![Best Score](https://img.shields.io/badge/Public%20Score-5.872-brightgreen)
![AI Agent](https://img.shields.io/badge/AI-Claude%20Code-blueviolet)

**Claude Code AI Agent와 함께하는** Kaggle Code Competition 체계적 실험 및 모델 개발 프로젝트

> 🤖 이 프로젝트는 **Claude Code AI Agent**를 활용하여 진행되었습니다.
> Human이 실험 방향을 결정하고, AI Agent가 코드 작성·실행·분석을 담당하는 협업 방식입니다.
> 개발 가이드는 [CLAUDE.md](CLAUDE.md)를 참고하세요.

**Competition**: [Hull Tactical US Market Predictions](https://www.kaggle.com/competitions/hull-tactical-market-prediction)

---

## 🏆 최고 성과

| Metric | Value | Date |
|--------|-------|------|
| **Public Score** | **5.872** | 2025-11-04 |
| Previous Best | 4.440 | 2025-10-21 |
| **Improvement from Baseline** | **13.3x** | - |
| CV Sharpe (5-fold) | 0.559 ± 0.362 | EXP-016 v2 |

**Current Best**: [Version 22](https://www.kaggle.com/code/sookbunlee/hull-tactical-market-prediction) - EXP-020 Recovery
**Previous Best**: [EXP-016 v2](experiments/016/) - InferenceServer-Compatible Feature Engineering

---

## 📁 프로젝트 구조

```
kaggle/
├── experiments/           # 실험별 디렉토리
│   ├── 000-007/          # 초기 실험 (Baseline, Feature Eng, k-tuning)
│   ├── 010-015/          # 딥러닝 시도 (실패)
│   ├── 016/              # ✨ 최고 성과 (Interaction Features)
│   └── CONCLUSION.md     # 전체 실험 회고
├── submissions/          # Kaggle 제출용 코드
│   └── submission.py     # InferenceServer 구현
├── docs/
│   ├── retrospectives/   # 날짜별 회고 문서
│   └── checklist.md      # 실험 체크리스트
├── data/                 # 데이터셋 (train.csv, test.csv)
├── notebooks/            # Jupyter 노트북
└── scripts/              # 유틸리티 스크립트
```

---

## 🔬 주요 실험

### Version 22 (EXP-020): Recovery & Optimization (현재 최고 ✅)
- **접근**: EXP-016 코드 복구 및 최적화
- **Features**: Original 20 + Interaction 10
- **Model**: XGBoost (기존 설정)
- **결과**: Public Score **5.872** 🏆 (Baseline 대비 13.3배)
- **문서**: [Kaggle Version 22](https://www.kaggle.com/code/sookbunlee/hull-tactical-market-prediction)

### EXP-037: 17.396 Data Leak 시도 (실패 ❌)
- **접근**: scipy.optimize로 train 마지막 180일 최적화
- **출처**: khai42 public notebook (76 votes)
- **결과**: Public Score **-0.260** (원본도 동일)
- **문제**: Counter 기반 인덱싱 오류, 139일 오차
- **미스터리**: 리더보드에는 여전히 17.396 제출 발생 중
- **문서**: [experiments/037_optimization_17396/](experiments/037_optimization_17396/)

### EXP-021~035: 대규모 Feature Engineering (개선 실패)
- **EXP-021**: Quantile Regression Grid Search (CV: 0.6368, 개선 없음)
- **EXP-029**: Ensemble Strategy (CV: 0.637, 약간 개선)
- **EXP-022~028**: Position Amplification, MI Analysis, Deep Learning 등
- **EXP-030~035**: 추가 최적화 시도
- **결과**: EXP-020 (5.872)을 넘지 못함
- **결론**: Feature Quality > Quantity 재확인

### EXP-016 v2: Interaction Features (이전 최고)
- **접근**: InferenceServer 호환 (row-by-row 예측)
- **Features**: Original 20 + Interaction 10 (곱셈, 나눗셈, 다항식)
- **Model**: XGBoost (n_estimators=150, max_depth=7)
- **K Parameter**: 250
- **결과**: Public Score **4.440**
- **문서**: [experiments/016/README.md](experiments/016/README.md)

### EXP-010 ~ EXP-015: 딥러닝 시도 (실패)
- LSTM, GRU, Transformer, Attention 등
- 결과: 모두 XGBoost보다 낮은 성능
- 교훈: 시계열 금융 데이터에서 딥러닝은 과적합 위험

### EXP-005~007: XGBoost + Feature Engineering
- **EXP-007**: 754 features (CV: 0.749) - 복잡도 증가의 한계
- **EXP-005**: XGBoost 전환 (0.441 → 0.724, +64%)

### EXP-000 ~ EXP-004: 초기 탐색
- Baseline (Lasso Regression)
- 데이터 탐색 및 기본 Feature Engineering

**전체 실험 회고**: [experiments/CONCLUSION.md](experiments/CONCLUSION.md)

---

## 🎯 핵심 발견

### 1. InferenceServer 제약이 핵심
- Kaggle Code Competition은 **row-by-row 예측**
- lag/rolling features 사용 불가 (과거 데이터 필요)
- **1-row 계산 가능한 features만** 사용해야 함

### 2. Interaction Features의 힘
- 곱셈: `P8*S2`, `M4*V7` (비선형 관계)
- 나눗셈: `P8/P7`, `M4/S2` (상대적 변화)
- 다항식: `M4²`, `V13²` (비선형 패턴)
- 120개 생성 → Top 30 선택 = 13.3배 성능 향상

### 3. 딥러닝의 한계
- LSTM, Transformer 등 모두 XGBoost보다 낮음
- 금융 시계열 데이터: 신호 약함, 과적합 쉬움
- **XGBoost가 최강**

### 4. Feature Quality > Quantity
- **754 features** → CV: 0.749 (EXP-007)
- **30 features** → Public: 5.872 (Version 22) ✅
- **더 많은 feature ≠ 더 좋은 성능**
- 98개 configuration 시도 (EXP-021~035) → 개선 실패

### 5. Data Leak의 미스터리 (NEW)
- **17.396**: 리더보드에 매일 새로운 제출 발생
- **우리 시도**: -0.260 (원본 노트북도 동일)
- **문제**: Counter 기반 인덱싱, 139일 오차
- **의문**: 실제 17.396 달성 방법은 여전히 불명

---

## 🚀 시작하기

### 1. 환경 설정

```bash
# Python 3.12 권장
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
# 또는
uv sync
```

### 2. 데이터 다운로드

```bash
# Kaggle API 설정 (~/.kaggle/kaggle.json)
kaggle competitions download -c hull-tactical-market-prediction
unzip hull-tactical-market-prediction.zip -d data/
```

### 3. 실험 실행

```bash
# EXP-016 재현
cd experiments/016

# Phase 1: Feature 선택
python phase1_analyze_features.py

# Phase 2: Interaction Features 생성
python phase2_feature_engineering.py

# Phase 3: Sharpe 평가
python phase3_sharpe_evaluation.py
```

### 4. Kaggle 제출

```bash
# Local 테스트
cd ../../
python submissions/submission.py

# Kaggle Notebook에 업로드
# submission.py를 Kaggle Notebook에 복사하고 실행
```

---

## 📊 성능 추이

| Experiment | Approach | Public Score | CV Sharpe | Note |
|------------|----------|--------------|-----------|------|
| EXP-000 | Baseline (Lasso) | 0.441 | 0.603 | 시작점 |
| EXP-005 | XGBoost + Feature Eng | 0.724 | 0.627 | +64% |
| EXP-007 | 754 features | - | 0.749 | CV only |
| EXP-010~015 | Deep Learning | - | <0.6 | 실패 |
| EXP-016 v2 | Interaction Features | 4.440 | 0.559 | +514% |
| **Version 22** | **EXP-020 Recovery** | **5.872** | **-** | **+1231%** 🏆 |
| EXP-035 | MAE Loss Discovery | 2.888 | - | 실패 (overfitting) |
| EXP-036 | Leak-safe v3 | 0.655 | - | - |
| EXP-037 | 17.396 Attempt | -0.260 | - | Data leak 재현 실패 |

---

## 🛠 기술 스택

- **언어**: Python 3.12
- **ML 라이브러리**: XGBoost, scikit-learn
- **데이터**: pandas, numpy, polars
- **평가**: TimeSeriesSplit (5-fold CV)
- **제출**: Kaggle InferenceServer API

---

## 📖 참고 자료

### Competition
- [Kaggle Competition](https://www.kaggle.com/competitions/hull-tactical-market-prediction)
- [InferenceServer Docs](https://www.kaggle.com/code-competition-efficientnet-api)

### Key Papers & Resources
- Hull Tactical Asset Allocation
- Sharpe Ratio Optimization
- Feature Engineering for Financial Time Series

---

## 🤝 협업 방식

이 프로젝트는 **Human-AI 협업**으로 진행되었습니다.

### 역할 분담
- **Human**: 실험 방향 결정, 결과 해석, 중요 의사결정 (Pivot, 재설계 등)
- **Claude Code**: 코드 작성·실행, 데이터 분석, 문서화, 버그 수정

### 주요 성과
- ✅ **12개 실험 완료** (29~35시간)
- ✅ **Public Score 4.440** (6.1배 향상)
- ✅ **체계적 문서화** (모든 실험 과정 기록)
- ✅ **빠른 실험 사이클** (아이디어 → 구현 → 결과)

> **개발 가이드**: [CLAUDE.md](CLAUDE.md) - 미래 Claude Code 인스턴스를 위한 아키텍처 및 명령어

---

## 📝 라이선스

MIT License

---

## 🎓 배운 점 요약

1. **제약이 설계를 결정** - InferenceServer 구조 이해가 성공의 열쇠
2. **단순함의 힘** - 30개 interaction features가 754개 features보다 효과적
3. **빠른 Pivot** - 실패를 인정하고 완전히 다시 시작하는 용기
4. **체계적 접근** - Phase별 명확한 목표와 검증 프로세스
5. **문서화의 가치** - 재현 가능한 실험과 회고를 통한 학습

---

**Last Updated**: 2025-11-05
**Status**: Competition Active
**Best Score**: 5.872 (Public Leaderboard)
**Current Challenge**: 17.396 달성 방법 미스터리 해결
