# Kaggle: Hull Tactical Market Prediction

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Status](https://img.shields.io/badge/Status-Active-success)
![Best Score](https://img.shields.io/badge/Public%20Score-4.440-brightgreen)

Kaggle Code Competition을 위한 체계적 실험 및 모델 개발 프로젝트

**Competition**: [Hull Tactical US Market Predictions](https://www.kaggle.com/competitions/hull-tactical-market-prediction)

---

## 🏆 최고 성과

| Metric | Value | Date |
|--------|-------|------|
| **Public Score** | **4.440** | 2025-10-21 |
| Previous Best | 0.724 | - |
| **Improvement** | **6.1x** | - |
| CV Sharpe (5-fold) | 0.559 ± 0.362 | EXP-016 v2 |

**Experiment**: [EXP-016 v2](experiments/016/) - InferenceServer-Compatible Feature Engineering

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

### EXP-016 v2: Interaction Features (최종 ✅)
- **접근**: InferenceServer 호환 (row-by-row 예측)
- **Features**: Original 20 + Interaction 10 (곱셈, 나눗셈, 다항식)
- **Model**: XGBoost (n_estimators=150, max_depth=7)
- **K Parameter**: 250
- **결과**: Public Score **4.440** 🏆
- **문서**: [experiments/016/README.md](experiments/016/README.md)

### EXP-010 ~ EXP-015: 딥러닝 시도 (실패)
- LSTM, GRU, Transformer, Attention 등
- 결과: 모두 XGBoost보다 낮은 성능
- 교훈: 시계열 금융 데이터에서 딥러닝은 과적합 위험

### EXP-007: Feature Engineering 확장
- 754 features (lag, rolling, cross-sectional, volatility, momentum)
- CV Sharpe: 0.749
- 결과: 0.75가 해당 접근의 상한으로 판단

### EXP-005: XGBoost + Feature Engineering
- Baseline에서 XGBoost로 전환
- CV Sharpe: 0.627
- Kaggle: 0.441 → 0.724 (+64%)

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
- 120개 생성 → Top 30 선택 = 6.1배 성능 향상

### 3. 딥러닝의 한계
- LSTM, Transformer 등 모두 XGBoost보다 낮음
- 금융 시계열 데이터: 신호 약함, 과적합 쉬움
- **XGBoost가 최강**

### 4. Feature 많다고 좋은 게 아님
- 754 features → Sharpe 0.749 (제한적)
- 30 features (interaction) → Public 4.440 (최고)
- **Quality > Quantity**

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
| **EXP-016 v2** | **Interaction Features** | **4.440** | **0.559** | **+514%** 🏆 |

---

## 📚 회고 문서

프로젝트 전체 과정에서 얻은 인사이트와 교훈을 정리한 문서들:

- [2025-10-21 회고](docs/retrospectives/2025-10-21.md) - EXP-016 v2 완전 재설계 성공
- [2025-10-13 회고](docs/retrospectives/2025-10-13.md) - EXP-006, 007 실험 및 한계 인식
- [전체 실험 회고](RETROSPECTIVE.md) - 종합 회고 및 결론

**핵심 교훈**:
1. 제약 조건을 먼저 파악하라 (InferenceServer)
2. 완전 재설계의 용기 (Sunk cost 극복)
3. Interaction features > 복잡한 features
4. XGBoost > Deep Learning (시계열 금융)
5. 체계적 실험 설계 (Phase별 검증)

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

## 🤝 기여

이 프로젝트는 개인 학습 및 실험 목적입니다.

### 작업자
- **Human**: 실험 설계, 방향 결정, 피드백
- **Claude (AI Assistant)**: 코드 작성, 실험 실행, 문서화

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

**Last Updated**: 2025-10-21  
**Status**: Competition Active  
**Best Score**: 4.440 (Public Leaderboard)
