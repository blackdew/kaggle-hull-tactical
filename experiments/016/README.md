# EXP-016: Feature Engineering Deep Dive - 1.0 돌파

## 목표
**Sharpe 1.0+ 달성** (현재 최고 0.749 대비 +34%)

## 배경
- **EXP-007 최고 성능**: Sharpe 0.749 (XGBoost + 754 features)
- **이전 실패**: 얕은 시도, 754개에서 멈춤, Hyperparameter tuning 부족
- **새로운 접근**: 깊게 파기, 포기하지 않기, 한계까지 밀어붙이기

## 전략
1. **Phase 1**: 754 features 심층 분석 → 진짜 효과적인 것만 선택
2. **Phase 2**: 1500+ features로 확장 (Interaction, Polynomial, Domain)
3. **Phase 3**: Feature selection + Hyperparameter tuning
4. **Phase 4**: Ensemble (조건부)

## 실험 진행 상황

**시작일**: 2025-10-18
**현재 상태**: Phase 2 완료 ❌ (실패), Phase 3 준비 중
**현재 Sharpe**: **0.874** (Top 20 features) ← **+16.7%** vs EXP-007!

### Phase 1 완료 ✅ 성공
- [x] Phase 1.1: Feature Importance Analysis (~15분)
- [x] Phase 1.2: Null Importance Test (~30분)
- [x] Phase 1.4: Baseline Comparison (~10분)
- **핵심 발견: Top 20 features > 754 features (+21%)**
- **Less is More!**

### Phase 2 완료 ❌ 실패
- [x] Phase 2.1: Interaction Features (~5분) - **실패 -21.6%**
- **발견: Feature 추가 = 과적합**
- **확인: Top 20이 최적**

### Phase 3 준비 중
- [ ] Phase 3.3: Hyperparameter Tuning (Top 20 고정)
- **목표: Sharpe 1.0+**
- **전략: Feature 고정, Model 최적화**

## 폴더 구조
```
experiments/016/
├── README.md              # 이 파일
├── CHECKLIST.md           # 실험 체크리스트 (진행 상황)
├── HYPOTHESES.md          # 가설 및 실험 설계
├── feature_analysis.py    # Phase 1: Feature 분석
├── feature_engineering.py # Phase 2: Feature 확장 (예정)
├── run_experiments.py     # 실험 실행 (예정)
├── results/               # 결과 저장
└── REPORT.md             # 최종 리포트 (완료 후)
```

## 다음 단계
1. EXP-007 feature 생성 코드 확인
2. 754 features 로딩 및 전처리
3. Phase 1.1 시작: SHAP analysis

## 원칙
- ✅ 깊게 파기 - 각 단계 완료할 때까지
- ✅ 포기하지 않기 - 1.0 넘거나 진짜 한계 확인할 때까지
- ✅ 변명하지 않기 - 실제 측정으로 검증
- ✅ 문서화 - 모든 실험 결과 기록

---

**작성일**: 2025-10-18
**목표**: Sharpe 1.0+
**예상 기간**: 2~3주
