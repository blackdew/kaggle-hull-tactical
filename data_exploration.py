import pandas as pd
import numpy as np
# matplotlib 백엔드 설정
import matplotlib
matplotlib.use('Agg')  # GUI 없이 그래프 생성
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터 로드
print("데이터 로딩 중...")
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

print(f"Train 데이터 크기: {train_df.shape}")
print(f"Test 데이터 크기: {test_df.shape}")

# 기본 정보 출력
print("\n=== Train 데이터 기본 정보 ===")
print(train_df.info())

print("\n=== 컬럼 분석 ===")
print(f"전체 컬럼 수: {len(train_df.columns)}")

# 컬럼 그룹별 분석
d_cols = [col for col in train_df.columns if col.startswith('D')]
e_cols = [col for col in train_df.columns if col.startswith('E')]
i_cols = [col for col in train_df.columns if col.startswith('I')]
m_cols = [col for col in train_df.columns if col.startswith('M')]
p_cols = [col for col in train_df.columns if col.startswith('P')]
s_cols = [col for col in train_df.columns if col.startswith('S')]
v_cols = [col for col in train_df.columns if col.startswith('V')]

print(f"D 컬럼 (날짜 관련): {len(d_cols)}")
print(f"E 컬럼 (이벤트 관련): {len(e_cols)}")
print(f"I 컬럼 (지수 관련): {len(i_cols)}")
print(f"M 컬럼 (거시경제 관련): {len(m_cols)}")
print(f"P 컬럼 (가격 관련): {len(p_cols)}")
print(f"S 컬럼 (감정 관련): {len(s_cols)}")
print(f"V 컬럼 (변동성 관련): {len(v_cols)}")

# 타겟 변수 분석
target_cols = ['forward_returns', 'risk_free_rate', 'market_forward_excess_returns']
print(f"\n=== 타겟 변수 통계 ===")
print(train_df[target_cols].describe())

# 결측치 분석
print(f"\n=== 결측치 분석 ===")
missing_counts = train_df.isnull().sum()
missing_percent = (missing_counts / len(train_df)) * 100
missing_df = pd.DataFrame({
    'Missing_Count': missing_counts,
    'Missing_Percent': missing_percent
}).sort_values('Missing_Count', ascending=False)

print("결측치가 있는 컬럼들:")
print(missing_df[missing_df['Missing_Count'] > 0].head(10))

# 타겟 변수 시각화
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.hist(train_df['forward_returns'].dropna(), bins=50, alpha=0.7)
plt.title('Forward Returns 분포')
plt.xlabel('Forward Returns')

plt.subplot(1, 3, 2)
plt.hist(train_df['risk_free_rate'].dropna(), bins=50, alpha=0.7)
plt.title('Risk Free Rate 분포')
plt.xlabel('Risk Free Rate')

plt.subplot(1, 3, 3)
plt.hist(train_df['market_forward_excess_returns'].dropna(), bins=50, alpha=0.7)
plt.title('Market Forward Excess Returns 분포')
plt.xlabel('Market Forward Excess Returns')

plt.tight_layout()
plt.savefig('target_distributions.png', dpi=150, bbox_inches='tight')
plt.show()

# D 컬럼 (날짜 관련) 분석
print(f"\n=== D 컬럼 분석 ===")
print(train_df[d_cols].describe())

# 시간에 따른 타겟 변수 변화
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(train_df['date_id'], train_df['forward_returns'])
plt.title('Forward Returns Over Time')
plt.xlabel('Date ID')
plt.ylabel('Forward Returns')

plt.subplot(2, 2, 2)
plt.plot(train_df['date_id'], train_df['market_forward_excess_returns'])
plt.title('Market Forward Excess Returns Over Time')
plt.xlabel('Date ID')
plt.ylabel('Market Forward Excess Returns')

plt.tight_layout()
plt.savefig('time_series_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n데이터 탐색 완료!")