# chap03_analysis.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- 초기 설정 ---
# 소수점 3자리까지 출력 설정
# .py 스크립트에서는 %precision 매직 명령어 대신 pd.set_option 사용
pd.set_option('precision', 3)

# === 3.1 두 데이터 사이의 관계를 나타내는 지표 ===

# --- 데이터 불러오기 및 DataFrame 생성 ---
# 이 코드가 작동하려면 ../data/ch2_scores_em.csv 파일이 필요합니다.
df = pd.read_csv('../data/ch2_scores_em.csv',
                 index_col='student number')

en_scores = np.array(df['english'])[:10]
ma_scores = np.array(df['mathematics'])[:10]

scores_df = pd.DataFrame({'english': en_scores,
                          'mathematics': ma_scores},
                         index=pd.Index(['A', 'B', 'C', 'D', 'E',
                                         'F', 'G', 'H', 'I', 'J'],
                                        name='student'))
print("===== 10명의 영어 및 수학 점수 =====")
print(scores_df)
print("\n")

# --- 3.1.1 공분산 (Covariance) ---
print("===== 공분산 계산 과정 =====")
summary_df = scores_df.copy()
summary_df['english_deviation'] = \
    summary_df['english'] - summary_df['english'].mean()
summary_df['mathematics_deviation'] = \
    summary_df['mathematics'] - summary_df['mathematics'].mean()
summary_df['product of deviations'] = \
    summary_df['english_deviation'] * summary_df['mathematics_deviation']
print(summary_df)

print("\n공분산 (직접 계산):", summary_df['product of deviations'].mean())

cov_mat = np.cov(en_scores, ma_scores, ddof=0)
print("\nNumPy 공분산 행렬:\n", cov_mat)
print(f"영어-수학 공분산: {cov_mat[0, 1]}")
print(f"영어 분산: {cov_mat[0, 0]}, 수학 분산: {cov_mat[1, 1]}")
print(f"np.var() 결과: 영어 분산: {np.var(en_scores, ddof=0)}, 수학 분산: {np.var(ma_scores, ddof=0)}")
print("\n")

# --- 3.1.2 상관계수 (Correlation coefficient) ---
print("===== 상관계수 계산 =====")
corr_manual = np.cov(en_scores, ma_scores, ddof=0)[0, 1] / \
              (np.std(en_scores) * np.std(ma_scores))
print("상관계수 (수식으로 계산):", corr_manual)

print("\nNumPy 상관행렬:\n", np.corrcoef(en_scores, ma_scores))
print("\nPandas 상관행렬:\n", scores_df.corr())
print("\n")

# === 3.2 2차원 데이터의 시각화 ===

# --- 전체 학생 데이터 불러오기 ---
english_scores = np.array(df['english'])
math_scores = np.array(df['mathematics'])

# --- 3.2.1 산점도 (Scatter Plot) ---
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)
ax.scatter(english_scores, math_scores)
ax.set_xlabel('english')
ax.set_ylabel('mathematics')
ax.set_title('Scatter Plot')
print("산점도 그래프를 표시합니다. 창을 닫아야 다음 코드가 실행됩니다.")
plt.show()

# --- 3.2.2 회귀직선 (Regression Line) ---
poly_fit = np.polyfit(english_scores, math_scores, 1)
poly_1d = np.poly1d(poly_fit)
xs = np.linspace(english_scores.min(), english_scores.max())
ys = poly_1d(xs)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)
ax.set_xlabel('english')
ax.set_ylabel('mathematics')
ax.scatter(english_scores, math_scores, label='score')
ax.plot(xs, ys, color='gray',
        label=f'{poly_fit[1]:.2f}+{poly_fit[0]:.2f}x')
ax.legend(loc='upper left')
ax.set_title('Regression Line Plot')
print("회귀직선 그래프를 표시합니다. 창을 닫아야 다음 코드가 실행됩니다.")
plt.show()

# --- 3.2.3 히트맵 (Heatmap) ---
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111)

c = ax.hist2d(english_scores, math_scores,
              bins=[9, 8], range=[(35, 80), (55, 95)])
ax.set_xlabel('english')
ax.set_ylabel('mathematics')
ax.set_xticks(c[1])
ax.set_yticks(c[2])
fig.colorbar(c[3], ax=ax)
ax.set_title('Heatmap')
print("히트맵 그래프를 표시합니다. 창을 닫아야 다음 코드가 실행됩니다.")
plt.show()

# === 3.3 앤스컴의 예 (Anscombe's Quartet) ===

# --- 데이터 불러오기 ---
# 이 코드가 작동하려면 ../data/ch3_anscombe.npy 파일이 필요합니다.
anscombe_data = np.load('../data/ch3_anscombe.npy')
print("===== 앤스컴 데이터셋 =====")
print("Shape:", anscombe_data.shape)
print("Data[0]:\n", anscombe_data[0])
print("\n")

# --- 통계량 계산 ---
stats_df = pd.DataFrame(index=['X_mean', 'X_variance', 'Y_mean',
                               'Y_variance', 'X&Y_correlation',
                               'X&Y_regression line'])
for i, data in enumerate(anscombe_data):
    dataX = data[:, 0]
    dataY = data[:, 1]
    poly_fit = np.polyfit(dataX, dataY, 1)
    stats_df[f'data{i + 1}'] = \
        [f'{np.mean(dataX):.2f}',
         f'{np.var(dataX):.2f}',
         f'{np.mean(dataY):.2f}',
         f'{np.var(dataY):.2f}',
         f'{np.corrcoef(dataX, dataY)[0, 1]:.2f}',
         f'{poly_fit[1]:.2f}+{poly_fit[0]:.2f}x']
print("앤스컴 데이터셋 통계량:\n", stats_df)
print("\n")

# --- 앤스컴 데이터셋 시각화 ---
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10),
                         sharex=True, sharey=True)
xs = np.linspace(0, 30, 100)
for i, data in enumerate(anscombe_data):
    poly_fit = np.polyfit(data[:, 0], data[:, 1], 1)
    poly_1d = np.poly1d(poly_fit)
    ys = poly_1d(xs)

    ax = axes[i // 2, i % 2]
    ax.set_xlim([4, 20])
    ax.set_ylim([3, 13])
    ax.set_title(f'data{i + 1}')
    ax.scatter(data[:, 0], data[:, 1])
    ax.plot(xs, ys, color='gray')

plt.tight_layout()
print("앤스컴 데이터셋 시각화 그래프를 표시합니다.")
plt.show()