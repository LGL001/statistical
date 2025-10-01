# chap04_inference.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- 초기 설정 ---
# .py 스크립트에서는 %precision 매직 명령어를 사용하지 않으므로 주석 처리
# np.set_printoptions(precision=3) 등으로 대체 가능


# === 4.1 모집단과 표본 ===

# --- 데이터 불러오기 ---
# 이 코드가 작동하려면 ../data/ch4_scores400.csv 파일이 필요합니다.
df = pd.read_csv('../data/ch4_scores400.csv')
scores = np.array(df['score'])
print("===== 모집단 점수 데이터 (처음 10개) =====")
print(scores[:10])
print("\n")

# --- 표본 추출 방법 ---
print("===== 표본 추출 예시 =====")
# 복원 추출 (replace=True가 기본값)
print("복원 추출:", np.random.choice([1, 2, 3], 3))

# 비복원 추출
print("비복원 추출:", np.random.choice([1, 2, 3], 3, replace=False))

# 시드 고정
np.random.seed(0)
print("시드 고정 후 추출:", np.random.choice([1, 2, 3], 3))

# 표본 평균 계산
np.random.seed(0)
sample = np.random.choice(scores, 20)
print("\n표본 평균:", sample.mean())
print("모 평균:", scores.mean())

# 반복에 따른 표본 평균의 변화
print("\n5회 반복 시 표본 평균:")
for i in range(5):
    sample = np.random.choice(scores, 20)
    print(f'{i+1}번째 무작위 추출로 얻은 표본평균: {sample.mean():.2f}')
print("\n")


# === 4.2 확률 모델 ===

# --- 확률분포 ---
print("===== 불공정한 주사위 확률분포 실험 =====")
dice = [1, 2, 3, 4, 5, 6]
prob = [1/21, 2/21, 3/21, 4/21, 5/21, 6/21]

print("확률에 따른 1회 추출:", np.random.choice(dice, p=prob))

# 100회 실험
num_trial = 100
sample = np.random.choice(dice, num_trial, p=prob)
print("\n100회 실험 샘플 (일부):", sample[:10])

# 도수분포표
freq, _ = np.histogram(sample, bins=6, range=(1, 7))
freq_df = pd.DataFrame({'frequency':freq,
                        'relative frequency':freq / num_trial},
                       index = pd.Index(np.arange(1, 7), name='dice'))
print("\n100회 실험 도수분포표:\n", freq_df)

# 100회 실험 히스토그램
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
ax.hist(sample, bins=6, range=(1, 7), density=True, rwidth=0.8)
ax.hlines(prob, np.arange(1, 7), np.arange(2, 8), colors='gray')
ax.set_xticks(np.linspace(1.5, 6.5, 6))
ax.set_xticklabels(np.arange(1, 7))
ax.set_xlabel('dice')
ax.set_ylabel('relative frequency')
ax.set_title('Dice Roll Simulation (100 trials)')
print("\n100회 실험 히스토그램을 표시합니다. 창을 닫아야 다음 코드가 실행됩니다.")
plt.show()


# 10000회 실험 히스토그램
num_trial = 10000
sample = np.random.choice(dice, size=num_trial, p=prob)

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
ax.hist(sample, bins=6, range=(1, 7), density=True, rwidth=0.8)
ax.hlines(prob, np.arange(1, 7), np.arange(2, 8), colors='gray')
ax.set_xticks(np.linspace(1.5, 6.5, 6))
ax.set_xticklabels(np.arange(1, 7))
ax.set_xlabel('dice')
ax.set_ylabel('relative frequency')
ax.set_title('Dice Roll Simulation (10000 trials)')
print("\n10000회 실험 히스토그램을 표시합니다. 창을 닫아야 다음 코드가 실행됩니다.")
plt.show()


# === 4.3 추측통계에서의 확률 ===
print("===== 추측통계 실험 =====")

# 모집단 히스토그램
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
ax.hist(scores, bins=100, range=(0, 100), density=True)
ax.set_xlim(20, 100)
ax.set_ylim(0, 0.042)
ax.set_xlabel('score')
ax.set_ylabel('relative frequency')
ax.set_title('Population Score Distribution')
print("\n모집단 점수 분포 히스토그램을 표시합니다. 창을 닫아야 다음 코드가 실행됩니다.")
plt.show()

# 1회 표본 추출
print("무작위 표본 1개 추출:", np.random.choice(scores))

# 10000개 표본 추출 후 히스토그램
sample = np.random.choice(scores, 10000)
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
ax.hist(sample, bins=100, range=(0, 100), density=True)
ax.set_xlim(20, 100)
ax.set_ylim(0, 0.042)
ax.set_xlabel('score')
ax.set_ylabel('relative frequency')
ax.set_title('Sample Score Distribution (n=10000)')
print("\n10000개 표본의 히스토그램을 표시합니다. 창을 닫아야 다음 코드가 실행됩니다.")
plt.show()

# 표본 평균의 분포
sample_means = [np.random.choice(scores, 20).mean()
                for _ in range(10000)]

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
ax.hist(sample_means, bins=100, range=(0, 100), density=True)
ax.vlines(np.mean(scores), 0, 1, 'gray') # 모평균을 세로선으로 표시
ax.set_xlim(50, 90)
ax.set_ylim(0, 0.13)
ax.set_xlabel('sample mean score')
ax.set_ylabel('relative frequency')
ax.set_title('Distribution of Sample Means')
print("\n표본 평균의 분포 히스토그램을 표시합니다.")
plt.show()