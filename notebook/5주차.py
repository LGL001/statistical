import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %precision 3
# %matplotlib inline

# ### 표본의 추출방법
print("--- [표본 추출] ---")
try:
    df = pd.read_csv('../data/ch4_scores400.csv')
    scores = np.array(df['score'])
    print(f"Scores (first 10): {scores[:10]}")
except FileNotFoundError:
    print("Error: '../data/ch4_scores400.csv' 파일을 찾을 수 없습니다.")
    print("실습을 계속하려면 CSV 파일을 올바른 경로에 위치시켜주세요.")
    # 파일이 없을 경우, 실행 중단을 위해 임시 scores 배열을 비워둡니다.
    scores = np.array([])

# scores 배열이 정상적으로 로드되었을 때만 나머지 코드 실행
if scores.size > 0:
    print(f"Choice 3 from [1,2,3] (with replace): {np.random.choice([1, 2, 3], 3)}")
    print(f"Choice 3 from [1,2,3] (no replace): {np.random.choice([1, 2, 3], 3, replace=False)}")

    np.random.seed(0)
    print(f"Choice 3 from [1,2,3] (seed=0): {np.random.choice([1, 2, 3], 3)}")

    np.random.seed(0)
    sample = np.random.choice(scores, 20)
    print(f"Sample mean (seed=0, n=20): {sample.mean()}")

    print(f"Population mean: {scores.mean()}")

    print("\n--- [5 Samples] ---")
    for i in range(5):
        sample = np.random.choice(scores, 20)
        print(f'{i+1}번째 무작위 추출로 얻은 표본평균', sample.mean())

    # ### 확률분포
    print("\n--- [확률분포] ---")
    dice = [1, 2, 3, 4, 5, 6]
    prob = [1/21, 2/21, 3/21, 4/21, 5/21, 6/21]

    print(f"Choice 1 from dice (prob): {np.random.choice(dice, p=prob)}")

    num_trial = 100
    sample = np.random.choice(dice, num_trial, p=prob)
    print(f"Sample (n=100):\n {sample}")

    freq, _ = np.histogram(sample, bins=6, range=(1, 7))
    df_freq = pd.DataFrame({'frequency':freq,
                            'relative frequency':freq / num_trial},
                           index = pd.Index(np.arange(1, 7), name='dice'))
    print(f"\nFrequency (n=100):\n {df_freq}")


    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.hist(sample, bins=6, range=(1, 7), density=True, rwidth=0.8)
    # 실제의 확률분포를 가로선으로 표시
    ax.hlines(prob, np.arange(1, 7), np.arange(2, 8), colors='gray')
    # 막대 그래프의 [1.5, 2.5, ..., 6.5]에 눈금을 표시
    ax.set_xticks(np.linspace(1.5, 6.5, 6))
    # 주사위 눈의 값은 [1, 2, 3, 4, 5, 6]
    ax.set_xticklabels(np.arange(1, 7))
    ax.set_xlabel('dice')
    ax.set_ylabel('relative frequency')
    ax.set_title('Histogram (n=100)')
    plt.show()


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
    ax.set_title('Histogram (n=10000)')
    plt.show()

    # ## 추측통계에서의 확률
    print("\n--- [추측통계에서의 확률] ---")

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.hist(scores, bins=100, range=(0, 100), density=True)
    ax.set_xlim(20, 100)
    ax.set_ylim(0, 0.042)
    ax.set_xlabel('score')
    ax.set_ylabel('relative frequency')
    ax.set_title('Population Histogram (scores)')
    plt.show()

    print(f"Choice 1 from scores: {np.random.choice(scores)}")

    sample = np.random.choice(scores, 10000)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.hist(sample, bins=100, range=(0, 100), density=True)
    ax.set_xlim(20, 100)
    ax.set_ylim(0, 0.042)
    ax.set_xlabel('score')
    ax.set_ylabel('relative frequency')
    ax.set_title('Sample Histogram (n=10000)')
    plt.show()

    sample_means = [np.random.choice(scores, 20).mean()
                    for _ in range(10000)]

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.hist(sample_means, bins=100, range=(0, 100), density=True)
    # 모평균을 세로선으로 표시
    ax.vlines(np.mean(scores), 0, 1, 'gray')
    ax.set_xlim(50, 90)
    ax.set_ylim(0, 0.13)
    ax.set_xlabel('score')
    ax.set_ylabel('relative frequency')
    ax.set_title('Sampling Distribution of Sample Means (n=20, 10000 trials)')
    plt.show()