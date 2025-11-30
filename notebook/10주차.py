import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import comb, factorial


np.set_printoptions(precision=3)


print("--- 6.0 준비 ---")

# 그래프 선의 종류
linestyles = ['-', '--', ':']


def E(X, g=lambda x: x):
    x_set, f = X
    return np.sum([g(x_k) * f(x_k) for x_k in x_set])


def V(X, g=lambda x: x):
    x_set, f = X
    mean = E(X, g)
    return np.sum([(g(x_k) - mean) ** 2 * f(x_k) for x_k in x_set])


def check_prob(X):
    x_set, f = X
    prob = np.array([f(x_k) for x_k in x_set])
    assert np.all(prob >= 0), 'minus probability'
    prob_sum = np.round(np.sum(prob), 6)
    assert prob_sum == 1, f'sum of probability{prob_sum}'
    print(f'expected value {E(X):.4f}')
    print(f'variance {(V(X)):.4f}')


def plot_prob(X):
    x_set, f = X
    prob = np.array([f(x_k) for x_k in x_set])

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.bar(x_set, prob, label='prob')
    ax.vlines(E(X), 0, 1, label='mean', colors='black')  # PPT의 ax.vlines 색상 등 조정
    ax.set_xticks(np.append(x_set, E(X)))
    ax.set_ylim(0, prob.max() * 1.2)
    ax.legend()
    plt.title("Probability Distribution")
    plt.show()


# ==========================================
# 6.1 베르누이 분포
# ==========================================
print("\n--- 6.1 베르누이 분포 ---")


def Bern(p):
    x_set = np.array([0, 1])

    def f(x):
        if x in x_set:
            return p ** x * (1 - p) ** (1 - x)
        else:
            return 0

    return x_set, f


# 사용자 정의 함수 실습
p = 0.3
X = Bern(p)
print(f"User Defined Bern({p}) 확인:")
check_prob(X)
plot_prob(X)

# Scipy.stats 실습
print(f"\nScipy Bern({p}) 확인:")
rv = stats.bernoulli(p)
print(f"pmf(0), pmf(1): {rv.pmf(0):.3f}, {rv.pmf(1):.3f}")
print(f"cdf([0, 1]): {rv.cdf([0, 1])}")
print(f"mean, var: {rv.mean():.3f}, {rv.var():.3f}")

# ==========================================
# 6.2 이항분포

print("\n--- 6.2 이항분포 ---")


def Bin(n, p):
    x_set = np.arange(n + 1)

    def f(x):
        if x in x_set:
            return comb(n, x) * p ** x * (1 - p) ** (n - x)
        else:
            return 0

    return x_set, f


# 사용자 정의 함수 실습
n = 10
p = 0.3
X = Bin(n, p)
print(f"User Defined Bin({n}, {p}) 확인:")
check_prob(X)
plot_prob(X)

# Scipy.stats 그래프 실습 (다양한 p값)
print("\nScipy Binom Plotting...")
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)

x_set = np.arange(n + 1)
for p, ls in zip([0.3, 0.5, 0.7], linestyles):
    rv = stats.binom(n, p)
    ax.plot(x_set, rv.pmf(x_set),
            label=f'p:{p}', ls=ls, color='gray')
ax.set_xticks(x_set)
ax.legend()
plt.title("Binomial Distribution (Scipy)")
plt.show()

# ==========================================
# 6.3 기하분포

print("\n--- 6.3 기하분포 ---")


def Ge(p):
    x_set = np.arange(1, 30)  # PPT 코드상 1~29 범위

    def f(x):
        if x in x_set:
            return p * (1 - p) ** (x - 1)
        else:
            return 0

    return x_set, f


# 사용자 정의 함수 실습
p = 0.5
X = Ge(p)
print(f"User Defined Ge({p}) 확인:")
check_prob(X)
plot_prob(X)

# Scipy.stats 그래프 실습 (다양한 p값)
print("\nScipy Geom Plotting...")
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)

x_set = np.arange(1, 15)
for p, ls in zip([0.2, 0.5, 0.8], linestyles):
    rv = stats.geom(p)
    ax.plot(x_set, rv.pmf(x_set),
            label=f'p:{p}', ls=ls, color='gray')
ax.set_xticks(x_set)
ax.legend()
plt.title("Geometric Distribution (Scipy)")
plt.show()

# ==========================================
# 6.4 포아송 분포

print("\n--- 6.4 포아송 분포 ---")


def Poi(lam):
    x_set = np.arange(20)

    def f(x):
        if x in x_set:
            return np.power(lam, x) / factorial(x) * np.exp(-lam)
        else:
            return 0

    return x_set, f


# 사용자 정의 함수 실습
lam = 3
X = Poi(lam)
print(f"User Defined Poi({lam}) 확인:")
check_prob(X)
plot_prob(X)

# Scipy.stats 그래프 실습 (다양한 lambda 값)
print("\nScipy Poisson Plotting...")
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)

x_set = np.arange(20)
for lam, ls in zip([3, 5, 8], linestyles):
    rv = stats.poisson(lam)
    ax.plot(x_set, rv.pmf(x_set),
            label=f'lam:{lam}', ls=ls, color='gray')
ax.set_xticks(x_set)
ax.legend()
plt.title("Poisson Distribution (Scipy)")
plt.show()