import numpy as np
import matplotlib.pyplot as plt

# %precision 3
# %matplotlib inline

# ## 1차원 이산형 확률분포
# ### 1차원 이산확률분포의 정의

print("--- [1차원 이산형 확률분포] ---")
x_set = np.array([1, 2, 3, 4, 5, 6])

def f(x):
    if x in x_set:
        return x / 21
    else:
        return 0

X = [x_set, f]

# 확률 p_k를 구한다
prob = np.array([f(x_k) for x_k in x_set])
# x_k와 p_k의 대응을 사전식으로 표시
print(dict(zip(x_set, prob)))

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
ax.bar(x_set, prob)
ax.set_xlabel('value')
ax.set_ylabel('probability')
ax.set_title('1D Marginal Probability Distribution')
plt.show()

print(f"모든 확률이 0 이상인가? {np.all(prob >= 0)}")
print(f"확률의 총합: {np.sum(prob)}")

def F(x):
    return np.sum([f(x_k) for x_k in x_set if x_k <= x])

print(f"F(3) = {F(3)}")

y_set = np.array([2 * x_k + 3 for x_k in x_set])
prob = np.array([f(x_k) for x_k in x_set])
print(dict(zip(y_set, prob)))

# ### 1차원 이산형 확률변수의 지표
# #### 평균
print("\n--- [1차원 이산형 확률변수의 지표] ---")
print(f"E(X) (직접 계산): {np.sum([x_k * f(x_k) for x_k in x_set])}")

def E(X, g=lambda x: x):
    x_set, f = X
    return np.sum([g(x_k) * f(x_k) for x_k in x_set])

print(f"E(X) (함수): {E(X)}")
print(f"E(2X + 3): {E(X, g=lambda x: 2*x + 3)}")
print(f"2*E(X) + 3: {2 * E(X) + 3}")

# #### 분산
mean = E(X)
print(f"V(X) (직접 계산): {np.sum([(x_k-mean)**2 * f(x_k) for x_k in x_set])}")

def V(X, g=lambda x: x):
    x_set, f = X
    mean = E(X, g)
    return np.sum([(g(x_k)-mean)**2 * f(x_k) for x_k in x_set])

print(f"V(X) (함수): {V(X)}")
print(f"V(2X + 3): {V(X, lambda x: 2*x + 3)}")
print(f"2**2 * V(X): {2**2 * V(X)}")