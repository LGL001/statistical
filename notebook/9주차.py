import numpy as np
import matplotlib.pyplot as plt

# %precision 3
# %matplotlib inline

# ## 2차원 이산형 확률분포
# ### 2차원 이산형 확률분포의 정의
print("\n--- [2차원 이산형 확률분포] ---")
x_set = np.arange(2, 13)
y_set = np.arange(1, 7)

def f_XY(x, y):
    if 1 <= y <=6 and 1 <= x - y <= 6:
        return y * (x-y) / 441
    else:
        return 0

XY = [x_set, y_set, f_XY]

prob = np.array([[f_XY(x_i, y_j) for y_j in y_set]
                 for x_i in x_set])

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111)

c = ax.pcolor(prob)
ax.set_xticks(np.arange(prob.shape[1]) + 0.5, minor=False)
ax.set_yticks(np.arange(prob.shape[0]) + 0.5, minor=False)
ax.set_xticklabels(np.arange(1, 7), minor=False)
ax.set_yticklabels(np.arange(2, 13), minor=False)
# y축을 내림차순의 숫자가 되게 하여, 위 아래를 역전시킨다
ax.invert_yaxis()
# x축의 눈금을 그래프 위쪽에 표시
ax.xaxis.tick_top()
fig.colorbar(c, ax=ax)
ax.set_title('2D Joint Probability Distribution Heatmap')
plt.show()

print(f"모든 확률이 0 이상인가? {np.all(prob >= 0)}")
print(f"확률의 총합: {np.sum(prob)}")

def f_X(x):
    return np.sum([f_XY(x, y_k) for y_k in y_set])

def f_Y(y):
    return np.sum([f_XY(x_k, y) for x_k in x_set])

X = [x_set, f_X]
Y = [y_set, f_Y]

prob_x = np.array([f_X(x_k) for x_k in x_set])
prob_y = np.array([f_Y(y_k) for y_k in y_set])

fig = plt.figure(figsize=(12, 4))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

ax1.bar(x_set, prob_x)
ax1.set_title('X_marginal probability distribution')
ax1.set_xlabel('X_value')
ax1.set_ylabel('probability')
ax1.set_xticks(x_set)

ax2.bar(y_set, prob_y)
ax2.set_title('Y_marginal probability distribution')
ax2.set_xlabel('Y_value')
ax2.set_ylabel('probability')

plt.show()

# ### 2차원 이산형 확률분포의 지표
print("\n--- [2차원 이산형 확률변수의 지표] ---")
print(f"E(X) (직접 계산): {np.sum([x_i * f_XY(x_i, y_j) for x_i in x_set for y_j in y_set])}")

def E(XY, g):
    x_set, y_set, f_XY = XY
    return np.sum([g(x_i, y_j) * f_XY(x_i, y_j)
                   for x_i in x_set for y_j in y_set])

mean_X = E(XY, lambda x, y: x)
print(f"E(X) (함수): {mean_X}")

mean_Y = E(XY, lambda x, y: y)
print(f"E(Y) (함수): {mean_Y}")

a, b = 2, 3

print(f"E(aX + bY): {E(XY, lambda x, y: a*x + b*y)}")
print(f"a*E(X) + b*E(Y): {a * mean_X + b * mean_Y}")

print(f"V(X) (직접 계산): {np.sum([(x_i-mean_X)**2 * f_XY(x_i, y_j) for x_i in x_set for y_j in y_set])}")

def V(XY, g):
    x_set, y_set, f_XY = XY
    mean = E(XY, g)
    return np.sum([(g(x_i, y_j)-mean)**2 * f_XY(x_i, y_j)
                   for x_i in x_set for y_j in y_set])

var_X = V(XY, g=lambda x, y: x)
print(f"V(X) (함수): {var_X}")

var_Y = V(XY, g=lambda x, y: y)
print(f"V(Y) (함수): {var_Y}")

def Cov(XY):
    x_set, y_set, f_XY = XY
    mean_X = E(XY, lambda x, y: x)
    mean_Y = E(XY, lambda x, y: y)
    return np.sum([(x_i-mean_X) * (y_j-mean_Y) * f_XY(x_i, y_j)
                    for x_i in x_set for y_j in y_set])

cov_xy = Cov(XY)
print(f"Cov(X, Y): {cov_xy}")

print(f"V(aX + bY): {V(XY, lambda x, y: a*x + b*y)}")
print(f"a**2*V(X) + b**2*V(Y) + 2*a*b*Cov(X,Y): {a**2 * var_X + b**2 * var_Y + 2*a*b * cov_xy}")

print(f"상관계수: {cov_xy / np.sqrt(var_X * var_Y)}")