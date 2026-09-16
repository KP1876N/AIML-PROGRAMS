import numpy as np
import matplotlib.pyplot as plt

def lowess(x, y, f=0.25):
    n = len(x)
    r = int(np.ceil(f * n))
    result = np.zeros(n)

    for i in range(n):
        dist = np.abs(x - x[i])
        h = np.sort(dist)[r]

        w = (1 - (dist / h) ** 3) ** 3
        w[dist > h] = 0

        A = np.array([
            [np.sum(w), np.sum(w*x)],
            [np.sum(w*x), np.sum(w*x*x)]
        ])

        b = np.array([
            np.sum(w*y),
            np.sum(w*x*y)
        ])

        beta = np.linalg.solve(A, b)
        result[i] = beta[0] + beta[1]*x[i]

    return result


np.random.seed(17)

x = np.linspace(0, 10, 100)
y = 2.3*x + 4 + np.random.randn(100)*1.5

smooth = lowess(x, y)

plt.scatter(x, y, label="Data")
plt.plot(x, smooth, label="LOWESS")
plt.plot(x, np.polyval(np.polyfit(x, y, 1), x),
         label="Linear Regression")

plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()
