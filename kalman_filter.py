import numpy as np
import matplotlib.pyplot as plt


class Point:
    def __init__(self, acc=0, d_acc=0) -> None:
        self.loc = np.random.randint(-10, 10)
        self.velocity = np.random.randint(1, 10)
        self.acc = acc
        self.d_acc = d_acc
        # self.acc = np.random.randint(-1,3)
        # self.d_acc = np.random.randint(-2,2)
        self.time = 0
        self.history = []

    def __call__(self, steps=1):
        # s = 1/2 at^2 + v0t
        for _ in range(steps):
            self.history.append(self.loc)
            self.loc += (1/2)*self.acc*(steps**2) + self.velocity*(steps)
            self.velocity += self.acc
            self.acc += self.d_acc
            self.time += 1

    def __str__(self) -> str:
        return(f'TIME STEP: {self.time}\nloc: {self.loc:.2f}, vel: {self.velocity:.2f} \nacc: {self.acc:.2f}, acc change: {self.d_acc:.2f}')


class Filters:
    def __init__(self, observations=[]) -> None:
        self.obs = observations

    def alphaFilter(self):
        N = len(self.obs)
        x_pred = np.zeros(len(self.obs))
        x_pred[0] = self.obs[0]
        for n in range(1, N):
            alpha = 1/(n)
            x_pred[n] = x_pred[n-1] + alpha*(self.obs[n - 1] - x_pred[n-1])
        return x_pred

    def alphaBetaFilter(self, alpha=.2, beta=.1):
        N = len(self.obs)
        x_pred = np.zeros(len(self.obs))
        v_pred = np.zeros_like(x_pred)
        x_pred[0] = self.obs[0]  # is given
        v_pred[0] = (self.obs[1] - self.obs[0])  # is given
        for n in range(1, N):
            x_temp = x_pred[n-1] + alpha*(self.obs[n - 1] - x_pred[n-1])
            v_pred[n] = v_pred[n-1] + beta*(self.obs[n - 1] - x_pred[n-1])
            x_pred[n] = x_temp + v_pred[n]
        return x_pred

    def alphaBetaGammaFilter(self, alpha=.5, beta=.4, gamma = .1):
        N = len(self.obs)
        x_pred = np.zeros(len(self.obs))
        v_pred = np.zeros_like(x_pred)
        a_pred = np.zeros_like(x_pred)
        x_pred[0] = self.obs[0]  # is given
        v_pred[0] = (self.obs[1] - self.obs[0])  # is given
        a_pred[0] = 0  # is given
        for n in range(1, N):
            x_temp = x_pred[n-1] + alpha*(self.obs[n - 1] - x_pred[n-1])
            v_pred[n] = v_pred[n-1] + beta*(self.obs[n - 1] - x_pred[n-1])
            a_pred[n] = a_pred[n-1] + gamma*(self.obs[n - 1] - x_pred[n-1])*2
            x_pred[n] = x_temp + v_pred[n] + a_pred[n]*.5
        return x_pred


p = Point(acc=.5, d_acc=-.01)
p(100)
a = p.history

#a = np.array([1030, 989, 1017, 1009, 1013, 979, 1008, 1042, 1012,1011])
f = Filters(a)
x = f.alphaFilter()
y = f.alphaBetaFilter()
z = f.alphaBetaGammaFilter()
N = list(range(0, len(a)))
plt.plot(N, a)
plt.plot(x)
plt.plot(y)
plt.plot(z)
plt.legend(['true values', 'alpha predictet', 'alpha - beta ', 'a-b-g'])
plt.show()
