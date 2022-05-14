from pprint import pprint
from matplotlib import projections
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


class MultiDimPoint:
    def __init__(self, acc=[0, 0], d_acc=[0, 0]) -> None:
        self.loc = np.random.randint(-10, 10, 2)
        self.velocity = np.random.randint(-10, 10, 2)
        self.acc = acc
        self.d_acc = d_acc
        # self.acc = np.random.randint(-1,3)
        # self.d_acc = np.random.randint(-2,2)
        self.time = 0
        self.history = []

    def __call__(self, steps=1, noiseWeight = 1):
        noise = (sum(self.velocity)/2)*noiseWeight/100
        # s = 1/2 at^2 + v0t
        for _ in range(steps):
            self.history.append(list(self.loc))
            self.history[-1][0] += np.random.random()*noise
            self.history[-1][1] += np.random.random()*noise
            self.time += 1
            for dim in range(2):
                self.loc[dim] += round((1/2)*self.acc[dim]
                                       * (steps**2) + self.velocity[dim]*(steps), 2)
                self.velocity[dim] += round(self.acc[dim], 2)
                self.acc[dim] += round(self.d_acc[dim], 2)

    def __str__(self) -> str:
        return(f'TIME STEP: {self.time}\nloc: {self.loc}, vel: {self.velocity} \nacc: {self.acc}, acc change: {self.d_acc}')

    def plotRoute(self):
        ax = plt.axes()
        a = np.array(self.history).T
        ax.plot(*a, 'gray')
        plt.show()


class OneDimFilters:
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

    def alphaBetaGammaFilter(self, alpha=.5, beta=.4, gamma=.1):
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

    def kalmanFilter(self, r=5, std=15, q=.01):
        N = len(self.obs)
        X = np.zeros(N)
        K = np.zeros(N)
        P = np.zeros(N)
        X[0] = self.obs[0]  # is given
        P[0] = std  # is given
        for n in range(1, N):
            K[n] = P[n - 1]/(P[n - 1] + r)
            X[n] = X[n - 1] + K[n]*(self.obs[n] - X[n - 1])
            P[n] = (1 - K[n])*P[n - 1] + q
        return X


class KalmanFilter:
    def __init__(self, observations=[]) -> None:
        self.X = np.array(observations)
        self.pred = []
        
    def makeQMatrix(self, q, dims=2, timeStep=1):
        matrix = [[4, 2, 2], [2, 1, 1], [2, 1, 1]]
        N = 3
        Q = np.zeros(shape=(N*dims, N*dims))
        for y in range(N):
            for x in range(N):
                Q[y][x] = round(timeStep/matrix[y][x], 3)
                Q[y + 3][x + 3] = Q[y][x]
        return Q*q

    def makeFMatrix(self, dims=2, timeStep=1):
        N = 3
        matrix = [[1, 1, .5], [0, 1, 1], [0, 0, 1]]
        F = np.zeros(shape=(N*dims, N*dims))
        for y in range(N):
            for x in range(N):
                F[y][x] = round(timeStep*(matrix[y][x]), 3)
                F[y + 3][x + 3] = F[y][x]
        return F

    def makeHMatrix(self, dims=2):
        return np.array([[1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]])

    def makeRMatrix(self, variance):
        return np.array([[variance[0], 0], [0, variance[-1]]])

    
    def multiDimKalman(self, r=5, x_var=(.2, .1), q=.01, acc_var=.5):
        '''
        `acc_var` - random variance in acceleration
        '''
        N = len(self.X)
        Q = self.makeQMatrix(q)
        F = self.makeFMatrix()
        R = self.makeRMatrix(x_var)
        x_pred = np.zeros((N, 6, 2))
        x_pred[0,0,0] = self.X[0,0]
        x_pred[0,3,1] = self.X[0,1]
        K = np.zeros((N, 6, 2))
        P = np.zeros((N, 6, 6))
        H = self.makeHMatrix()
        for n in range(1, N):
            P[n] = np.dot(np.dot(F, P[n - 1]),F.T) + Q
            K[n] = np.dot(np.dot(P[n-1], H.T), np.linalg.inv(np.dot(np.dot(H,P[n - 1]),H.T)  + R))
            x_pred[n] = x_pred[n - 1] + np.dot(K[n],(self.X[n] - np.dot(H,x_pred[n - 1])))
        self.pred = x_pred
        return x_pred
    
    def plot(self):
        time = list(range(len(self.X)))
        plt.figure(figsize=(10,10))
        ax = plt.axes(projection = '3d')
        a = np.array(self.X).T
        ax.plot3D(*a,time, 'gray')
        x = self.pred[:,0,0]
        y = self.pred[:,3,1]
        ax.scatter3D(x, y, time)
        plt.legend(['true', 'predicted'])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        plt.show()

