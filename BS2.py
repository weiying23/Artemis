# BLACK SCHOLES SOLUTION WITH NUMERICAL PARTIAL DIFFERENTIAL EQUATIONS

# This is applicable to an european put option

# environment setting, before any codes
import numpy as np
import scipy.linalg as slinalg
import matplotlib.pyplot as plt
import numpy as np

import numpy.polynomial.legendre as npleg
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
from IPython.display import clear_output, display


# Data Initialization

class putOption:

    def __init__(self):
        self.K = 110  # strike
        self.S0 = 118.5  # Initial Stock Price
        self.r = 0.01  # Risk Free Interest Rate
        self.sig = 0.16  # Volatility
        self.Smax = 250  # Largest value of the underlying asset
        self.Tmax = 0.632876712  # Time to Expiration in years
        self.var = self.sig * self.sig
        self.N = 10  # Number of time steps
        self.M = 10  # Number of stock steps
        self.Time = (self.generate_grid_y(0, self.Tmax, self.N - 2))[::-1]
        self.Stock = self.generate_grid_y(0, self.Smax, self.M - 2)
        self.deltaS = self.Stock[1] - self.Stock[0]
        self.deltaT = self.Time[1] - self.Time[0]

        # We generate a uniform grid

    def generate_grid_y(self, left, right, m):
        h = (right - left) / (m + 1)
        y = np.zeros(m + 2)
        for j in range(m + 2):
            y[j] = j * h
        return y

    # We set boundary conditons

    def gL(self, t):
        return self.K * np.exp(-self.r * t)

    def gR(self, t):
        return 0

    def gB(self, s):
        return max(self.K - s, 0)

    def gT(self, s):
        return max(self.K - s, 0)

    # coefficients

    def alpha(self, i):
        return 0.5 * self.deltaT * (self.r * (i + 1) - self.var * (i + 1) ** 2)

    def beta(self, i):
        return 1 + (self.var * (i + 1) ** 2 + self.r) * self.deltaT

    def gamma(self, i):
        return -0.5 * self.deltaT * (self.r * (i + 1) + self.var * (i + 1) ** 2)

    # Assemble Matrix

    def assemble_matrix_A(self):
        self.A = np.zeros((self.M - 2, self.M - 2))
        for i in range(self.M - 2):
            self.A[i, i] = self.beta(i)
            if (i - 1 >= 0):
                self.A[i, i - 1] = self.alpha(i)

            if (i + 1 <= self.M - 3):
                self.A[i, i + 1] = self.gamma(i)

        return self.A

    # Matrix of put Price
    def putMatrix(self):
        # Matrix A
        self.A = self.assemble_matrix_A()

        # Matrix of put Price
        self.P = np.zeros((self.N, self.M))

        for j in range(self.M):
            self.P[0, j] = self.gT((j + 1) * self.deltaS)
            self.P[self.N - 1, j] = self.gB((j + 1) * self.deltaS)

        for i in range(self.N):
            self.P[i, self.M - 1] = self.gR((i + 1) * self.deltaT)
            self.P[i, 0] = self.gL((i + 1) * self.deltaT)

        # Matrix of the two edges conditions

        self.C = np.zeros((self.N - 1, self.M - 2))
        for i in range(self.N - 1):
            self.C[i][0] = self.alpha(1) * self.gL((i + 1) * self.deltaT)
            self.C[i][self.M - 3] = self.gamma(self.M - 1) * self.gR((i + 1) * self.deltaT)

        # Now we fill the matrix of prices by moving backward in the time

        for k in range(self.N):
            self.curvec = self.P[self.N - 1 - k][1:self.M - 1] - self.C[self.N - 2 - k]
            self.vec = np.linalg.solve(self.A, self.curvec)
            self.P[self.N - 1 - k - 1][1:self.M - 1] = self.vec
        return self.P


# Quick Visualization

put = putOption()
P = put.putMatrix()
fig = plt.figure()
for i in range(put.N):
    plt.plot(put.Stock, P[i])
plt.show()

# For better visualization, use plot_surface function from mpl_toolkits.mplot3d library