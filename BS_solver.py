# ARTEMIS BS-Equation solver
# European call/put options

# import packages
import matplotlib.pyplot as plt
import numpy as np
import time

# Data Initialization

process_start = time.process_time()


class Option():

    def __init__(self):
        self.K = 100  # strike
        self.S0 = 120  # Initial Stock Price
        self.r = 0.1  # Risk Free Interest Rate
        self.sig = 0.16  # Volatility
        self.Smax = 150  # Largest value of the underlying asset
        self.Tmax = 0.25   # Time to Expiration in years
        self.var = self.sig * self.sig
        self.N = 2000  # Number of time steps
        self.M = 200  # Number of stock steps
        self.Time = (self.generate_grid_y(0, self.Tmax, self.N - 2))[::-1]
        self.Stock = self.generate_grid_y(0, self.Smax, self.M - 2)
        self.deltaS = self.Stock[1] - self.Stock[0]
        self.deltaT = -self.Time[1] + self.Time[0] # delta T should be poitive if the BS is inversed for tau=T-t
        self.RKstep = [0.25, 0.3333, 0.5, 1]

        print(self.deltaS)
        print(self.deltaT)
        print('TimeStep limit:',1/(self.var*(self.M-1)+0.5*self.r))

        # We generate a uniform grid

    def generate_grid_y(self, left, right, m):
        h = (right - left) / (m + 1)
        y = np.zeros(m + 2)
        for j in range(m + 2):
            y[j] = j * h
        return y

    # We set boundary conditions

    def gL(self, t, opt):
        if opt == 'put':
            return self.K * np.exp(-self.r * t)
        elif opt == 'call':
            return 0
    def gR(self, t, opt):
        if opt == 'put':
            return 0
        elif opt == 'call':
            return self.Smax-self.K * np.exp(-self.r * t)
    def gB(self, s, opt):
        if opt == 'put':
            return max(self.K - s, 0)
        elif opt == 'call':
            return max(s - self.K, 0)

    # central DIFFERENCE stepping
    def stepping(self,opt):
        print('Option:',opt)
        self.S = self.Stock
        self.P = np.zeros((self.N, self.M))
    # apply boundary condition s
        for ii in range(self.M):
            self.P[0, ii] = self.gB((ii) * self.deltaS, opt)
        for jj in range(self.N):
            self.P[jj, self.M - 1] = self.gR((jj) * self.deltaT, opt)
            self.P[jj, 0] = self.gL((jj) * self.deltaT, opt)
    # start time stepping
        # RK-4 steps
        for i in range(4):
            coeff = self.RKstep[i]
            ddt = coeff * self.deltaT
            for j in range(0, self.N-1):
               for i in range(1, self.M-1):
                  rhs = 0.5 * self.var * self.S[i]**2 * (self.P[j,i+1]-2*self.P[j,i]+self.P[j,i-1])/(self.deltaS**2) + self.r * self.S[i] * (self.P[j,i+1]-self.P[j,i-1])/2/self.deltaS - self.r * self.P[j,i]
                  self.P[j+1,i] = rhs * ddt + self.P[j,i]
            return self.P



sony = Option()
alfa = sony.stepping('call')
fig = plt.figure()
for i in range(sony.N):
    plt.plot(sony.Stock, alfa[i])
plt.show()

end = time.time()
process_end = time.process_time()
print("Proc Time:", process_end - process_start)