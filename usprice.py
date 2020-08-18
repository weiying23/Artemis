# ARTEMIS US option solver
# front-fixing

import matplotlib.pyplot as plt
import numpy as np

class Option():

    def __init__(self):
        self.K = 1  # strike/exercise
        self.r = 0.1  # Risk Free Interest Rate
        self.sig = 0.2  # Volatility
        self.Tmax = 1.0   # Time to Expiration in years
        self.xinf = 2     ## unique for US option, a very large value
        self.var = self.sig * self.sig

        self.N = 5000  # Number of time steps
        self.M = 100  # Number of stock steps
        self.Time = (self.generate_grid_y(0, self.Tmax, self.N))
        self.x = self.generate_grid_y(1, self.xinf, self.M)
        #print(self.Time)
        #print(self.x)
        self.P = np.zeros((self.N+2,self.M+2))
        self.S = np.zeros(self.N+2)
        self.deltax = self.x[1] - self.x[0]
        self.deltaT = self.Time[1] - self.Time[0]  # delta T should be poitive if the BS is inversed for tau=T-t

        print(self.deltaT)
        print(1 / (self.var * (self.M - 1) + 0.5 * self.r))

        # We generate a uniform grid

    def generate_grid_y(self, left, right, m):
        h = (right - left) / (m + 1)
        y = np.zeros(m + 2)
        for j in range(m + 2):
            y[j] = j * h + left
        return y

    # We set boundary conditions

    # central DIFFERENCE stepping
    def stepping(self,opt):
        inv_T = 1.0/self.deltaT
        inv_X = 1.0/self.deltax
        A = np.zeros(self.M+1)
        B = np.zeros(self.M+1)
        C = np.zeros(self.M+1)
        D = np.zeros((self.N+2,self.M+1))
        for j in range(self.M+2):
            self.P[self.N+1,j] = 0
        self.S[self.N+1] = self.K
        for n in range(self.N+2)[::-1]:
            self.P[n,self.M+1] = 0
        for j in range(1,self.M+1):
            A[j] = 0.5*self.var*self.x[j]**2*self.deltaT*inv_X**2-self.x[j]*(self.r-inv_T)*self.deltaT*inv_X*0.5
            B[j] = 1-self.var*self.x[j]**2*self.deltaT*inv_X**2-self.r*self.deltaT
            C[j] = 0.5*self.var*self.x[j]**2*self.deltaT*inv_X**2+self.x[j]*(self.r-inv_T)*self.deltaT*inv_X*0.5
        for n in range(self.N+1)[::-1]:
            for j in range(1,self.M+1):
                D[n+1,j] = 0.5*self.x[j]*inv_X*(self.P[n+1,j+1]-self.P[n+1,j-1])/self.S[n+1]
            self.S[n] = (self.K-(A[1]*self.P[n+1,0]+B[1]*self.P[n+1,1]+C[1]*self.P[n+1,2]))/(D[n+1,1]+(1+self.deltax))
            self.P[n,0] = self.K-self.S[n]
            self.P[n,1] = self.K-(1+self.deltax)*self.S[n]
            for j in range(2,self.M+1):
                self.P[n,j]=A[j]*self.P[n+1,j-1]+B[j]*self.P[n+1,j]+C[j]*self.P[n+1,j+1]+D[n+1,j]*self.S[n]
        return self.P


sony = Option()
alfa = sony.stepping("put")
fig = plt.figure()
plt.plot(sony.x, alfa[0])
plt.xlim(xmin = 1.0)
plt.ylim(ymin = 0)
plt.show()