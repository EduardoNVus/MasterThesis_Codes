import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from tqdm import tqdm
from scipy.signal import hilbert

@njit # RK2
def trajectory2(tmax,h,N,J,x0, omegas, deltas, I):
    # tmax is the maximum time,
    # h is the time-step
    # J is the coupling matrix
    # x0 is the initial condition
   
    T=int(tmax/h)+1 #sets the length of the time vector
    t=np.linspace(0,tmax,T) #create time vector
    x=np.zeros((N,len(t))) #create vector for storage of the variables
    x[:,0]=x0
    h2=h/2
   
    for tt in range(1,T): #for each time step...
        k1 = funct(x[:,tt-1],J,N, t[tt-1], omegas, deltas, I)
        k2 = funct(x[:,tt-1]+h2*k1,J,N,t[tt-1]+h2, omegas, deltas, I)
        x[:,tt] = x[:,tt-1] + k2*h
       
    return x,t

@njit 
def funct(x, J, N, t, omegas, deltas, I):
    func=np.zeros(N)
    Jx = np.dot(J, x)
    h = np.zeros(N)
    for i in range(N): 
        h[i] = np.cos(omegas[i]*t + deltas[i])
    func = np.tanh(Jx) - x + I*h
    return func

@njit
def synaptic(J, J0, N, g):
    #Given a J, g and a J0 and a size N, gives the matrix of Gaussian entries symmetric
    W=(J/np.sqrt(N))*np.random.normal(0,1,(N,N)) #create random gaussian connectivity matrix with sigma=J/np.sqrt(N)
    M=J0/N*np.ones((N,N)) #Create matrix of means
    return g*(M+W)
#################################
# Parameters____________________________________
tmax = 100
trelaj = 100
h = 0.1
T = int(tmax/h)+1
N = 100

I = 1                   # Amplitude of the oscillatory input
fixed_g = 2             # Global coupling strength
deltas = np.zeros(N)    # Phases of the oscillatory input
omega = 0.1             # Frequency of the oscillatory input
Jo = 0.0
J = 1.0               
#_______________________________________________
W = synaptic(J, 0, N, 1) 
M_ones = 1/N*np.ones((N,N)) 
W_final = fixed_g*(W + Jo)

x0 = np.random.rand(N) 
X, t = trajectory2(trelaj,h,N,W_final,x0, omega*np.ones(N), deltas, I) # Relajación del sistema
X, t = trajectory2(tmax,h,N,W_final,X[:,-1], omega*np.ones(N), deltas, I) # Evolución relevante
# _____________________________________________
plt.plot(t,X[0,:],label = "Neuron 1",color = "black")
plt.plot(t,X[1,:],label = "Neuron 2",color = "red")
plt.ylabel("x(t)", fontsize=15)
plt.xlabel("t (s)", fontsize=15)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.show()