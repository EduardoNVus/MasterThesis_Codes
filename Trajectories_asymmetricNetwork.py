import numpy as np
import matplotlib.pyplot as plt
from numba import njit

@njit
def func(X,t,params):
    g = params[0]  
    N = params[1]  
    W = params[2] 

    derivadaX = np.zeros(N)

    derivadaX=-X+np.tanh(g*np.dot(W,X))

    return derivadaX

@njit
def RK4(N, x0, h, tmax, params): 
    # Rk4 method with parameters:
    # N->Dimensionality of the system
    # x0->Initial condition
    # h->Time step
    # tmax->Maximum time for the simulation
    # params->Parameters for the function of evaluation
    
    T=int(tmax/h)+1 #Calculate the dimension of the time vector. It must be an integer
    t=np.linspace(0,tmax,T) #Calculate the vector of times. It is a vector of linearly spaced steps with size h from 0 to T
    X=np.zeros((N,T)) #create the vector of the solution. It has size N (dimension of the system) times T, the number of 
    #times where we will evaluate the solution. We initialize it with zeros.
    X[:,0]=x0 #initialize the system
    h2=h/2 #half the step
    h6=h/6 #a sixth of the step
    
    for tt in range(1,T): #loop on times, from 1 to T (the time 0 has already been set)
        k1=func(X[:,tt-1], t[tt-1], params)
        k2=func(X[:,tt-1]+h2*k1, t[tt-1]+h2, params)
        k3=func(X[:,tt-1]+h2*k2, t[tt-1]+h2, params)
        k4=func(X[:,tt-1]+h*k3, t[tt-1]+h, params)
        X[:,tt]=X[:,tt-1]+h6*(k1+2*k2+2*k3+k4) #update the vector X according to the method
        
    return t,X,T
# --------------------------------------------------
N = 100                 # System size
Xo = np.random.normal(0,0.5,size = N)    # Iniqual condition
g = 0.5     # Control global parameter
J = 3       # Variance parameter
J0 = 1      # Mean parameter

tmax = 125 # Time of simulation
h = 0.1 # Time step
# --------------------------------------
W = J/np.sqrt(N)*np.random.normal(0,1,(N,N)) + J0/N*np.ones((N,N)) # Synaptic matrix
t, X, T = RK4(N,Xo,h,tmax,(g,N,W)) # Trajectory of the system
# Mean activity
X_average = np.zeros(T)
X2_average = np.zeros(T)
for i in range(T):
    X_average[i] = np.mean(X[:,i]) # Mean activity M (t)
    X2_average[i] = np.mean(X[:,i]**2) # Mean-squared activity q (t)
    
#Plots
plt.plot(t, X_average,"r-",label = "M (t)")
plt.plot(t, X2_average,"b-",label = "q (t)")
for i in range(N-50):
    plt.plot(t, X[i,:], "grey", alpha=0.1)
plt.xlabel("Time",fontsize=15)
plt.ylabel(r"Activity $x_{i} (t)$", fontsize=15)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.ylim(-1,1)
plt.legend(fontsize=13)
plt.show()
