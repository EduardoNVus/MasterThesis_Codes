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
    for i in range(N): # Señal externa
        h[i] = np.cos(omegas[i]*t + deltas[i])
    func = np.tanh(Jx) - x + I*h
    return func

@njit
def measure_synchrony_numba_mean_std(X, Y, N, tol=1e-6):
    """
    Compute synchrony order parameters for excitatory and inhibitory signals.
    Uses np.mean and np.std (with no axis argument) on the final 1D arrays.
   
    Parameters:
        X, Y : 2D arrays containing the signal components.
                The excitatory data is taken as the first N rows, and
                the inhibitory data from row N onward.
        N   : number of excitatory rows.
        tol : tolerance for checking activity (if the mean squared
              excitatory value is below tol, the function returns zeros).
             
    Returns:
        R_E, R_I, R_total, Ce_E, Ce_I, Ce_total
          – respectively, the mean and standard deviation of the modulus
             of the excitatory, inhibitory, and total (combined) complex order parameters.
    """
    n_rows, n_cols = X.shape
    n_inhib = n_rows - N

    # Slice excitatory and inhibitory parts
    E_x = X
    E_y = Y

    # Check for no net excitatory activity using 1D np.mean
    if np.mean(E_x**2) < tol or np.mean(E_y**2) < tol:
        return 0.0, 0.0

    # Compute the excitatory order parameter per column:
    # For each column, calculate the average of exp(1j*theta) over the excitatory rows.
    Z_E = np.empty(n_cols, dtype=np.complex128)
    for j in range(n_cols):
        s = 0.0 + 0.0j
        for i in range(N):
            theta = np.arctan2(E_y[i, j], E_x[i, j])
            # Reconstruct the complex exponential using cosine and sine.
            s += complex(np.cos(theta), np.sin(theta))
        Z_E[j] = s / N


    # Compute the absolute values (the modulus) per column.
    absE = np.empty(n_cols, dtype=np.float64)

    for j in range(n_cols):
        absE[j] = np.abs(Z_E[j]) 


    # Use np.mean and np.std on the 1D arrays (axis argument is not used)
    R_E      = np.mean(absE) # TODO Media
    Ce_E     = np.std(absE)  # TODO Desviación estándar


    return R_E, Ce_E


@njit
def synaptic(J, J0, N, g):
    #Given a J, g and a J0 and a size N, gives the matrix of Gaussian entries symmetric
    W=(J/np.sqrt(N))*np.random.normal(0,1,(N,N)) #create random gaussian connectivity matrix with sigma=J/np.sqrt(N)
    M=J0/N*np.ones((N,N)) #Create matrix of means
    return g*(M+W)
#####################################################
# Parameters____________________________________
tmax = 250
trelaj = 100
h = 0.1
T = int(tmax/h)+1
N = 100

M = 20                  # Number of realizations
I = 1                   # Amplitude of the external signal
fixed_g = 2             # Global coupling strength
deltas = np.zeros(N)    # Phases
#_______________________________________________

J  = 1
J0 = J * np.linspace(0.01, 2.5, 20)
omegas = np.linspace(0.01, 20, 20)

RE, CeE = np.zeros((len(omegas), len(J0))), np.zeros((len(omegas), len(J0)))

M_ones = 1/N*np.ones((N,N)) #Create matrix of means

#_____________________________________________________
for k in range(M):
    print('------------------------Measurement number m=',k)
    W = synaptic(J, 0, N, 1)
    for i in tqdm(range(len(J0))): 
        J0M = J0[i]*M_ones 
        W_final = fixed_g*(W + J0M) 
        for j in range(len(omegas)): 
            x0 = np.random.rand(N) 
            X, t = trajectory2(trelaj,h,N,W_final,x0, omegas[j]*np.ones(N), deltas, I) 
            X, t = trajectory2(tmax,h,N,W_final,X[:,-1], omegas[j]*np.ones(N), deltas, I)
            Y = np.imag(hilbert(X)) 
           
            R_E,Ce_E = measure_synchrony_numba_mean_std(X, Y, N)

            RE[i,j]+=R_E      
               
            CeE[i,j]+=Ce_E

             
RE/=M
CeE/=M
# Save
np.savetxt("RE_JoJ", RE, delimiter=",", header="RE values for different J0 and omegas")
np.savetxt("CeE_JoJ", CeE, delimiter=",", header="CeE values for different J0 and omegas")
##############################################
# Plot
z = 30 # Number of levels in the plot

RE = np.loadtxt("RE_JoJ", delimiter=",", comments="#")
figM, axM = plt.subplots(figsize = (8,6))
contour = axM.contourf(J0,omegas,RE.T,levels = z,cmap = "jet")

plt.xlabel(r"$J_{0}/J$",fontsize = 15)
plt.ylabel(r"$ \omega $",fontsize = 15)
plt.xticks(fontsize = 13)
plt.yticks(fontsize = 13)
plt.title(r"Order parameter $r$",fontsize = 15)
cbar = figM.colorbar(contour)
cbar.set_ticks([0.1, 0.25, 0.5, 0.75, 0.95])
cbar.ax.tick_params(labelsize=13)

CeE = np.loadtxt("CeE_JoJ", delimiter=",", comments="#")

figM, axM = plt.subplots(figsize = (8,6))
contour = axM.contourf(J0,omegas,CeE.T,levels = z,cmap = "jet")

plt.xlabel(r"$J_{0}/J$",fontsize = 15)
plt.ylabel(r"$\omega$",fontsize = 15)
plt.xticks(fontsize = 13)
plt.yticks(fontsize = 13)
plt.title(r"Standar deviation $\sigma_r$",fontsize = 15)
cbarC = figM.colorbar(contour)
cbarC.set_ticks([0.01, 0.08, 0.15])
cbarC.ax.tick_params(labelsize=13)

plt.show()