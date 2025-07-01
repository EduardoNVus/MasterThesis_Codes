import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import datetime
from scipy.integrate import quad
from scipy.stats import norm
from tqdm import tqdm

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

    if np.any(np.isnan(X)):
        print("¡HAY UN Nan!")
        
    return t,X,T

@njit
def Tmap(IgJ_array,JoJ_array,g,N,Xo,h,tmax,to,tm):
        
        Z = np.random.normal(0,1,size = (N,N)) # Quenched disorder
        M_matrix = np.zeros((len(IgJ_array),len(JoJ_array))) # Grid where the values of M are stored
        C_matrix = np.zeros((len(IgJ_array),len(JoJ_array))) # Grid where the values of q are stored
        for JoJ in JoJ_array: # -------------------------------------------------------------------------------
            for IgJ in IgJ_array: # 
                J = 1/(g*IgJ)
                Jo = JoJ*J
                W= J/np.sqrt(N)*Z + Jo/N*np.ones((N,N)) #Quenched disorder
                # ---------------------------------------------------------------------------------------------
                # We solve the system
                t_sol,X_sol, T = RK4(N,Xo,h,tmax,(g,N,W)) # Trajectory of the system
                X_average = np.zeros(T)
                X_average2 = np.zeros(T)
                for i in range(T):
                    X_average[i] = np.mean(X_sol[:,i]) # Mean activity M (t)
                    X_average2[i] = np.mean(X_sol[:,i]**2) # Mean-squared activity q (t)
                            
                Msini = np.where(t_sol >= to)[0][0]        # Index of the instant to
                Msfin = np.where(t_sol >= (to + tm))[0][0] # Index of the instant to + tm  
                
                Ms = np.mean(X_average[Msini:Msfin]) # We calculate the average of the activity M in the interval [to, to + tm]
                Cs =np.mean(X_average2[Msini:Msfin])
                # --------------------------------------------------------------------------------------------
                M_matrix[int(np.where(IgJ_array == IgJ)[0][0])][int(np.where(JoJ_array == JoJ)[0][0])] = Ms
                C_matrix[int(np.where(IgJ_array == IgJ)[0][0])][int(np.where(JoJ_array == JoJ)[0][0])] = Cs
        return M_matrix,C_matrix

@njit
def Average_Tmap(s,IgJ_array,JoJ_array,g,N,Xo,h,tmax,to,tm):
    average_M = np.zeros((len(IgJ_array),len(JoJ_array)))
    average_C = np.zeros((len(IgJ_array),len(JoJ_array)))
    s0 = 0
    for realizations in range(s):
        M,C = Tmap(IgJ_array,JoJ_array,g,N,Xo,h,tmax,to,tm)
        average_M += np.abs(M)
        average_C += C

        s0 += 1
        print(f"Step {s0}/{s}")
    
    average_M = average_M/s
    average_C = average_C/s
    return average_M,average_C
##############################################################
# Control parameters
g = 1     
IgJ_max = 3 
IgJ_min = 0.1 
JoJ_max = 3   
JoJ_min = 0.1 
hIgJ = 0.05
hJoJ = 0.05

N = 100                 
Xo = np.random.normal(0,1,size = N)    
# Simulation parameters
s = 100  # Number of realizations  
tmax = 125 # Time of simulation
h = 0.1 # Time step
to = 100 # Initial time from which the temporal average is taken
tm =20 # Time interval for the temporal average
# ---------------------------------------------
IgJ_len = int((IgJ_max-IgJ_min)/hIgJ) + 1
JoJ_len = int((JoJ_max-JoJ_min)/hJoJ) + 1
IgJ_array = np.linspace(IgJ_min,IgJ_max,IgJ_len) 
JoJ_array = np.linspace(JoJ_min,JoJ_max,JoJ_len) 
#######################################################################
# Execution
M,C = Average_Tmap(s,IgJ_array,JoJ_array,g,N,Xo,h,tmax,to,tm)
    # Save results
date = datetime.datetime.now()
comment = "Simulación Firing-Rate con señal aleatoria, sin campo externo , matriz sináptica totalmente asimétrica (ganma = 0)"
np.savetxt(f"M",M,delimiter = ",",header = f"N = {N}, g = {g},1/gJ = ({IgJ_array[0]},{IgJ_array[-1]}), Jo/J = ({JoJ_array[0]},{JoJ_array[-1]}), h1/gJ = {hIgJ}| s = {s}, h = {h}, tmax = {tmax}, to = {to},tm = {tm} | Date: {str(date)} \n Comentario: {comment}")
np.savetxt(f"C",C, delimiter = ",",header = f"N = {N}, g = {g}, 1/gJ = ({IgJ_array[0]},{IgJ_array[-1]}), Jo/J = ({JoJ_array[0]},{JoJ_array[-1]}), hJo/J = {hJoJ}| s = {s}, h = {h}, tmax = {tmax}, to = {to},tm = {tm} | Date: {str(date)} \n Comentario: {comment}")
#######################################################################
# Load results
M_read = np.loadtxt("M",delimiter = ",", comments="#")
C_read = np.loadtxt("C",delimiter = ",", comments="#")
# -----------------------------------------------------------
# Theoretical bifurcation line between the chaotic phase and the persistent phase
def f(q, k):
    if q < 1e-10:
        return 0.0  
    integrand = lambda z: (np.tanh(np.sqrt(q) * z / k))**2 * norm.pdf(z)
    result, _ = quad(integrand, -10, 10, epsabs=1e-9)
    return result

# Function to solve for q using fixed-point iteration
def solve_q(k, tol=1e-6, max_iter=1000):
    q = 0.01  # valor inicial pequeño
    for _ in range(max_iter):
        q_new = f(q, k)
        if abs(q_new - q) < tol:
            return q_new
        q = q_new
    return q  

# Barrido en k
k_values = np.linspace(0.1, 1.0, 100)
q_values = [solve_q(k) for k in tqdm(k_values)]
# --------------------------------------------
# Plots
z = 30 #Number of levels in the contour plot
IgJ_bif = k_values
JoJ_bif = IgJ_bif/(1-np.array(q_values))

figM, axM = plt.subplots(figsize = (8,6))

contour = axM.contourf(JoJ_array,IgJ_array,M_read,levels = z,cmap = "jet")
axM.plot([1,3],[1,3],color = "white",linestyle = "-",linewidth = 4)
axM.plot([0.1,1],[1,1],color = "white",linestyle = "-",linewidth = 4)
axM.plot(JoJ_bif,IgJ_bif,color = "white",linestyle = "-",linewidth = 4)

plt.xlabel("Jo/J",fontsize = 15)
plt.xticks(fontsize = 13)
plt.ylabel("1/gJ",fontsize = 15)
plt.yticks(fontsize = 13)
plt.title("M",fontsize = 15)
cbarM = figM.colorbar(contour)
cbarM.ax.tick_params(labelsize=13)
cbarM.set_ticks([0, 0.25, 0.5, 0.75, 1])

figC, axC = plt.subplots(figsize = (8,6))

contour = axC.contourf(JoJ_array,IgJ_array,C_read,levels = z,cmap = "jet")
axC.plot([1,3],[1,3],color = "white",linestyle = "-",linewidth = 4)
axC.plot([0.1,1],[1,1],color = "white",linestyle = "-",linewidth = 4)
axC.plot(JoJ_bif,IgJ_bif,color = "white",linestyle = "-",linewidth = 4)

plt.xlabel("Jo/J", fontsize = 15)
plt.ylabel("1/gJ",fontsize = 15)
plt.xticks(fontsize = 13)
plt.yticks(fontsize = 13)
plt.title("q",fontsize = 15)
cbarC = figM.colorbar(contour)
cbarC.ax.tick_params(labelsize=13)
cbarC.set_ticks([0, 0.25, 0.5, 0.75, 1])

plt.show()