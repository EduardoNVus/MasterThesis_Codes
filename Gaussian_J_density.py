import numpy as np
from numba import njit
import matplotlib.pyplot as plt

@njit
def synaptic(J, J0, N, g):
    #Given a J, g and a J0 and a size N, gives the matrix of Gaussian entries symmetric
    W=(J/np.sqrt(N))*np.random.normal(0,1,(N,N)) #create random gaussian connectivity matrix with sigma=J/np.sqrt(N)
    W=(W+W.transpose())*np.sqrt(2)/2 #Make it symmetrical
    M=J0/N*np.ones((N,N)) #Create matrix of means
    return g*(M+W)

def semicircle_density(x, g=1.0):
    if np.abs(x) > 2 * g:
        return 0.0
    return (1 / (2 * np.pi * g**2)) * np.sqrt(4 * g**2 - x**2)

# -------------------------------
N = 100
J = 1.0
J0 = 3
g = 1.0

s = 500 # Realizations
mode = "s" # The simulation has two modes: "s" for a symmetric matrix (Wigners law) and "a" for an asymmetric matrix (Girko law)
# -------------------------------
eigenlist = np.zeros(round(s*N),dtype = complex) # Store eigenvalues
for i in range(s):
    if mode == "s":
        W = synaptic(J, J0, N, g)  # Generate the synaptic matrix
    elif mode == "a":
        W = np.random.normal(J0/N, J/np.sqrt(N), (N,N))
    else:
        raise ValueError("Mode must be 's' for symmetric or 'a' for asymmetric.")
    
    eigenvalues = np.linalg.eigvals(W)  # Compute eigenvalues
    eigenlist[i * int(N):(i + 1) * int(N)] = eigenvalues  # Store eigenvalues

if mode == "s":
    plt.hist(eigenlist, bins=100, density=True, alpha=0.5, color='blue', edgecolor='black', histtype='stepfilled', label='Empirical')
    if J0 > J:
        plt.axvline(x=(J0 + J**2/J0), color='red', linestyle='--', label='outlier')
    plt.plot(np.linspace(-2 * g, 2 * g, 1000), [semicircle_density(x, g) for x in np.linspace(-2 * g, 2 * g, 1000)], 'r-', lw=2, label='Theoretical')
    plt.xlabel(r"$\lambda$", fontsize=15)
    plt.ylabel(r"$\rho (\lambda)$", fontsize=15)
elif mode == "a":
    eigen_real = np.real(eigenlist)
    eigen_imag = np.imag(eigenlist)
    # Plot the eigenvalues in the complex plane
    plt.plot(eigen_real, eigen_imag, 'o', markersize=2, color='blue', alpha=0.5)
    plt.plot(np.cos(np.linspace(0,2*np.pi,1000)), np.sin(np.linspace(0,2*np.pi,1000)) , 'r-', lw=2, label='Unit circle')
    if J0 > J:
        plt.axvline(x=J0, color='red', linestyle='--', label='outlier')
    plt.xlabel(r"Re($\lambda$)", fontsize=15)
    plt.ylabel(r"Im($\lambda$)", fontsize=15)

plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
#plt.legend(fontsize=15, loc='upper right')
plt.show()