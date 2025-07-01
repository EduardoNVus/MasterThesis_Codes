import numpy as np
import matplotlib.pyplot as plt
def log_density(x, intervalo, n_puntos):
    """
    Estimates the density function of a variable from a set of values,
    using logarithmically spaced bins. The density is normalized (PDF).

    Parameters:
    ----------
    x : array_like
        Set of variable values.
    intervalo : tuple
        Interval (min, max) in which the density is calculated (both must be > 0).
    n_puntos : int
        Number of logarithmic bins to divide the interval.

    Returns:
    -------
    bin_centers : ndarray
        Centers of the bins (in logarithmic scale).
    densidad : ndarray
        Normalized density (PDF) in each bin.
    """
    xmin, xmax = intervalo
    if xmin <= 0 or xmax <= 0:
        raise ValueError("The interval must be strictly positive for logarithmic scale.")
    if xmin >= xmax:
        raise ValueError("xmin must be less than xmax.")

    # Create logarithmic bins
    bins = np.logspace(np.log10(xmin), np.log10(xmax), n_puntos + 1)

    # Unnormalized histogram
    counts, edges = np.histogram(x, bins=bins)

    # Bin widths
    bin_widths = np.diff(edges)

    # Normalized density (PDF): count / (total * bin width)
    total_count = len(x)
    density = counts / (total_count * bin_widths)

    # Bin centers (geometric mean, appropriate for log scale)
    bin_centers = np.sqrt(edges[:-1] * edges[1:])

    return bin_centers, density

def synaptic(J, J0, N, g):
    # Given J, g, J0, and size N, returns a symmetric Gaussian matrix
    W = (J/np.sqrt(N)) * np.random.normal(0,1,(N,N)) # create random Gaussian connectivity matrix with sigma=J/np.sqrt(N)
    M = J0/N * np.ones((N,N)) # Create matrix of means
    return g * (M + W)

# Parameters
n = 3000   # Rows (variables)
m = 2000   # Columns (samples)
y = m / n

# Empirical distribution ------------------------------------
X = np.random.randn(n, m) # Data matrix with standard Gaussian entries
X1 = np.random.randn(3,2)
X2 = np.random.randn(30,20)
X3 = np.random.randn(300,200)

S = (1/n) * np.dot(X.T,X) # Sample covariance matrix
S1 = (1/3) * np.dot(X1.T,X1) # Sample covariance matrix
S2 = (1/30) * np.dot(X2.T,X2) # Sample covariance matrix
S3 = (1/300) * np.dot(X3.T,X3) # Sample covariance matrix

eigenvalues = np.linalg.eigvalsh(S) # Eigenvalues of the covariance matrix
eigenvalues1 = np.linalg.eigvalsh(S1) # Eigenvalues of the covariance matrix
eigenvalues2 = np.linalg.eigvalsh(S2) # Eigenvalues of the covariance matrix
eigenvalues3 = np.linalg.eigvalsh(S3) # Eigenvalues of the covariance matrix

x_emp, dens_emp = log_density(eigenvalues, (1e-5, 1e2), 100) # Empirical density estimation
x_emp1, dens_emp1 = log_density(eigenvalues1, (1e-5, 1e2), 100) # Empirical density estimation
x_emp2, dens_emp2 = log_density(eigenvalues2, (1e-5, 1e2), 100) # Empirical density estimation
x_emp3, dens_emp3 = log_density(eigenvalues3, (1e-5, 1e2), 100) # Empirical density estimation

# Theoretical distribution --------------------------------------
def mp_density(lambda_val, y):
    lambda_min = (1 - np.sqrt(y))**2
    lambda_max = (1 + np.sqrt(y))**2
    density = np.zeros_like(lambda_val)
    mask = (lambda_val >= lambda_min) & (lambda_val <= lambda_max) # Compact support
    density[mask] = (1 / (2 * np.pi * y * lambda_val[mask])) * \
                    np.sqrt((lambda_max - lambda_val[mask]) * (lambda_val[mask] - lambda_min))
    return density

# Theoretical limits of the distribution
lambda_min = (1 - np.sqrt(y))**2
lambda_max = (1 + np.sqrt(y))**2

lambda_teo = np.linspace(lambda_min,lambda_max,1000) # Theoretical domain of eigenvalues
dens_teo = mp_density(lambda_teo,y)

# Plot
#plt.plot(x_emp, dens_emp, 'bo',label='Empiric')
bins = np.logspace(np.log10(1e-2), np.log10(1e1), 100)  # Logarithmic bins

plt.plot(x_emp1, dens_emp1, 'g', label='3x2', markersize=6, marker = "p",linestyle = "dashed")
plt.plot(x_emp2, dens_emp2, 'y', label='30x20', markersize=6, marker = "*",linestyle = "dashed")
plt.plot(x_emp3, dens_emp3, 'm', label='300x200', markersize=6, marker = "o",linestyle = "dashed")
plt.plot(x_emp, dens_emp, 'b', label='3000x2000', markersize=6, marker = "s",linestyle = "dashed")

# plt.hist(eigenvalues, bins=bins, density=True, alpha=0.5, label='Empiric', color='black', edgecolor='black', histtype='stepfilled')
# plt.hist(eigenvalues1, bins=bins, density=True, alpha=0.5, label='Empiric 3x2', color='green', edgecolor='black', histtype='stepfilled')
# plt.hist(eigenvalues2, bins=bins, density=True, alpha=0.5, label='Empiric 30x20', color='orange', edgecolor='black', histtype='stepfilled')
# plt.hist(eigenvalues3, bins=bins, density=True, alpha=0.5, label='Empiric 300x200', color='purple', edgecolor='black', histtype='stepfilled')

plt.plot(lambda_teo, dens_teo, 'r-', lw=2, label='Theoretical')

plt.xlabel(r"$\lambda$",fontsize = 15)
plt.ylabel(r"$\rho (\lambda)$",fontsize = 15)
plt.xlim(0.001, 4)
# plt.yscale('log')
# plt.xscale('log')
plt.xticks(fontsize = 13)
plt.yticks(fontsize = 13)
plt.grid()
plt.legend(fontsize = 12,loc = 'upper right')
#plt.savefig("MP_log.svg", bbox_inches='tight', dpi=300,format='svg')
plt.show()