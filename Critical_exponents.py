from numba import njit
import numpy as np
import matplotlib.pyplot as plt

# This scripts need the previous use of script HeatMaps_M_q
# ---------------------------------------------------------
@njit
def lin_fit(x, y):
    '''
    Make a linear fit to the data (x, y) using the least squares method.
    
    Parameters:
    x : array_like
        Independent variable data.
    y : array_like
        Dependent variable data.
    '''
    x = np.asarray(x)
    y = np.asarray(y)

    if x.size != y.size:
        raise ValueError("x and y must have the same size.")
    
    n = x.size
    m_x = np.mean(x)
    m_y = np.mean(y)

    SS_xy = np.sum(y * x) - n * m_y * m_x
    SS_xx = np.sum(x * x) - n * m_x * m_x

    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1 * m_x

    # Predictions and residuals
    y_fit = b_0 + b_1 * x
    residuals = y - y_fit

    # Residual variance
    sigma2 = np.sum(residuals**2) / (n - 2)

    # Standard errors of the coefficients
    err_b1 = np.sqrt(sigma2 / SS_xx)
    err_b0 = np.sqrt(sigma2 * (1/n + m_x**2 / SS_xx))

    # Coefficient of determination (R^2)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - m_y)**2)
    R2 = 1 - ss_res / ss_tot

    return b_1, b_0, err_b1, err_b0, R2
# ---------------------------------------------------------
g = 1     
IgJ_max = 3 
IgJ_min = 0.1 
JoJ_max = 3   
JoJ_min = 0.1 
hIgJ = 0.05
hJoJ = 0.05
IgJ_len = int((IgJ_max-IgJ_min)/hIgJ) + 1
JoJ_len = int((JoJ_max-JoJ_min)/hJoJ) + 1
IgJ_array = np.linspace(IgJ_min,IgJ_max,IgJ_len) 
JoJ_array = np.linspace(JoJ_min,JoJ_max,JoJ_len) 
# ---------------------------------------------------------
# Load results
M_read = np.loadtxt("M",delimiter = ",", comments="#")
C_read = np.loadtxt("C",delimiter = ",", comments="#")

index_05 = np.where(JoJ_array >= 0.4)[0][0]
index_15 = np.where(JoJ_array >= 1.4)[0][0]
index_25 = np.where(JoJ_array >= 2.4)[0][0]

M_05 = M_read[:,index_05]
M_15 = M_read[:,index_15]
M_25 = M_read[:,index_25]

C_05 = C_read[:,index_05]
C_15 = C_read[:,index_15]
C_25 = C_read[:,index_25]

# --------------------------------------------------------
g_domain_05 = 1/IgJ_array -1/1
g_domain_15 = 1/IgJ_array -1/1.5
g_domain_25 = 1/IgJ_array -1/2.5

M_05_critic = M_05[g_domain_05 > 0]   
C_05_critic = C_05[g_domain_05 > 0]      
g_domain_05 = g_domain_05[g_domain_05 > 0]

M_15_critic = M_15[g_domain_15 > 0]
C_15_critic = C_15[g_domain_15 > 0]         
g_domain_15 = g_domain_15[g_domain_15 > 0]

M_25_critic = M_25[g_domain_25 > 0]
C_25_critic = C_25[g_domain_25 > 0]
g_domain_25 = g_domain_25[g_domain_25 > 0]
# --------------------------------------------------------
# --------------------------------------------------------
# Bifurcation line
plt.figure()
plt.plot(IgJ_array,M_05,"ko-",label = r"$J_0/J = 0.5$")
plt.plot(IgJ_array,M_15,"ro-", label = r"$J_0/J = 1.5$")
plt.plot(IgJ_array,M_25,"bo-", label = "$J_0/J = 2.5$")

plt.xlabel("1/gJ",fontsize = 15)
plt.ylabel("M",fontsize = 15)
plt.xticks(fontsize = 13)
plt.yticks(fontsize = 13)
plt.grid()
plt.legend(fontsize = 13)
plt.savefig("MC_sinSeñalExterna/Exponentes_M.svg", format='svg', bbox_inches='tight')


plt.figure()
plt.plot(IgJ_array,C_05,"ko-",label = r"$J_0/J = 0.5$")
plt.plot(IgJ_array,C_15,"ro-", label = r"$J_0/J = 1.5$")
plt.plot(IgJ_array,C_25,"bo-", label = r"$J_0/J = 2.5$")

plt.xlabel("1/gJ",fontsize = 15)
plt.ylabel("q",fontsize = 15)
plt.xticks(fontsize = 13)
plt.yticks(fontsize = 13)
plt.grid()
plt.legend(fontsize = 13)
plt.savefig("MC_sinSeñalExterna/Exponentes_C.svg", format='svg', bbox_inches='tight')
# --------------------------------------------------------
# --------------------------------------------------------
# ---------------------------------------------------------
a1,c1,u_a1,u_c1,R2_1 =   lin_fit(np.log(g_domain_05)[-7:],np.log(M_05_critic)[-7:])
a15,c15,u_a15,u_c15,R2_15 =   lin_fit(np.log(g_domain_15)[-8:],np.log(M_15_critic)[-8:])
a25,c25,u_a25,u_c25,R2_25 =   lin_fit(np.log(g_domain_25)[-9:],np.log(M_25_critic)[-9:])
print(f"alpha (0.5) = {a1:.3f} ± {u_a1:.3f}, alpha (1.5) = {a15:.3f} ± {u_a15:.3f}, alpha (2.5) = {a25:.2f} ± {u_a25:.2f}")
# # ---------------------------------------------------------
# Coefficient alpha
plt.figure()

plt.plot(g_domain_05,M_05_critic,"ko",label = r"$J_0/J = 0.5$") 
plt.plot(g_domain_15,M_15_critic,"ro", label = r"$J_0/J = 1.5$") 
plt.plot(g_domain_25,M_25_critic,"bo", label = r"$J_0/J = 2.5$")

#plt.plot(g_domain_05,np.e**c1 * g_domain_05**a1,"k--")
plt.plot(g_domain_15,np.e**c15 * g_domain_15**a15,"r--")
plt.plot(g_domain_25,np.e**c25 * g_domain_25**a25,"b--")

plt.xscale("log")
plt.yscale("log")

plt.xlabel(r"$g-g_{c}$",fontsize = 15)
plt.ylabel("M",fontsize = 15)
#plt.xlim(xmin = 0, xmax = 0.5)
plt.xticks(fontsize = 13)
plt.yticks(fontsize = 13)
plt.grid()
plt.legend(fontsize = 13)

# ---------------------------------------------------------
# Coefficient beta
b1,c1,u_b1,u_c1,R2_1 =   lin_fit(np.log(g_domain_05)[-8:],np.log(C_05_critic)[-8:])
b15,c15,u_b15,u_c15,R2_15 =   lin_fit(np.log(g_domain_15)[-8:-1],np.log(C_15_critic)[-8:-1])
b25,c25,u_b25,u_c25,R2_25 =   lin_fit(np.log(g_domain_25)[-8:-1],np.log(C_25_critic)[-8:-1])
print(f"beta (0.5) = {b1:.3f} ± {u_b1:.3f}, beta (1.5) = {b15:.3f} ± {u_b15:.3f}, beta (2.5) = {b25:.2f} ± {u_b25:.2f}")

plt.figure()

plt.plot(g_domain_05,C_05_critic,"ko",label = r"$J_0/J = 0.5$") 
plt.plot(g_domain_15,C_15_critic,"ro", label = r"$J_0/J = 1.5$") 
plt.plot(g_domain_25,C_25_critic,"bo", label = r"$J_0/J = 2.5$")

plt.plot(g_domain_05,np.e**c1 * g_domain_05**b1,"k--")
plt.plot(g_domain_15,np.e**c15 * g_domain_15**b15,"r--")
plt.plot(g_domain_25,np.e**c25 * g_domain_25**b25,"b--")

plt.xscale("log")
plt.yscale("log")

plt.xlabel(r"$g-g_{c}$",fontsize = 15)
plt.ylabel("q",fontsize = 15)
#plt.xlim(xmin = 0, xmax = 0.5)
plt.xticks(fontsize = 13)
plt.yticks(fontsize = 13)
plt.grid()
plt.legend(fontsize = 13)

plt.show()