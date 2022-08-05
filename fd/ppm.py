#--------------------------------------------------------------------------------------
# Python script to test accuracy order of finite-difference (FD)  methods
# Luan Santos - 2022 
#--------------------------------------------------------------------------------------
# Source code directory
srcdir = "../src/"

import sys
import os.path
sys.path.append(srcdir)

# Imports
import numpy as np
import matplotlib.pyplot as plt
from errors import *

c = 4*np.pi

# Primitive
def qprim(x):
    return np.sin(c*x)
    #return x**3/3
    #return x**4/4
    #return x**5/5

# q
def q(x):
    return c*np.cos(c*x)
    #return x*x
    #return x*x*x
    #return x*x*x*x

# first derivative of q
def dxq(x):
    return -c*c*np.sin(c*x)
    #return 2*x
    #return 3*x*x
    #return 4*x*x*x    

# second derivative of q
def d2q(x):
    return -c*c*c*np.cos(c*x)
    #return 2*np.ones(np.shape(x))
    #return 6*x
    #return 12*x*x

####################################################################################
# Given the average values of a scalar field Q, this routine constructs 
# a piecewise parabolic aproximation of Q using its average value.
####################################################################################
def ppm_reconstruction(Q, N):
    # Compute the slopes dQ0 (centered finite difference)
    # Formula 1.7 from Collela and Woodward 1984 and Figure 2 from Carpenter et al 1990.
    dQ0 = np.zeros(N+5)
    dQ0[1:N+4] = 0.5*(Q[2:N+5] - Q[0:N+3]) # Interior values are in 2:N+2

    dQ = dQ0

    # Values of Q at right edges (q_(j+1/2)) - Formula 1.6 from Collela and Woodward 1984
    Q_edges = np.zeros(N+1)
    Q_edges[0:N+1] = 0.5*(Q[1:N+2] + Q[2:N+3]) - (dQ[2:N+3] - dQ[1:N+2])/6.0

    # Assign values of Q_R and Q_L
    q_R = np.zeros(N+5)
    q_R[2:N+2] = Q_edges[1:N+1]
    q_L = np.zeros(N+5)
    q_L[2:N+2] = Q_edges[0:N]

    # Compute the polynomial coefs
    # q(x) = q_L + z*(dq + q6*(1-z)) z in [0,1]
    dq = np.zeros(N+5)
    q6 = np.zeros(N+5)
    dq[2:N+2] = q_R[2:N+2] - q_L[2:N+2]
    q6[2:N+2] = 6*Q[2:N+2] - 3*(q_R[2:N+2] + q_L[2:N+2])

    # Periodic boundary conditions
    q_L[N+2:N+5] = q_L[2:5]
    q_L[0:2]     = q_L[N:N+2]

    q_R[N+2:N+5] = q_R[2:5]
    q_R[0:2]     = q_R[N:N+2]

    q6[N+2:N+5] = q6[2:5]
    q6[0:2]     = q6[N:N+2]

    dq[N+2:N+5] = dq[2:5]
    dq[0:2]     = dq[N:N+2]

    return dq, q6, q_L, q_R

# Interval
a = -1.0
b =  1.0

# Test variables 
Ntest = 10
error_linf, error_l1, error_l2 = np.zeros(Ntest), np.zeros(Ntest), np.zeros(Ntest)

# Spacing (Δx)
dx = np.zeros(Ntest)
Ns = np.zeros(Ntest)
N = 100
dx[0] = (b-a)/N
for k in range(1, Ntest):
    dx[k] = dx[k-1]*0.5

for k in range(0, Ntest):
    Δx = dx[k]
    # Points where we compute the FV
    x = np.linspace(a, b, N+1)
    xc = (x[1:N+1]+x[0:N])*0.5
    Q = np.zeros(N+5) # Interior values are in 2:N+2
    Q[2:N+2] = (qprim(x[1:N+1]) - qprim(x[0:N]))/Δx
    
    Q[N+2:N+5] = Q[2:5]
    Q[0:2]     = Q[N:N+2]
    
    Q[1]   = (qprim(x[0]-0.0*Δx) - qprim(x[0]-1.0*Δx))/Δx
    Q[0]   = (qprim(x[0]-1.0*Δx) - qprim(x[0]-2.0*Δx))/Δx
    Q[N+2] = (qprim(x[N]+1.0*Δx) - qprim(x[N]-0.0*Δx))/Δx
    Q[N+3] = (qprim(x[N]+2.0*Δx) - qprim(x[N]+1.0*Δx))/Δx
    Q[N+4] = (qprim(x[N]+3.0*Δx) - qprim(x[N]+2.0*Δx))/Δx

    dq, q6, q_L, q_R = ppm_reconstruction(Q, N)
    #print(np.amax(abs(Q[2:N+2]-q(xc))))
    #print(np.amax(abs(q_L[2:N+2]-q(x[0:N]))))
    #print(np.amax(abs(q6[2:N+2]))/(Δx)**2)
    q_d2L = d2q(x[0:N])
    q_dL  = dxq(x[0:N])
    #print(q6[2:N+2]/(Δx)**2)
    #print(q_d2L/2.0)

    error_linf[k], error_l1[k], error_l2[k] = compute_errors(q_L[2:N+2], q(x[0:N]))
    #error_linf[k], error_l1[k], error_l2[k] = compute_errors(fd_R, exact_df_R)
    #error_linf[k], error_l1[k], error_l2[k] = compute_errors(fd2, exact_d2f)    
    #print_errors_simul(error_linf, error_l1, error_l2, k)
    print('\n')
    print(N, error_linf[k])
    aux = (2.0*Q[0:N] -13.0*Q[1:N+1] + 15.0*Q[2:N+2] -5.0*Q[3:N+3] + Q[4:N+4])
    aux = aux/(6.0*Δx)
    erro = (dq[2:N+2]+q6[2:N+2])/(Δx)-aux
    print(np.amax(abs(erro))/np.amax(abs((dq[2:N+2]+q6[2:N+2])/(Δx))))
    #print(np.amax(abs(q6[2:N+2]/(Δx)**2 + q_d2L/2.0))/np.amax(abs(q_d2L/2.0))) 
    #print(np.amax(abs(q_dL-(dq[2:N+2]+q6[2:N+2])/(Δx)))/np.amax(abs(dq[2:N+2]+q6[2:N+2])/(Δx)))
    N = 2*N
#plot_errors_loglog(1.0/dx, error_linf, error_l1, error_l2, 'fd', 'FD')
