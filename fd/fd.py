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

# Function to be tested
def f(x):
    return np.exp(-x*x)

# Derivative of f
def df(x):
    return -2.0*x*np.exp(-x*x)

# Interval
a = -1.0
b =  1.0

# Points where we compute the FD
N = 10
x = np.linspace(a, b, N)

# Exact derivative at x
exact_df = df(x)

# Test variables 
Ntest = 15
error_linf, error_l1, error_l2 = np.zeros(Ntest), np.zeros(Ntest), np.zeros(Ntest)

# Spacing (h)
dx = np.zeros(Ntest)
dx[0] = 0.1
for k in range(1, Ntest):
    dx[k] = dx[k-1]*0.5

for k in range(0, Ntest):
    h = dx[k]
    
    Q0 = (-f(x-2*h) + f(x-h) )/h
    Q1 =   f(x+h)/h
    Q2 =  -f(x-h)/h
    Q3 = ( f(x+2*h) - f(x+h) )/h

    dQ0_1 = (Q2 - Q0)/2.0
    dQ1_1 = (Q3 - Q1)/2.0
    
    dQ0_2 = (Q2 - Q1)*2.0
    dQ1_2 = (Q3 - Q2)*2.0
    
    dQ0_3 = (Q2 - Q0)/1.0
    dQ1_3 = (Q3 - Q1)/1.0
    
    dQ0 = dQ0_3
    dQ1 = dQ1_3
    
    #dQ = np.minimum(abs(dQ0[1:N+4]), abs(dQ1[1:N+4]))
    #dQ[1:N+4] = np.minimum(dQ[1:N+4], abs(dQ2[1:N+4]))*np.sign(dQ0[1:N+4])
    #mask = ( (Q[2:N+5] - Q[1:N+4]) * (Q[1:N+4] - Q[0:N+3]) > 0.0 ) # Indexes where (Q_{j+1}-Q_{j})*(Q_{j}-Q{j-1}) > 0
    #Q[1:N+4][~mask] = 0.0
    #dQ = dQ0


    fd = (Q1 + Q2)/2.0 -(np.sign(dQ1_1)*abs(dQ1) - np.sign(dQ0_1)*abs(dQ0))/6.0

    error_linf[k], error_l1[k], error_l2[k] = compute_errors(fd, exact_df)
    print_errors_simul(error_linf, error_l1, error_l2, k)
    print('\n')

plot_errors_loglog(1.0/dx, error_linf, error_l1, error_l2, 'fd', 'FD')

