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

# Function to be tested
def f(x):
    #return np.exp(-x*x)
    #return x*x
    #return x*x*x
    #return x*x*x*x
    #return x**5
    return np.sin(c*x)

# Derivative of f
def df(x):
    #return -2.0*x*np.exp(-x*x)
    #return 2*x
    #return 3*x*x
    #return 4*x*x*x
    #return 5.0*x**4
    return c*np.cos(c*x)

# 2nd Derivative of f
def d2f(x):
    #return -4.0*x*(2.0*x*x-3.0)*np.exp(-x*x)
    #return 2.0
    #return 6.0*x
    #return 12.0*x*x
    #return 20.0*x**3
    return -c*c*np.sin(c*x)

# Interval
a = -1.0
b =  1.0

# Points where we compute the FD
N = 10
x = np.linspace(a, b, N)

# Exact derivative at x
exact_df = d2f(x)

# Test variables 
Ntest = 10
error_linf, error_l1, error_l2 = np.zeros(Ntest), np.zeros(Ntest), np.zeros(Ntest)

# Spacing (h)
dx = np.zeros(Ntest)
dx[0] = 0.1
for k in range(1, Ntest):
    dx[k] = dx[k-1]*0.5

#print(exact_df)
for k in range(0, Ntest):
    h = dx[k]
    
    #Q1 = (f(x-1*h) - f(x-2*h))/h
    #Q2 = (f(x+0*h) - f(x-1*h))/h
    #Q3 = (f(x+1*h) - f(x+0*h))/h
    #Q4 = (f(x+2*h) - f(x+1*h))/h
    #Q5 = (f(x+3*h) - f(x+2*h))/h

    #fd = -2*f(x-2*h) + 15*f(x-1*h) - 28*f(x) + 20*f(x+1*h) -6*f(x+2*h)+ f(x+3*h)
    #fd = (2.0*Q1 -13.0*Q2 + 15.0*Q3 -5.0*Q4 + Q5)
    #fd = fd/(6*h)

    fd1 = -2.0*f(x-2*h) + 15.0*f(x-1*h) - 28.0*f(x+0*h) + 20.0*f(x+1*h) - 6.0*f(x+2*h) + f(x+3*h)
    fd1 = fd1/(6*h*h)

    #print(np.amax(fd-fd1))
    #print(fd)
    #print(fd1)
    #print(np.amax(abs(fd1-exact_df))/np.amax(abs(exact_df)))
    error_linf[k], error_l1[k], error_l2[k] = compute_errors(fd1, exact_df)
    print_errors_simul(error_linf, error_l1, error_l2, k)
    print('\n')

plot_errors_loglog(1.0/dx, error_linf, error_l1, error_l2, 'fd', 'FD')
