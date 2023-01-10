####################################################################################
#
# Module for advection test case set up (initial condition, exact solution and etc)
#
# Luan da Fonseca Santos - October 2022
# (luan.santos@usp.br)
#
####################################################################################

import numpy as np

####################################################################################
# Initial condition
####################################################################################
def q0_adv(x, simulation):
    y = qexact_adv(x, 0, simulation)
    return y

####################################################################################
# Initial condition antiderivative
####################################################################################
def q0_antiderivative_adv(x, simulation):
    if simulation.ic == 1:
        y = -np.cos(2.0*np.pi*x)/(2.0*np.pi) + 1.0*x

    elif simulation.ic == 3:
        mask1 = np.logical_and(x>=15.0,x<=20.0)
        mask2 = np.logical_and(x>=20.0,x<=25.0)
        y = x*0
        y[mask1==True] = (x[mask1==True]-15.0)**2/10
        y[mask2==True] = 5.0 - (25.0-x[mask2==True])**2/10.0
        mask3 = x>25.0
        y[mask3==True] = 5

    elif simulation.ic == 4:
        mask = np.logical_and(x>=15.0, x<=25.0)
        y = x*0
        y[mask==True] = x[mask==True] - 15.0
        mask2 = x>25.0
        y[mask2==True] = 10

    return y

####################################################################################
# Exact solution to the advection problem
####################################################################################
def qexact_adv(x, t, simulation):
    x0 = simulation.x0
    xf = simulation.xf
    ic = simulation.ic

    if simulation.ic >= 1 and simulation.ic <= 5 : # constant speed
        u = velocity_adv_1d(x, t, simulation)
        X = x-u*t
        mask = (X != xf)
        X[mask] = (X[mask]-x0)%(xf-x0) + x0 # maps back to [x0,xf]

        if simulation.ic == 1:
            y = np.sin(2.0*np.pi*X) + 1.0

        elif simulation.ic == 2 or simulation.ic == 5:
            y = np.exp(-10*(np.sin(-np.pi*X))**2)

        elif simulation.ic == 3:
            mask1 = np.logical_and(X>=15.0,X<=20.0)
            mask2 = np.logical_and(X>=20.0,X<=25.0)
            y = x*0
            y[mask1==True] = ( X[mask1==True] - 15.0)/5.0
            y[mask2==True] = (-X[mask2==True] + 25.0)/5.0

        elif simulation.ic == 4:
            mask = np.logical_and(X>=15.0,X<=25.0)
            y = x*0
            y[mask==True] = 1.0
    return y

####################################################################################
# Exact average solution to the advection problem
####################################################################################
def Qexact_adv(x, t, simulation):
    x0 = simulation.x0
    xf = simulation.xf
    ic = simulation.ic
    N  = simulation.N
    i0   = simulation.i0
    iend = simulation.iend

    if simulation.ic >= 1 and simulation.ic <= 5 : # constant speed
        u = velocity_adv_1d(x, t, simulation)
        X = x-u*t
        #mask = (X != xf)
        #X[mask] = (X[mask]-x0)%(xf-x0) + x0 # maps back to [x0,xf]
        #X[N] = xf
        if simulation.ic == 1:
            y = (q0_antiderivative_adv(X[i0+1:iend+1], simulation)-q0_antiderivative_adv(X[i0:iend], simulation))/simulation.dx

        elif simulation.ic >= 2 :
            y = qexact_adv(simulation.xc[i0:iend], t, simulation)
    return y

####################################################################################
# Velocity field
####################################################################################
def velocity_adv_1d(x, t, simulation):
    if simulation.ic == 1:
        u = 0.1
    elif simulation.ic == 2:
        u = 0.1
    elif simulation.ic == 3:
        u = 0.5
    elif simulation.ic == 4:
        u = 0.5
    elif simulation.ic == 5:
        u = np.sin(np.pi*x)**2
    elif simulation.ic == 6:
        u = 1.0
    return u
