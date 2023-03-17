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
        k = 5 #wavenumber
        y = -np.cos(k*2.0*np.pi*x)/(k*2.0*np.pi) + 1.0*x

    elif simulation.ic == 3:
        mask1 = np.logical_and(x>=0.4,x<=0.5)
        mask2 = np.logical_and(x>=0.5,x<=0.6)
        y = x*0
        y[mask1==True] = (x[mask1==True]-0.4)**2/0.2
        y[mask2==True] =0.1-(0.6-x[mask2==True])**2/0.2
        mask3 = x>0.6
        y[mask3==True] = 0.1

    elif simulation.ic == 4:
        mask = np.logical_and(x>=0.4, x<=0.6)
        y = x*0
        y[mask==True] = x[mask==True] - 0.4
        mask2 = x>0.6
        y[mask2==True] = 0.2

    return y

####################################################################################
# Exact solution to the advection problem
####################################################################################
def qexact_adv(x, t, simulation):
    x0 = simulation.x0
    xf = simulation.xf
    ic = simulation.ic

    #-----------------------------------------------------------------------
    # X = characteristic curve
    if simulation.vf == 1: # constant velocity_adv_1d
        u = velocity_adv_1d(x, t, simulation)
        X = x-u*t
        mask = (X != xf)
        X[mask] = (X[mask]-x0)%(xf-x0) + x0 # maps back to [x0,xf]

    elif simulation.vf == 2: # variable spped
        #characteristic curve not available for this test :(
        X = x#-u0*np.sin(2.0*np.pi*t/T)*T/(2.0*np.pi)

    #-----------------------------------------------------------------------
    if simulation.ic == 1:
        k = 5.0 #wavenumber
        y = np.sin(k*2.0*np.pi*X) + 1.0

    elif simulation.ic == 2:
        y = np.exp(-10*(np.cos(-np.pi*X))**2)

    elif simulation.ic == 3:
        mask1 = np.logical_and(X>=0.4,X<=0.5)
        mask2 = np.logical_and(X>=0.5,X<=0.6)
        y = x*0
        y[mask1==True] = ( X[mask1==True] - 0.4)/0.1
        y[mask2==True] = (-X[mask2==True] + 0.6)/0.1

    elif simulation.ic == 4:
        mask = np.logical_and(X>=0.4,X<=0.6)
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

    if simulation.vf == 1: # constant velocity
        u = velocity_adv_1d(x, t, simulation)
        X = x-u*t

        if simulation.ic == 1:
            y = (q0_antiderivative_adv(X[i0+1:iend+1], simulation)-q0_antiderivative_adv(X[i0:iend], simulation))/simulation.dx
            return y

        elif simulation.ic >= 2 :
            y = qexact_adv(simulation.xc[i0:iend], t, simulation)
            return y

    if simulation.vf == 2: # variable speed
        y = qexact_adv(simulation.xc[i0:iend], t, simulation)
        return y

    else:
        print('ERROR in Qexact_adv: invalid vector field: ', vf)


####################################################################################
# Velocity field
####################################################################################
def velocity_adv_1d(x, t, simulation):
    if simulation.vf == 1:
        u = 0.2

    elif simulation.vf == 2:
        T = 5
        u0 = 0.2
        u1 = 0.2
        u = u0*np.sin(np.pi*(x-t/T))**2*np.cos(np.pi*t/T) + u1

    return u

