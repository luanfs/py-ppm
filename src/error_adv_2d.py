###################################################################################
#
# Module to compute the error convergence in L_inf, L1 and L2 norms
# for the advection equation using the Piecewise Parabolic Method (PPM)
# Luan da Fonseca Santos - April 2022
#
####################################################################################

from advection_2d import adv_2d
import numpy as np
from errors import *
from parameters_2d import simulation_par_2d, graphdir, velocity_adv_2d

def error_analysis_adv2d(simulation):
    # Initial condition
    ic = simulation.ic

    # Monotonization method
    mono = simulation.mono

    # Test case
    tc = simulation.tc

    # CFL number for all simulations
    CFL = 0.25

    # Interval
    x0 = simulation.x0
    xf = simulation.xf
    y0 = simulation.y0
    yf = simulation.yf

    # Number of tests
    Ntest = 7

    # Number of cells
    N = np.zeros(Ntest)
    N[0] = 10
    M = np.zeros(Ntest)
    M[0] = 10

    # Timesteps
    dt = np.zeros(Ntest)

    u, v = velocity_adv_2d(x0, y0, 0, simulation)
    # Period
    if simulation.ic >= 1 and simulation.ic <= 4: # constant velocity
        Tf = (simulation.xf-simulation.x0)/(abs(u))
        dt[0] = CFL/(N[0] * np.sqrt( np.max(abs(u))**2 + np.max(abs(v))**2 ) )*(xf-x0)
    elif simulation.ic == 5: #deformational flow
        Tf = 5.0
        dt[0] = 0.05
    else:
        exit()

    # Array of time steps

    # Errors array
    error_linf = np.zeros(Ntest)
    error_l1   = np.zeros(Ntest)
    error_l2   = np.zeros(Ntest)

    # Compute number of cells and time step for each simulation
    for i in range(1, Ntest):
        N[i]  = N[i-1]*2.0
        M[i]  = M[i-1]*2.0
        dt[i] = dt[i-1]*0.5

    # Let us test and compute the error
    for i in range(0, Ntest):
        # Update simulation parameters
        simulation = simulation_par_2d(int(N[i]), int(M[i]), dt[i], Tf, ic, tc, mono)

        # Run advection routine and get the errors
        error_linf[i], error_l1[i], error_l2[i] =  adv_2d(simulation, False)
        print('\nParameters: N = '+str(int(N[i]))+', dt = '+str(dt[i]))

        # Output
        print_errors_simul(error_linf, error_l1, error_l2, i)

    # Plot the errors
    title = simulation.title + '- ' + simulation.fvmethod + ' - ' + simulation.icname + ' - monotonization = ' + simulation.monot
    filename = graphdir+'1d_adv_tc'+str(tc)+'_'+simulation.fvmethod+'_mono'+simulation.monot+'_ic'+str(ic)+'_parabola_errors.png'
    plot_errors_loglog(N, error_linf, error_l1, error_l2, filename, title)

    # Print final message
    print('\nGraphs have been ploted in '+graphdir)
    print('Convergence graphs has been ploted in '+filename)
