####################################################################################
#
# Module to compute the error convergence in L_inf, L1 and L2 norms
# for the advection equation using the Piecewise Parabolic Method (PPM)
# Luan da Fonseca Santos - April 2022
#
####################################################################################

from advection_1d import adv_1d
import numpy as np
from errors import *
from parameters_1d import simulation_adv_par_1d, graphdir
from advection_ic  import velocity_adv_1d

def error_analysis_adv1d(simulation):
    # Initial condition
    ic = simulation.ic

    # Flux method
    flux_method = simulation.flux_method

    # Test case
    tc = simulation.tc

    # CFL number for all simulations
    CFL = 0.8

    # Interval
    x0 = simulation.x0
    xf = simulation.xf

    # Number of tests
    Ntest = 10

    # Number of cells
    N = np.zeros(Ntest)
    N[0] = 10

    if simulation.ic >= 1 and simulation.ic <= 4: # constant velocity
        u = velocity_adv_1d(x0, 0, simulation)
        # Period
        Tf = (simulation.xf-simulation.x0)/(abs(u))
        # Array of time steps
        dt = np.zeros(Ntest)
        dt[0] = CFL/(N[0]*abs(u))*(xf-x0)
    else:
        exit()

    # Errors array
    error_linf = np.zeros(Ntest)
    error_l1   = np.zeros(Ntest)
    error_l2   = np.zeros(Ntest)

    # Compute number of cells and time step for each simulation
    for i in range(1, Ntest):
        N[i]  = N[i-1]*2.0
        dt[i] = dt[i-1]*0.5

    # Let us test and compute the error
    for i in range(0, Ntest):
        # Update simulation parameters
        simulation = simulation_adv_par_1d(int(N[i]), dt[i], Tf, ic, tc, flux_method)

        # Run advection routine and get the errors
        error_linf[i], error_l1[i], error_l2[i] =  adv_1d(simulation, False)
        print('\nParameters: N = '+str(int(N[i]))+', dt = '+str(dt[i]))

        # Output
        print_errors_simul(error_linf, error_l1, error_l2, i)

    # Plot the errors
    title = simulation.title + '- ' + simulation.flux_method_name + ' - ' + simulation.icname
    filename = graphdir+'1d_adv_tc'+str(tc)+'_'+simulation.flux_method_name+'_ic'+str(ic)+'_parabola_errors.png'
    plot_errors_loglog(N, error_linf, error_l1, error_l2, filename, title)

    # Plot the convergence rate
    title = 'Convergence rate - ' + simulation.flux_method_name + ' - ' + simulation.icname
    filename = graphdir+'1d_adv_tc'+str(tc)+'_'+simulation.flux_method_name+'_ic'+str(ic)+'_convergence_rate.png'
    plot_convergence_rate(N, error_linf, error_l1, error_l2, filename, title)

    # Print final message
    print('\nGraphs have been ploted in '+graphdir)
    print('Convergence graphs has been ploted in '+filename)
