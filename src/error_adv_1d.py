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
from parameters import simulation_par_1d, graphdir

def error_analysis_adv1d(simulation):
    # adv velocity
    u = simulation.u

    # Period
    Tf = (simulation.xf-simulation.x0)/abs(u)

    # Initial condition
    ic = simulation.ic

    # Monotonization method
    mono = simulation.mono

    # Test case
    tc = simulation.tc

    # CFL number for all simulations
    CFL = 0.5

    # Interval
    x0 = simulation.x0
    xf = simulation.xf

    # Number of tests
    Ntest = 11

    # Number of cells
    N = np.zeros(Ntest)
    N[0] = 10

    # Array of time steps
    dt = np.zeros(Ntest)
    dt[0] = CFL/(N[0]*abs(u))*(xf-x0)

    # Errors array
    error_linf = np.zeros(Ntest)
    error_l1   = np.zeros(Ntest)
    error_l2   = np.zeros(Ntest)

    # Error at edges
    error_ed_linf = np.zeros(Ntest)
    error_ed_l1   = np.zeros(Ntest)
    error_ed_l2   = np.zeros(Ntest)

    # Compute number of cells and time step for each simulation
    for i in range(1, Ntest):
        N[i]  = N[i-1]*2.0
        dt[i] = dt[i-1]*0.5

    # Let us test and compute the error
    for i in range(0, Ntest):
        # Update simulation parameters
        simulation = simulation_par_1d(int(N[i]), dt[i], Tf, ic, tc, mono)

        # Run advection routine and get the errors
        error_linf[i], error_l1[i], error_l2[i], error_ed_linf[i], error_ed_l1[i], error_ed_l2[i] =  adv_1d(simulation, False)
        print('\nParameters: N = '+str(int(N[i]))+', dt = '+str(dt[i]))

        # Output
        #print_errors_simul(error_linf, error_l1, error_l2, i)
        print_errors_simul(error_ed_linf, error_ed_l1, error_ed_l2, i)

    # Plot the errors
    title = simulation.title + '- ' + simulation.fvmethod + ' - ' + simulation.icname + ' - monotonization = ' + simulation.monot
    filename = graphdir+'tc'+str(tc)+'_'+simulation.fvmethod+'_mono'+simulation.monot+'_ic'+str(ic)+'_parabola_errors.png'
    plot_errors_loglog(N, error_linf, error_l1, error_l2, filename, title)

    title = 'Edges error \n' + simulation.title + '- ' + simulation.fvmethod + ' - ' + simulation.icname + ' - monotonization = ' + simulation.monot
    filename2 = graphdir+'tc'+str(tc)+'_'+simulation.fvmethod+'_mono'+simulation.monot+'_ic'+str(ic)+'_edge_errors.png'
    plot_errors_loglog(N, error_ed_linf, error_ed_l1, error_ed_l2, filename2, title)

 
    # Print final message
    print('\nGraphs have been ploted in '+graphdir)
    print('Convergence graphs has been ploted in '+filename+' and '+filename2)
