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
    recon = simulation.recon

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

    # Array of time steps
    dt = np.zeros(Ntest)
    if simulation.ic >= 1 and simulation.ic <= 4: # constant velocity
        u = velocity_adv_1d(x0, 0, simulation)
        # Period
        Tf = 5.0#(simulation.xf-simulation.x0)/(abs(u))

        dt[0] = CFL/(N[0]*abs(u))*(xf-x0)

    elif simulation.ic==5: #variable velocity
        u0 = 0.2
        dt[0] = CFL/(N[0])*(xf-x0)/u0
        Tf = 5.0
    elif simulation.ic==6: #variable velocity
        u0 = 0.2
        dt[0] = CFL/(N[0])*(xf-x0)/u0
        Tf = 5.0
    else:
        print("ERROR: invalid IC ", simulation.ic)
        exit()

    # Errors array
    fluxes = (1,2,3,4)
    error_linf = np.zeros((Ntest, len(fluxes)))
    error_l1   = np.zeros((Ntest, len(fluxes)))
    error_l2   = np.zeros((Ntest, len(fluxes)))

    # Compute number of cells and time step for each simulation
    for i in range(1, Ntest):
        N[i]  = N[i-1]*2.0
        dt[i] = dt[i-1]*0.5

    for flux in fluxes:
        # Let us test and compute the error
        for i in range(0, Ntest):
            # Update simulation parameters
            simulation = simulation_adv_par_1d(int(N[i]), dt[i], Tf, ic, tc, flux)

            # Run advection routine and get the errors
            error_linf[i,flux-1], error_l1[i,flux-1], error_l2[i,flux-1] =  adv_1d(simulation, False)
            print('\nParameters: N = '+str(int(N[i]))+', dt = '+str(dt[i]))

            # Output
            print_errors_simul(error_linf[:,flux-1], error_l1[:,flux-1], error_l2[:,flux-1], i)

        # Plot the errors
        title = simulation.title + ' - ' + simulation.recon_name + ' - ' + simulation.icname
        filename = graphdir+'1d_adv_tc'+str(tc)+'_'+simulation.recon_name+'_ic'+str(ic)+'_parabola_errors.pdf'
        plot_errors_loglog(N, [error_linf[:,flux-1], error_l1[:,flux-1], error_l2[:,flux-1]], ['$L_\infty$', '$L_1$','$L_2$'], filename, title)

        # Plot the convergence rate
        title = 'Convergence rate - ' + simulation.recon_name + ' - ' + simulation.icname
        filename = graphdir+'1d_adv_tc'+str(tc)+'_'+simulation.recon_name+'_ic'+str(ic)+'_convergence_rate.pdf'

        plot_convergence_rate(N, [error_linf[:,flux-1], error_l1[:,flux-1], error_l2[:,flux-1]],['$L_\infty$', '$L_1$','$L_2$'], filename, title)

        # Print final message
        print('\nGraphs have been ploted in '+graphdir)
        print('Convergence graphs has been ploted in '+filename)

    # Plot the errors
    title = simulation.title + ' - ' + simulation.icname
    filename = graphdir+'1d_adv_tc'+str(tc)+'_ic'+str(ic)+'_parabola_errors.pdf'
    plot_errors_loglog(N, [error_linf[:,0],error_linf[:,1],error_linf[:,2],error_linf[:,3]], ['PPM', 'PPM_mono_CW84','PPM_hybrid','PPM_mono_L04'], filename, title)

    # Plot the convergence rate
    title = 'Convergence rate - ' + simulation.icname
    filename = graphdir+'1d_adv_tc'+str(tc)+'_ic'+str(ic)+'_convergence_rate.pdf'
    plot_convergence_rate(N, [error_linf[:,0],error_linf[:,1],error_linf[:,2],error_linf[:,3]], ['PPM', 'PPM_mono_CW84','PPM_hybrid','PPM_mono_L04'], filename, title)

    # Print final message
    print('\nGraphs have been ploted in '+graphdir)
    print('Convergence graphs has been ploted in '+filename)

