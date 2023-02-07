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

    # Vector field
    vf = simulation.vf

    # recon method
    recon = simulation.recon

    # departure point method
    dp = simulation.dp

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
    if simulation.vf == 1: # constant velocity
        u = velocity_adv_1d(x0, 0, simulation)
        dt[0] = CFL/(N[0]*abs(u))*(xf-x0)
        Tf = 5.0
    elif simulation.vf==2: #variable velocity
        u0 = 1.0
        dt[0] = CFL/(N[0])*(xf-x0)/u0
        Tf = np.pi
    elif simulation.vf==3: #variable velocity
        u0 = 0.2
        dt[0] = CFL/(N[0])*(xf-x0)/u0
        Tf = 5.0
    else:
        print("ERROR: invalid vector field ", simulation.vf)
        exit()

    # Errors array
    recons = (1,2,3,4)
    recons = (1,)
    error_linf = np.zeros((Ntest, len(recons)))
    error_l1   = np.zeros((Ntest, len(recons)))
    error_l2   = np.zeros((Ntest, len(recons)))

    # Compute number of cells and time step for each simulation
    for i in range(1, Ntest):
        N[i]  = N[i-1]*2.0
        dt[i] = dt[i-1]*0.5

    for recon in recons:
        # Let us test and compute the error
        for i in range(0, Ntest):
            # Update simulation parameters
            simulation = simulation_adv_par_1d(int(N[i]), dt[i], Tf, ic, vf, tc, recon, dp)

            # Run advection routine and get the errors
            error_linf[i,recon-1], error_l1[i,recon-1], error_l2[i,recon-1] =  adv_1d(simulation, False)
            print('\nParameters: N = '+str(int(N[i]))+', dt = '+str(dt[i]))

            # Output
            print_errors_simul(error_linf[:,recon-1], error_l1[:,recon-1], error_l2[:,recon-1], i)

        # Plot the errors
        title = simulation.title +' - '+ simulation.recon_name + ' - '+simulation.dp_name+ '\n' + simulation.icname+', velocity = ' + str(simulation.vf)
        filename = graphdir+'1d_adv_tc'+str(tc)+'_'+simulation.recon_name+'_'+simulation.dp_name+'_ic'+str(ic)+'_vf'+str(vf)+'_parabola_errors.pdf'
        plot_errors_loglog(N, [error_linf[:,recon-1], error_l1[:,recon-1], error_l2[:,recon-1]], ['$L_\infty$', '$L_1$','$L_2$'], filename, title)

        # Plot the convergence rate
        title = 'Convergence rate - ' + simulation.recon_name + ' - '+simulation.dp_name+ '\n' + simulation.icname+', velocity = ' + str(simulation.vf)

        filename = graphdir+'1d_adv_tc'+str(tc)+'_'+simulation.recon_name+'_'+simulation.dp_name+'_ic'+str(ic)+'_vf'+str(vf)+'_convergence_rate.pdf'
        plot_convergence_rate(N, [error_linf[:,recon-1], error_l1[:,recon-1], error_l2[:,recon-1]],['$L_\infty$', '$L_1$','$L_2$'], filename, title)

        # Print final message
        print('\nGraphs have been ploted in '+graphdir)
        print('Convergence graphs has been ploted in '+filename)

    exit()
    # Plot the errors
    title = simulation.title + ' - ' + simulation.icname+', velocity = ' + str(simulation.vf)+' dp = '+str(simulation.dp)
    filename = graphdir+'1d_adv_tc'+str(tc)+'_ic'+str(ic)+'_vf'+str(vf)+'_dp'+simulation.dp_name+'_parabola_errors.pdf'
    plot_errors_loglog(N, [error_linf[:,0],error_linf[:,1],error_linf[:,2],error_linf[:,3]], ['PPM', 'PPM_mono_CW84','PPM_hybrid','PPM_mono_L04'], filename, title)

    # Plot the convergence rate
    title = 'Convergence rate - ' + simulation.icname +', velocity = ' + str(simulation.vf)+', dp = '+str(simulation.dp)
    filename = graphdir+'1d_adv_tc'+str(tc)+'_ic'+str(ic)+'_vf'+str(vf)+'_dp'+simulation.dp_name+'_convergence_rate.pdf'
    plot_convergence_rate(N, [error_linf[:,0],error_linf[:,1],error_linf[:,2],error_linf[:,3]], ['PPM', 'PPM_mono_CW84','PPM_hybrid','PPM_mono_L04'], filename, title)

    # Print final message
    print('\nGraphs have been ploted in '+graphdir)
    print('Convergence graphs has been ploted in '+filename)

