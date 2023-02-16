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
        u0 = 0.2
        dt[0] = CFL/(N[0])*(xf-x0)/u0
        Tf = 5.0
    else:
        print("ERROR: invalid vector field ", simulation.vf)
        exit()

    # Errors array
    #recons = (1,2,3,4)
    recons = (simulation.recon,)
    #deps = (1,2)
    deps = (simulation.dp,)
    recon_names = ['PPM', 'PPM_mono_CW84','PPM_hybrid','PPM_mono_L04']
    dp_names = ['Euler', 'RK3']
    error_linf = np.zeros((Ntest, len(recons), len(deps)))
    error_l1   = np.zeros((Ntest, len(recons), len(deps)))
    error_l2   = np.zeros((Ntest, len(recons), len(deps)))

    # Compute number of cells and time step for each simulation
    for i in range(1, Ntest):
        N[i]  = N[i-1]*2.0
        dt[i] = dt[i-1]*0.5

    d = 0
    for dp in deps:
        rec = 0
        for recon in recons:
            # Let us test and compute the error
            for i in range(0, Ntest):
                # Update simulation parameters
                simulation = simulation_adv_par_1d(int(N[i]), dt[i], Tf, ic, vf, tc, recon, dp)

                # Run advection routine and get the errors
                error_linf[i,rec,d], error_l1[i,rec,d], error_l2[i,rec,d] =  adv_1d(simulation, False)
                print('\nParameters: N = '+str(int(N[i]))+', dt = '+str(dt[i]),', recon = ', recon, ', dp = ', dp)

                # Output
                print_errors_simul(error_linf[:,rec,d], error_l1[:,rec,d], error_l2[:,rec,d], i)

            # Plot the errors
            title = simulation.title +' - '+ simulation.recon_name + ' - '+simulation.dp_name+ '\n' + simulation.icname+', velocity = ' + str(simulation.vf)
            filename = graphdir+'1d_adv_tc'+str(tc)+'_'+simulation.recon_name+'_'+simulation.dp_name+'_ic'+str(ic)+'_vf'+str(vf)+'_parabola_errors.pdf'
            plot_errors_loglog(N, [error_linf[:,rec,d], error_l1[:,rec,d], error_l2[:,rec,d]], ['$L_\infty$', '$L_1$','$L_2$'], filename, title)

            # Plot the convergence rate
            title = 'Convergence rate - ' + simulation.recon_name + ' - '+simulation.dp_name+ '\n' + simulation.icname+', velocity = ' + str(simulation.vf)

            filename = graphdir+'1d_adv_tc'+str(tc)+'_'+simulation.recon_name+'_'+simulation.dp_name+'_ic'+str(ic)+'_vf'+str(vf)+'_convergence_rate.pdf'
            plot_convergence_rate(N, [error_linf[:,rec,d], error_l1[:,rec,d], error_l2[:,rec,d]],['$L_\infty$', '$L_1$','$L_2$'], filename, title)

            rec = rec+1

        # Plot the errors for different reconstruction methods
        errors = []
        rec_names = []
        for r in range(0,len(recons)):
            errors.append(error_linf[:,r,d])
            rec_names.append(recon_names[recons[r]-1])

        title = simulation.title + ' - ' + simulation.icname+', velocity = ' + str(simulation.vf)+' dp = '+str(simulation.dp)
        filename = graphdir+'1d_adv_tc'+str(tc)+'_ic'+str(ic)+'_vf'+str(vf)+'_dp'+simulation.dp_name+'_parabola_errors.pdf'
        plot_errors_loglog(N, errors, rec_names, filename, title)

        # Plot the convergence rate
        title = 'Convergence rate - ' + simulation.icname +', velocity = ' + str(simulation.vf)+', dp = '+str(simulation.dp)
        filename = graphdir+'1d_adv_tc'+str(tc)+'_ic'+str(ic)+'_vf'+str(vf)+'_dp'+simulation.dp_name+'_convergence_rate.pdf'
        plot_convergence_rate(N, errors, rec_names, filename, title)

        d = d+1

    # plot errors for different DP schemes
    for r in range(0, len(recons)):
        errors = []
        dep_name = []
        for d in range(0, len(deps)):
            errors.append(error_linf[:,r,d])
            dep_name.append(dp_names[deps[d]-1])

        title = simulation.title + ' - ' + simulation.icname+', velocity = ' + str(simulation.vf)+'\n recon = '+recon_names[recons[r]-1]
        filename = graphdir+'1d_adv_tc'+str(tc)+'_ic'+str(ic)+'_vf'+str(vf)+'_rec'+recon_names[recons[r]-1]+'_parabola_errors.pdf'
        plot_errors_loglog(N, errors, dep_name, filename, title)

        # Plot the convergence rate
        title = 'Convergence rate - ' + simulation.icname +', velocity = ' + str(simulation.vf)+ '\n recon = '+recon_names[recons[r]-1]
        filename = graphdir+'1d_adv_tc'+str(tc)+'_ic'+str(ic)+'_vf'+str(vf)+'_rec'+recon_names[recons[r]-1]+'_convergence_rate.pdf'
        plot_convergence_rate(N, errors, dep_name, filename, title)



    # Print final message
    print('\nGraphs have been ploted in '+graphdir)
    print('Convergence graphs has been ploted in '+filename)

