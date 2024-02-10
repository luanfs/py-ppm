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
    CFL = 0.1

    # Interval
    x0 = simulation.x0
    xf = simulation.xf

    # Number of tests
    Ntest = 4

    # Number of cells
    N = np.zeros(Ntest)
    N[0] = 48

    # Array of time steps
    dt = np.zeros(Ntest)
    if simulation.vf == 1: # constant velocity
        u = velocity_adv_1d(x0, 0, simulation)
        dt[0] = CFL/(N[0]*abs(u))*(xf-x0)
        Tf = 5.0
    elif simulation.vf==2: #variable velocity
        u0 = 0.4
        dt[0] = CFL/(N[0])*(xf-x0)/u0
        dt[0] = 0.04
        Tf = 5.0
    else:
        print("ERROR: invalid vector field ", simulation.vf)
        exit()

    # Errors array
    recons = (1,1,4,4)
    deps = (1,2,1,2)
    #recons = (simulation.recon,)
    #deps = (simulation.dp,)
    #recon_names = ['PPM-0', 'PPM-CW84','PPM-PL07','PPM-L04']
    dp_names = ['RK1', 'RK2','RK3']
    error_linf = np.zeros((Ntest, len(deps)))
    error_l1   = np.zeros((Ntest, len(deps)))
    error_l2   = np.zeros((Ntest, len(deps)))

    # Compute number of cells and time step for each simulation
    for i in range(1, Ntest):
        N[i]  = N[i-1]*2.0
        dt[i] = dt[i-1]*0.5

    for k in range(0,len(deps)):
        dp = deps[k]
        recon = recons[k]
        # Let us test and compute the error
        for i in range(0, Ntest):
            # Update simulation parameters
            simulation = simulation_adv_par_1d(int(N[i]), dt[i], Tf, ic, vf, tc, recon, dp)

            # Run advection routine and get the errors
            error_linf[i,k], error_l1[i,k], error_l2[i,k] =  adv_1d(simulation, False)
            print('\nParameters: N = '+str(int(N[i]))+', dt = '+str(dt[i]),', recon = ', recon, ', dp = ', dp)

            # Output
            print_errors_simul(error_linf[:,k], error_l1[:,k], error_l2[:,k], i)

    # plot errors for different all schemes in  different norms
    error_list = [error_linf, error_l1, error_l2]
    norm_list  = ['linf','l1','l2']
    norm_title  = [r'$L_{\infty}$',r'$L_1$',r'$L_2$']

    e = 0
    for errors in error_list:
        emin, emax = np.amin(errors), np.amax(errors)
     
        # convergence rate min/max
        n = len(errors)
        CR = np.abs(np.log(errors[1:n])-np.log(errors[0:n-1]))/np.log(2.0)
        CRmin, CRmax = np.amin(CR), np.amax(CR)       

        title = simulation.title + ' - ' + simulation.icname+', vf='+ str(simulation.vf)+\
            ', norm='+norm_title[e]
        filename = graphdir+'1d_adv_tc'+str(tc)+'_ic'+str(ic)+'_vf'+str(vf)+\
            '_norm'+norm_list[e]+'_parabola_errors.pdf'

        names = []
        errors_list = []
        for k in range(0,len(deps)):
            recon = recons[k]
            dp = deps[k]
            names.append('PPM'+str(recon)+' - RK'+str(dp)) 
            errors_list.append(errors[:,k])
        plot_errors_loglog(N, errors_list, names, filename, title, emin, emax)

        # Plot the convergence rate
        title = 'Convergence rate - ' + simulation.icname +', vf=' + str(simulation.vf)+\
        ', norm='+norm_title[e]
        filename = graphdir+'1d_adv_tc'+str(tc)+'_ic'+str(ic)+'_vf'+str(vf)+\
        '_norm'+norm_list[e]+'_convergence_rate.pdf'
        plot_convergence_rate(N, errors, names, filename, title, CRmin, CRmax)
        e = e+1
