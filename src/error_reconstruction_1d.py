####################################################################################
#
# Module to compute the error convergence in L_inf, L1 and L2 norms
# for the recontruction of a function using the Piecewise Parabolic Method (PPM)
# Luan da Fonseca Santos - April 2022
#
####################################################################################

import numpy as np
import reconstruction_1d as rec
from parameters_1d import  q0_adv, qexact_adv, q0_antiderivative_adv, simulation_par_1d, graphdir
from errors import *
from monotonization_1d import monotonization_1d

def error_analysis_recon_1d(simulation):
    # Initial condition
    ic = simulation.ic

    # Test case
    tc = simulation.tc

    # Monotonization method
    mono = simulation.mono

    # Interval
    x0 = simulation.x0
    xf = simulation.xf

    # Number of tests
    Ntest = 13

    # Number of cells
    Nc = np.zeros(Ntest)
    Nc[0] = 10

    # Errors array
    error_linf = np.zeros(Ntest)
    error_l1   = np.zeros(Ntest)
    error_l2   = np.zeros(Ntest)

    # Edges errors
    error_ed_linf = np.zeros(Ntest)
    error_ed_l1   = np.zeros(Ntest)
    error_ed_l2   = np.zeros(Ntest)

    # Compute number of cells for each simulation
    for i in range(1, Ntest):
        Nc[i] = Nc[i-1]*2.0

    # Aux. variables
    Nplot = 10000
    x0 = simulation.x0
    xf = simulation.xf
    xplot = np.linspace(x0, xf, Nplot)
    
    # Let us test and compute the error!
    for i in range(0, Ntest):
        # Update simulation parameters
        simulation = simulation_par_1d(int(Nc[i]), 1.0, 1.0, ic, tc, mono)
        N  = simulation.N
        x  = simulation.x
        xc = simulation.xc
        dx = simulation.dx
        q_parabolic = np.zeros(Nplot)
        dists = abs(np.add.outer(xplot,-xc[3:N+3]))
        neighbours = dists.argmin(axis=1)

        # Compute average values of Q (initial condition)
        Q = np.zeros(N+6)

        if (simulation.ic == 0 or simulation.ic == 1 or simulation.ic == 3 or simulation.ic == 4 or simulation.ic == 5):
            Q[3:N+3] = (q0_antiderivative_adv(x[4:N+4], simulation) - q0_antiderivative_adv(x[3:N+3], simulation))/dx
        elif (simulation.ic == 2):
            Q[3:N+3] = q0_adv(xc[3:N+3], simulation)
       
        # Periodic boundary conditions
        Q[N+3:N+6] = Q[3:6]
        Q[0:3]     = Q[N:N+3]

        # Reconstructs the values of Q using a piecewise parabolic polynomial
        dq, q6, q_L, q_R = rec.ppm_reconstruction(Q, simulation)

        # Applies monotonization on the parabolas
        monotonization_1d(Q, q_L, q_R, dq, q6, simulation)

        # Compute the parabola
        for k in range(0, N):
            z = (xplot[neighbours==k]-x[k+3])/dx # Maps to [0,1]
            q_parabolic[neighbours==k] = q_L[k+3] + dq[k+3]*z+ z*(1.0-z)*q6[k+3]

        # Compute exact solution
        q_exact = qexact_adv(xplot, 0, simulation)
        q_exact_edges = qexact_adv(x, 0, simulation)
        ymin = np.amin(q_exact)
        ymax = np.amax(q_exact)

        # Relative errors in different metrics
        error_linf[i], error_l1[i], error_l2[i] = compute_errors(q_exact, q_parabolic)
        error_ed_linf[i], error_ed_l1[i], error_ed_l2[i] = compute_errors(q_exact_edges[3:N+3], q_L[3:N+3])
        print('\nParameters: N = '+str(N))
        
        # Output
        #print_errors_simul(error_linf, error_l1, error_l2, i)
        print_errors_simul(error_ed_linf, error_ed_l1, error_ed_l2, i)

    # Plot the error graph
    title = 'Parabola errors\n ' + simulation.title + '- ' + simulation.fvmethod + ' - ' + simulation.icname + ' - monotonization = ' + simulation.monot
    filename = graphdir+'1d_tc'+str(tc)+'_'+simulation.fvmethod+'_mono'+simulation.monot+'_ic'+str(ic)+'_parabola_errors.png'
    plot_errors_loglog(Nc, error_linf, error_l1, error_l2, filename, title)

    title2 = 'Edge errors\n' + simulation.title + '- ' + simulation.fvmethod + ' - ' + simulation.icname + ' - monotonization = ' + simulation.monot
    filename2 = graphdir+'1d_tc'+str(tc)+'_'+simulation.fvmethod+'_mono'+simulation.monot+'_ic'+str(ic)+'_edge_errors.png'
    plot_errors_loglog(Nc, error_ed_linf, error_ed_l1, error_ed_l2, filename2, title2)

    # Plot the convergence rate - parabola
    title = 'Convergence rate parabola- ' + simulation.fvmethod + ' - ' + simulation.icname + ' - monotonization = ' + simulation.monot
    filename = graphdir+'1d_tc'+str(tc)+'_'+simulation.fvmethod+'_mono'+simulation.monot+'_ic'+str(ic)+'_convergence_rate_parabola.png'
    plot_convergence_rate(Nc, error_linf, error_l1, error_l2, filename, title)

    # Plot the convergence rate - edges
    title = 'Convergence rate at edges - ' + simulation.fvmethod + ' - ' + simulation.icname + ' - monotonization = ' + simulation.monot
    filename = graphdir+'1d_tc'+str(tc)+'_'+simulation.fvmethod+'_mono'+simulation.monot+'_ic'+str(ic)+'_convergence_rate_ed.png'
    plot_convergence_rate(Nc, error_ed_linf, error_ed_l1, error_ed_l2, filename, title)
    print('Convergence graphs has been ploted in '+filename+' and in '+filename2)
