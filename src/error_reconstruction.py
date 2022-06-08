####################################################################################
#
# Module to compute the error convergence in L_inf, L1 and L2 norms
# for the recontruction of a function using the Piecewise Parabolic Method (PPM)
# Luan da Fonseca Santos - April 2022
#
####################################################################################

import numpy as np
import reconstruction as rec
from parameters import q0, qexact, q0_antiderivative, simulation_par, graphdir
from errors import *
from monotonization import monotonization

def error_analysis_recon(simulation):
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
    Ntest = 14

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
        simulation = simulation_par(int(Nc[i]), 1.0, 1.0, ic, tc, mono)
        N  = simulation.N
        x  = simulation.x
        xc = simulation.xc
        dx = simulation.dx
        q_parabolic = np.zeros(Nplot)
        dists = abs(np.add.outer(xplot,-xc))
        neighbours = dists.argmin(axis=1)

        # Compute average values of Q (initial condition)
        Q = np.zeros(N+5)

        if (simulation.ic == 0 or simulation.ic == 1 or simulation.ic == 3 or simulation.ic == 4 or simulation.ic == 5):
            Q[2:N+2] = (q0_antiderivative(x[1:N+1], simulation) - q0_antiderivative(x[0:N], simulation))/dx
        elif (simulation.ic == 2):
            Q[2:N+2] = q0_antiderivative(x, simulation)/dx
       
        # Periodic boundary conditions
	    #Q[N+2:N+5] = Q[2:5]
        #Q[0:2]    = Q[N:N+2]
        if (simulation.ic == 0 or simulation.ic == 1 or simulation.ic == 3 or simulation.ic == 4 or simulation.ic == 5):
            Q[N+2] = (q0_antiderivative(xf+1.0*dx, simulation) - q0_antiderivative(xf+0.0*dx, simulation))/dx
            Q[N+3] = (q0_antiderivative(xf+2.0*dx, simulation) - q0_antiderivative(xf+1.0*dx, simulation))/dx
            Q[N+4] = (q0_antiderivative(xf+3.0*dx, simulation) - q0_antiderivative(xf+2.0*dx, simulation))/dx
            Q[1]   = (q0_antiderivative(x0-0.0*dx, simulation) - q0_antiderivative(x0-1.0*dx, simulation))/dx
            Q[0]   = (q0_antiderivative(x0-1.0*dx, simulation) - q0_antiderivative(x0-2.0*dx, simulation))/dx
        elif (simulation.ic == 2):
            Q[N+2] = q0_antiderivative([xf+0.0*dx, xf+1.0*dx], simulation)
            Q[N+3] = q0_antiderivative([xf+1.0*dx, xf+2.0*dx], simulation)
            Q[N+4] = q0_antiderivative([xf+2.0*dx, xf+3.0*dx], simulation)
            Q[1]   = q0_antiderivative([x0-1.0*dx, x0-0.0*dx], simulation)
            Q[0]   = q0_antiderivative([x0-2.0*dx, x0-1.0*dx], simulation)

        # Reconstructs the values of Q using a piecewise parabolic polynomial
        dq, q6, q_L, q_R = rec.ppm_reconstruction(Q, N)

        # Applies monotonization on the parabolas
        monotonization(Q, q_L, q_R, dq, q6, N, mono)

        # Compute the parabola
        for k in range(0, N):
            z = (xplot[neighbours==k]-x[k])/dx # Maps to [0,1]
            q_parabolic[neighbours==k] = q_L[k+2] + dq[k+2]*z+ z*(1.0-z)*q6[k+2]

        # Compute exact solution
        q_exact = qexact(xplot, 0, simulation)
        q_exact_edges = qexact(x, 0, simulation)
        ymin = np.amin(q_exact)
        ymax = np.amax(q_exact)

        # Relative errors in different metrics
        error_linf[i], error_l1[i], error_l2[i] = compute_errors(q_exact, q_parabolic)
        error_ed_linf[i], error_ed_l1[i], error_ed_l2[i] = compute_errors(q_exact_edges[0:N], q_L[2:N+2])
        print('\nParameters: N = '+str(N))
        
        # Output
        #print_errors_simul(error_linf, error_l1, error_l2, i)
        print_errors_simul(error_ed_linf, error_ed_l1, error_ed_l2, i)

    # Plot the error graph
    title = 'Parabola errors\n ' + simulation.title + '- ' + simulation.fvmethod + ' - ' + simulation.icname + ' - monotonization = ' + simulation.monot
    filename = graphdir+'tc'+str(tc)+'_'+simulation.fvmethod+'_mono'+simulation.monot+'_ic'+str(ic)+'_parabola_errors.png'
    plot_errors_loglog(Nc, error_linf, error_l1, error_l2, filename, title)

    title2 = 'Edge errors\n' + simulation.title + '- ' + simulation.fvmethod + ' - ' + simulation.icname + ' - monotonization = ' + simulation.monot
    filename2 = graphdir+'tc'+str(tc)+'_'+simulation.fvmethod+'_mono'+simulation.monot+'_ic'+str(ic)+'_edge_errors.png'
    plot_errors_loglog(Nc, error_ed_linf, error_ed_l1, error_ed_l2, filename2, title2)


    print('\nGraphs have been ploted in '+graphdir)
    print('Convergence graphs has been ploted in '+filename+' and in '+filename2)
