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

def error_analysis_recon(simulation):
    # Initial condition
    ic = simulation.ic

    # Interval
    x0 = simulation.x0
    xf = simulation.xf

    # Number of tests
    Ntest = 12

    # Number of cells
    Ns = np.zeros(Ntest)
    Ns[0] = 10

    # Errors array
    error_linf = np.zeros(Ntest)
    error_l1   = np.zeros(Ntest)
    error_l2   = np.zeros(Ntest)

    # Compute number of cells for each simulation
    for i in range(1,Ntest):
        Ns[i] = Ns[i-1]*2.0
    
    # Aux. variables
    Nplot = 10000
    x0 = simulation.x0
    xf = simulation.xf
    xplot = np.linspace(x0, xf, Nplot)
    
    # Let us test and compute the error!
    for i in range(0, Ntest):
        # Update simulation parameters
        simulation = simulation_par(int(Ns[i]), 1.0, 1.0, ic)
        N  = simulation.N
        x  = simulation.x
        xc = simulation.xc
        dx = simulation.dx
        Q_parabolic = np.zeros(Nplot)
        dists = abs(np.add.outer(xplot,-xc))
        neighbours = dists.argmin(axis=1)

        # Compute average values of Q (initial condition)
        Q = np.zeros(N+5)

        if (simulation.ic == 0 or simulation.ic == 1 or simulation.ic == 3 or simulation.ic == 4 or simulation.ic == 5):
            Q[2:N+2] = (q0_antiderivative(x[1:N+1], simulation) - q0_antiderivative(x[0:N], simulation))/dx
        elif (simulation.ic == 2):
            Q[2:N+2] = q0(xc, simulation)
      
        # Periodic boundary conditions
        Q[N+2:N+5] = Q[2:5]
        Q[0:2]    = Q[N:N+2]
        
        #Q[N]   = (q0_antiderivative(xf+1.0*dx, simulation) - q0_antiderivative(xf+0.0*dx, simulation))/dx
        #Q[N+1] = (q0_antiderivative(xf+2.0*dx, simulation) - q0_antiderivative(xf+1.0*dx, simulation))/dx
        #Q[N+2] = (q0_antiderivative(xf+3.0*dx, simulation) - q0_antiderivative(xf+2.0*dx, simulation))/dx
        #Q[1]   = (q0_antiderivative(x0-0.0*dx, simulation) - q0_antiderivative(x0-1.0*dx, simulation))/dx
        #Q[0]   = (q0_antiderivative(x0-1.0*dx, simulation) - q0_antiderivative(x0-2.0*dx, simulation))/dx

        # Reconstructs the values of Q using a piecewise parabolic polynomial
        da, a6, aL, aR = rec.ppm_reconstruction(Q, N)

        # Compute the parabola
        for k in range(0, N):
            z = (xplot[neighbours==k]-x[k])/dx # Maps to [0,1]
            Q_parabolic[neighbours==k] = aL[k+2] + da[k+2]*z+ z*(1.0-z)*a6[k+2]

        # Compute exact solution
        q_exact = qexact(xplot, 0, simulation)
        ymin = np.amin(q_exact)
        ymax = np.amax(q_exact)

        # Relative errors in different metrics
        error_linf[i], error_l1[i], error_l2[i] = compute_errors(q_exact, Q_parabolic)
        print('\nParameters: N = '+str(N))
        
        # Output
        print_errors_simul(error_linf, error_l1, error_l2, i)

    # Plot the error graph
    title = simulation.name+' - 1d PPM reconstruction errors'
    filename = graphdir+'recon_ppm_ic'+str(ic)+'_errors.png'
    plot_errors_loglog(Ns, error_linf, error_l1, error_l2, filename, title)

    print('\nGraphs have been ploted in '+graphdir)
    print('Convergence graphs has been ploted in '+graphdir+'recon_ppm_ic'+str(ic)+'_errors.png')
