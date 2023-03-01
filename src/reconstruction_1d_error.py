####################################################################################
#
# Module to compute the error convergence in L_inf, L1 and L2 norms
# for the recontruction of a function using the Piecewise Parabolic Method (PPM)
# Luan da Fonseca Santos - April 2022
#
####################################################################################

import numpy as np
from reconstruction_1d import ppm_reconstruction
from parameters_1d import  simulation_recon_par_1d, graphdir, ppm_parabola
from advection_ic  import  q0_adv, qexact_adv, q0_antiderivative_adv
from errors import *

def error_analysis_recon_1d(simulation):
    # Initial condition
    ic = simulation.ic

    # Flux method
    recon = simulation.recon

    # Interval
    x0 = simulation.x0
    xf = simulation.xf

    # Number of tests
    Ntest = 7

    # Number of cells
    Nc = np.zeros(Ntest)
    Nc[0] = 16

    # Reconstruction schemes
    recons = (1,2,3,4)
    recon_names = ['PPM', 'PPM-CW84','PPM-PL07','PPM-L04']

    # Errors array
    error_linf = np.zeros((Ntest, len(recons)))
    error_l1   = np.zeros((Ntest, len(recons)))
    error_l2   = np.zeros((Ntest, len(recons)))
    error_ed_linf = np.zeros((Ntest, len(recons)))
    error_ed_l1   = np.zeros((Ntest, len(recons)))
    error_ed_l2   = np.zeros((Ntest, len(recons)))


    # Compute number of cells for each simulation
    for i in range(1, Ntest):
        Nc[i] = Nc[i-1]*2.0

    # Aux. variables
    Nplot = 10000
    x0 = simulation.x0
    xf = simulation.xf
    xplot = np.linspace(x0, xf, Nplot)

    # Let us test and compute the error!
    rec = 0
    for recon in recons:
        for i in range(0, Ntest):
            # Update simulation parameters
            simulation = simulation_recon_par_1d(int(Nc[i]), ic, recon)
            N  = simulation.N
            x  = simulation.x
            xc = simulation.xc
            dx = simulation.dx

            # Ghost cells
            ngl = simulation.ngl
            ngr = simulation.ngr
            ng  = simulation.ng

            # Grid interior indexes
            i0 = simulation.i0
            iend = simulation.iend

            # PPM parabola
            px = ppm_parabola(simulation)

            # Plot vars
            q_parabolic = np.zeros(Nplot)
            dists = abs(np.add.outer(xplot,-xc[i0:iend]))
            neighbours = dists.argmin(axis=1)

            # Compute average values of Q (initial condition)
            Q = np.zeros(N+ng)

            if (simulation.ic == 0 or simulation.ic == 1 or simulation.ic == 3 or simulation.ic == 4 or simulation.ic == 5):
                Q[i0:iend] = (q0_antiderivative_adv(x[i0+1:iend+1], simulation) - q0_antiderivative_adv(x[i0:iend], simulation))/dx
            elif (simulation.ic == 2):
                Q[i0:iend] = q0_adv(xc[i0:iend], simulation)

            # Periodic boundary conditions
            Q[iend:N+ng] = Q[i0:i0+ngr]
            Q[0:i0]      = Q[N:N+ngl]

            # Reconstructs the values of Q using a piecewise parabolic polynomial
            ppm_reconstruction(Q, px, simulation)


            # Compute the parabola
            for k in range(0, N):
                z = (xplot[neighbours==k]-x[k+i0])/dx # Maps to [0,1]
                q_parabolic[neighbours==k] = px.q_L[k+i0] + px.dq[k+i0]*z+ z*(1.0-z)*px.q6[k+i0]

            # Compute exact solution
            q_exact = qexact_adv(xplot, 0, simulation)
            q_exact_edges = qexact_adv(x, 0, simulation)
            ymin = np.amin(q_exact)
            ymax = np.amax(q_exact)

            # Relative errors in different metrics
            error_linf[i,rec], error_l1[i,rec], error_l2[i,rec] = compute_errors(q_exact, q_parabolic)
            error_ed_linf[i,rec], error_ed_l1[i,rec], error_ed_l2[i,rec] = compute_errors(q_exact_edges[i0:iend], px.q_L[i0:iend])
            print('\nParameters: N = '+str(N)+', recon='+str(rec))

            # Output
            #print_errors_simul(error_linf, error_l1, error_l2, i)
            print_errors_simul(error_ed_linf[:,rec], error_ed_l1[:,rec], error_ed_l2[:,rec], i)
        rec = rec+1

    # plot errors for different all schemes in  different norms
    error_list = [error_ed_linf, error_ed_l1, error_ed_l2]
    #error_list = [error_linf, error_l1, error_l2]
    norm_list  = ['linf','l1','l2']
    norm_title  = [r'$L_{\infty}$',r'$L_1$',r'$L_2$']

    e = 0
    for error in error_list:
        emin, emax = np.amin(error[:]), np.amax(error[:])

        # convergence rate min/max
        n = len(error)
        CR = np.abs(np.log(error[1:n])-np.log(error[0:n-1]))/np.log(2.0)
        CRmin, CRmax = np.amin(CR), np.amax(CR)
        errors = []
        dep_name = []
        for r in range(0, len(recons)):
            errors.append(error[:,r])
            dep_name.append(recon_names[recons[r]-1])

        title = simulation.title + ' - ' + simulation.icname+', norm='+norm_title[e]
        filename = graphdir+'recon_1d_ic'+str(ic)+'_norm'+norm_list[e]+'_parabola_errors.pdf'

        plot_errors_loglog(Nc, errors, dep_name, filename, title, emin, emax)

        # Plot the convergence rate
        title = 'Convergence rate - ' + simulation.icname+', norm='+norm_title[e]
        filename = graphdir+'recon_1d_ic'+str(ic)+'_norm'+norm_list[e]+'_convergence_rate.pdf'
        plot_convergence_rate(Nc, errors, dep_name, filename, title, CRmin, CRmax)
        e = e+1

