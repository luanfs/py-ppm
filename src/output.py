####################################################################################
#
# Module for output routines.
#
# Luan da Fonseca Santos - October 2022
# (luan.santos@usp.br)
####################################################################################

import numpy as np
from advection_ic      import Qexact_adv, qexact_adv
from errors            import *
from diagnostics       import diagnostics_adv_1d
from parameters_1d     import graphdir
from plot              import plot_1dfield_graphs
from reconstruction_1d import ppm_reconstruction

####################################################################################
# Print the diagnostics variables on the screen
####################################################################################
def print_diagnostics_adv_1d(error_linf, error_l1, error_l2, mass_change, t, Nsteps):
    print('\nStep', t, 'from', Nsteps)
    print('Error (Linf, L1, L2) :',"{:.2e}".format(error_linf), "{:.2e}".format(error_l1), "{:.2e}".format(error_l2))
    print('Total mass variation:', "{:.2e}".format(mass_change))

####################################################################################
# Output and plot
####################################################################################
def output_adv(x, xc, simulation, Q, dq, q6, q_L, error_linf, error_l1, error_l2, plot, k, t, Nsteps, plotstep, total_mass0, CFL):
    N = simulation.N
    dx = simulation.dx
    dt = simulation.dt

    # Ghost cells
    ngl = simulation.ngl
    ngr = simulation.ngr
    ng  = simulation.ng

    # Grid interior indexes
    i0 = simulation.i0
    iend = simulation.iend

    if plot or k==Nsteps:
        # Compute exact averaged solution
        Q_exact = Qexact_adv(x, t, simulation)

        # Relative errors in different metrics
        #error_linf[k], error_l1[k], error_l2[k] = compute_errors(q_parabolic, q_exact)
        error_linf[k], error_l1[k], error_l2[k] = compute_errors(Q_exact, Q[i0:iend])
        if error_linf[k] > 10**(4):
            # CFL number
            #CFL = abs(np.amax(abs(u_edges)*dt/dx))
            print('\nStopping due to large errors.')
            print('The CFL number is', CFL)
            exit()

        # Diagnostic computation
        total_mass, mass_change = diagnostics_adv_1d(Q[i0:iend], simulation, total_mass0)

        if plot:
            # Print diagnostics on the screen
            print_diagnostics_adv_1d(error_linf[k], error_l1[k], error_l2[k], mass_change, k, Nsteps)

        if (k-1)%plotstep==0 or k==Nsteps:
            # Plotting variables
            x0 = simulation.x0
            xf = simulation.xf
            Nplot = 10000
            xplot = np.linspace(x0, xf, Nplot)
            q_parabolic = np.zeros(Nplot)
            dists = abs(np.add.outer(xplot,-xc[i0:iend]))
            neighbours = dists.argmin(axis=1)

            if k!=Nsteps:
                # the parabola coeffs must be taken from the previous step
                time = t-dt
            else:
                time = t
                dq, q6, q_L,_ = ppm_reconstruction(Q, simulation) #update parabola coeffs for final step

            # Compute exact solution
            q_exact = qexact_adv(xplot, time, simulation)

            # Compute the parabola
            for i in range(0, N):
                z = (xplot[neighbours==i]-x[i+i0])/dx # Maps to [0,1]
                q_parabolic[neighbours==i] = q_L[i+i0] + dq[i+i0]*z+ z*(1.0-z)*q6[i+i0]

            # Additional plotting variables
            ymin = np.amin(q_exact)
            ymax = np.amax(q_exact)
            icname = simulation.icname
            recon_name = simulation.recon_name # Monotonization scheme
            tc = simulation.tc
            ic = simulation.ic

            # Plot the solution graph
            qmin = str("{:.2e}".format(np.amin(Q)))
            qmax = str("{:.2e}".format(np.amax(Q)))
            CFL  = str("{:.2e}".format(CFL))
            time = str("{:.2e}".format(time))

            title = '1D advection - '+icname+' - time='+str(time)+', CFL='+str(CFL)+',\n N='+str(N)+', '+simulation.recon_name+ ', Min = '+ qmin +', Max = '+qmax
            filename = graphdir+'1d_adv_tc'+str(tc)+'_ic'+str(ic)+'_t'+str(k-1)+'_N'+str(N)+'_'+simulation.recon_name+'.png'
            plot_1dfield_graphs([q_exact, q_parabolic], ['Exact', 'Parabolic'], xplot, ymin, ymax, filename, title)
