####################################################################################
#
# Piecewise Parabolic Method (PPM) advection module
# Luan da Fonseca Santos - April 2022
# Solves the PDE Q_t+ u*Q_x = 0 with periodic boundary conditions
# The initial condition Q(x,0) is given in the module parameters_1d.py
#
# References:
# -  Phillip Colella, Paul R Woodward, The Piecewise Parabolic Method (PPM) for gas-dynamical simulations,
# Journal of Computational Physics, Volume 54, Issue 1, 1984, Pages 174-201, ISSN 0021-9991,
# https://doi.org/10.1016/0021-9991(84)90143-8.
#
# -  Carpenter , R. L., Jr., Droegemeier, K. K., Woodward, P. R., & Hane, C. E. (1990).
# Application of the Piecewise Parabolic Method (PPM) to Meteorological Modeling, Monthly Weather Review, 118(3),
# 586-612. Retrieved Mar 31, 2022,
# from https://journals.ametsoc.org/view/journals/mwre/118/3/1520-0493_1990_118_0586_aotppm_2_0_co_2.xml
#
####################################################################################

import numpy as np
from parameters_1d import q0_adv, qexact_adv, Qexact_adv, q0_antiderivative_adv, graphdir, velocity_adv_1d
from errors import *
from miscellaneous import diagnostics_adv_1d, print_diagnostics_adv_1d, plot_1dfield_graphs
from timestep import time_step_adv1d_ppm

def adv_1d(simulation, plot):
    N  = simulation.N    # Number of cells
    ic = simulation.ic   # Initial condition
    x  = simulation.x    # Grid
    xc = simulation.xc
    x0 = simulation.x0
    xf = simulation.xf
    dx = simulation.dx   # Grid spacing
    dt = simulation.dt   # Time step
    Tf = simulation.Tf   # Total period definition
    tc = simulation.tc
    icname = simulation.icname
    mono   = simulation.mono  # Monotonization scheme

    # Velocity at edges
    u_edges = np.zeros(N+1)
    u_edges[0:N+1] = velocity_adv_1d(x, 0, simulation)

    # CFL number
    CFL = abs(np.amax(abs(u_edges)*dt/dx))

    # Number of time steps
    Nsteps = int(Tf/dt)

    # Compute average values of Q (initial condition)
    Q = np.zeros(N+5)
    if (simulation.ic == 0 or simulation.ic == 1 or simulation.ic == 3 or simulation.ic == 4 or simulation.ic == 5):
        Q[2:N+2] = (q0_antiderivative_adv(x[1:N+1], simulation) - q0_antiderivative_adv(x[0:N], simulation))/dx
    elif (simulation.ic == 2):
        Q[2:N+2] = q0_adv(xc,simulation)
        #Q[2:N+2] = q0_antiderivative_adv(x, simulation)/dx

    # Periodic boundary conditions
    Q[N+2:N+5] = Q[2:5]
    Q[0:2]     = Q[N:N+2]

    # Plotting variables
    Nplot = 10000
    xplot = np.linspace(x0, xf, Nplot)
    q_parabolic = np.zeros(Nplot)
    q_exact = q0_adv(xplot, simulation)
    ymin = np.amin(q_exact)
    ymax = np.amax(q_exact)
    dists = abs(np.add.outer(xplot,-xc))
    neighbours = dists.argmin(axis=1)

    # Compute initial mass
    total_mass0, mass_change = diagnostics_adv_1d(Q, simulation, 1.0)

    # Error variables
    error_linf = np.zeros(Nsteps+1)
    error_l1   = np.zeros(Nsteps+1)
    error_l2   = np.zeros(Nsteps+1)

    # Time looping
    for t in range(1, Nsteps+1):
        # Velocity update
        u_edges[0:N+1] = velocity_adv_1d(x, t*dt, simulation)

        # Applies a PPM time step
        Q, dq, q6, q_L, _ = time_step_adv1d_ppm(Q, u_edges, N, simulation)

        # Output and plot
        if plot:
            # Compute the parabola
            for i in range(0, N):
                z = (xplot[neighbours==i]-x[i])/dx # Maps to [0,1]
                q_parabolic[neighbours==i] = q_L[i+2] + dq[i+2]*z+ z*(1.0-z)*q6[i+2]

            # Compute exact solution
            q_exact = qexact_adv(xplot, t*dt, simulation)

            # Compute exact averaged solution
            Q_exact = Qexact_adv(x, t*dt, simulation)

            # Diagnostic computation
            total_mass, mass_change = diagnostics_adv_1d(Q, simulation, total_mass0)

            # Relative errors in different metrics
            #error_linf[t], error_l1[t], error_l2[t] = compute_errors(q_parabolic, q_exact)
            error_linf[t], error_l1[t], error_l2[t] = compute_errors(Q_exact, Q[2:N+2])
            if error_linf[t] > 10**(4):
                # CFL number
                CFL = abs(np.amax(abs(u_edges)*dt/dx))
                print('\nStopping due to large errors.')
                print('The CFL number is', CFL)
                exit()

            # Plot the graph and print diagnostic
            #title = simulation.title +'- '+icname+' - time='+str(t*dt)+', CFL='+str(CFL)+',\n N='+str(N)+', '+simulation.fvmethod+', mono = '+simulation.monot
            #filename = graphdir+'1d_adv_tc'+str(tc)+'_ic'+str(ic)+'_t'+str(t)+'_N'+str(N)+'_'+simulation.fvmethod+'_mono'+simulation.monot+'.png'
            #plot_1dfield_graphs([q_exact, q_parabolic], ['Exact', 'Parabolic'], xplot, ymin, ymax, filename, title)
            print_diagnostics_adv_1d(error_linf[t], error_l1[t], error_l2[t], mass_change, t, Nsteps)
    #---------------------------------------End of time loop---------------------------------------

    if plot:
        # Plot the error graph
        title = simulation.title +'- '+icname+', CFL='+str(CFL)+',\n N='+str(N)+', '+simulation.fvmethod+', mono = '+simulation.monot
        filename = graphdir+'1d_adv_tc'+str(tc)+'_ic'+str(ic)+'_N'+str(N)+'_'+simulation.fvmethod+'_mono'+simulation.monot+'_erros.png'
        plot_time_evolution([error_linf, error_l1, error_l2], Tf, ['$L_\infty}$','$L_1$','$L_2$'], 'Error', filename, title)

        # Plot the solution graph
        qmin = str("{:.2e}".format(np.amin(q_parabolic)))
        qmax = str("{:.2e}".format(np.amax(q_parabolic)))
        title = '1D advection - '+icname+' - time='+str(t*dt)+', CFL='+str(CFL)+',\n N='+str(N)+', '+simulation.fvmethod+', mono = '+simulation.monot+ ', Min = '+ qmin +', Max = '+qmax
        filename = graphdir+'1d_adv_tc'+str(tc)+'_ic'+str(ic)+'_t'+str(t)+'_N'+str(N)+'_'+simulation.fvmethod+'_mono'+simulation.monot+'.png'
        plot_1dfield_graphs([q_exact, q_parabolic], ['Exact', 'Parabolic'], xplot, ymin, ymax, filename, title)
        print('\nGraphs have been ploted in '+ graphdir)
        print('Error evolution is shown in '+filename)

    else:
        # Compute exact solution
        Q_exact = Qexact_adv(x, Tf, simulation)

        q_exact_edges = qexact_adv(x, Tf, simulation)

        # Relative errors in different metrics
        error_inf, error_1, error_2 = compute_errors(Q[2:N+2], Q_exact)

        # Applies a PPM time step
        Q, dq, q6, q_L, _ = time_step_adv1d_ppm(Q, u_edges, N, simulation)

        # Compute the parabola
        for i in range(0, N):
            z = (xplot[neighbours==i]-x[i])/dx # Maps to [0,1]
            q_parabolic[neighbours==i] = q_L[i+2] + dq[i+2]*z+ z*(1.0-z)*q6[i+2]
        
        # Error at edges
        error_ed_linf, error_ed_l1, error_ed_l2 = compute_errors(q_exact_edges[0:N], q_L[2:N+2])

        # Plot the solution graph
        qmin = str("{:.2e}".format(np.amin(q_parabolic)))
        qmax = str("{:.2e}".format(np.amax(q_parabolic)))
        title = '1D advection - '+icname+' - time='+str(t*dt)+', CFL='+str(CFL)+',\n N='+str(N)+', '+simulation.fvmethod+', mono = '+simulation.monot+ ', Min = '+ qmin +', Max = '+qmax
        filename = graphdir+'1d_adv_tc'+str(tc)+'_ic'+str(ic)+'_t'+str(t)+'_N'+str(N)+'_'+simulation.fvmethod+'_mono'+simulation.monot+'.png'
        plot_1dfield_graphs([q_exact, q_parabolic], ['Exact', 'Parabolic'], xplot, ymin, ymax, filename, title)

        return error_inf, error_1, error_2, error_ed_linf, error_ed_l1, error_ed_l2
