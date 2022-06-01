####################################################################################
#
# Piecewise Parabolic Method (PPM) advection module
# Luan da Fonseca Santos - April 2022
# Solves the PDE Q_t+ u*Q_x = 0 with periodic boundary conditions
# The initial condition Q(x,0) is given in the module parameters.py
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
import matplotlib.pyplot as plt
from parameters import q0, qexact, q0_antiderivative, graphdir
import reconstruction as rec
from errors import *
from miscellaneous import diagnostics, print_diagnostics, plot_field_graphs

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
    u  = simulation.u    # Advection velocity
    name = simulation.name

    # CFL number
    CFL = u*dt/dx

    # Number of time steps
    Nsteps = int(Tf/dt)

    # Compute average values of Q (initial condition)
    Q = np.zeros(N+5)
    if (simulation.ic == 0 or simulation.ic == 1 or simulation.ic == 3 or simulation.ic == 4):
        Q[2:N+2] = (q0_antiderivative(x[1:N+1], simulation) - q0_antiderivative(x[0:N], simulation))/dx
    elif (simulation.ic == 2):
        Q[2:N+2] = q0(xc,simulation)

    # Periodic boundary conditions
    Q[N+2:N+5] = Q[2:5]
    Q[0:2]     = Q[N:N+2]

    # Plotting variables
    Nplot = 10000
    xplot = np.linspace(x0, xf, Nplot)
    q_parabolic = np.zeros(Nplot)
    q_exact = q0(xplot, simulation)
    ymin = np.amin(q_exact)
    ymax = np.amax(q_exact)
    dists = abs(np.add.outer(xplot,-xc))
    neighbours = dists.argmin(axis=1)

    # Fluxes
    f_L = np.zeros(N)
    f_R = np.zeros(N)

    # Aux. variables
    abar = np.zeros(N+1)   
    minus1toN = np.linspace(-1, N-2 ,N,dtype=np.int32)

    # Compute initial mass
    total_mass0, mass_change = diagnostics(Q, simulation, 1.0)

    # Errors variable
    error_linf = np.zeros(Nsteps+1)
    error_l1   = np.zeros(Nsteps+1)
    error_l2   = np.zeros(Nsteps+1)

    # Time looping
    for t in range(0, Nsteps+1):
        # Reconstructs the values of Q using a piecewise parabolic polynomial
        da, a6, Q_L, Q_R = rec.ppm_reconstruction(Q, N)

        # Compute the fluxes (formula 1.11 from Collela and Woodward 1983)
        y = u*dt/dx
        
        if u>=0:
            f_L = Q_R[2:N+2] - y*0.5*(da[2:N+2] - (1.0-2.0/3.0*y)*a6[2:N+2])
            abar[0:N] = f_L
        else:
            y = -y
            f_R = Q_L[3:N+3] + y*0.5*(da[3:N+3] + (1.0-2.0/3.0*y)*a6[3:N+3])
            abar[0:N] = f_R

        # Periodic boundary conditions
        abar[-1] = abar[N-1]

        # Update the values of Q_average (formula 1.12 from Collela and Woodward 1983)
        Q[2:N+2] = Q[2:N+2] + (u*dt/dx)*(abar[minus1toN]-abar[0:N])

        # Periodic boundary conditions
        Q[N+2:N+5] = Q[2:5]
        Q[0:2]     = Q[N:N+2]

        # Output and plot
        if plot:
            # Compute the parabola
            for i in range(0, N):
                z = (xplot[neighbours==i]-x[i])/dx # Maps to [0,1]
                q_parabolic[neighbours==i] = Q_L[i+2] + da[i+2]*z+ z*(1.0-z)*a6[i+2]

            # Compute exact solution
            q_exact = qexact(xplot, t*dt, simulation)
    
            # Diagnostic computation
            total_mass, mass_change = diagnostics(Q, simulation, total_mass0)

            # Relative errors in different metrics
            error_linf[t], error_l1[t], error_l2[t] = compute_errors(q_parabolic, q_exact)

            if error_linf[t] > 10**(4):
                print('\nStopping due to large errors.')
                print('The CFL number is', CFL)
                exit()

            # Plot the graph and print diagnostic
            title = name+' - 1d advection, time='+str(t*dt)+', CFL='+str(CFL)+', N='+str(N)
            filename = graphdir+'adv1d_ppm_ic'+str(ic)+'_t'+str(t)+'_N'+str(N)+'.png'
            plot_field_graphs([q_exact, q_parabolic], ['Exact', 'Parabolic'], xplot, ymin, ymax, filename, title)
            print_diagnostics(error_linf[t], error_l1[t], error_l2[t], mass_change, t, Nsteps)

    #---------------------------------------End of time loop---------------------------------------

    if plot:
        # Plot the error graph
        title = name+' - 1d advection, time='+str(t*dt)+', CFL='+str(CFL)+', N='+str(N)
        filename = graphdir+'adv1d_ppm_ic'+str(ic)+'_t'+str(t)+'_N'+str(N)+'.png'    
        plot_time_evolution([error_linf, error_l1, error_l2], Tf, ['$L_\infty}$','$L_1$','$L_2$'], 'Error', filename, title)
 
        # Plot the solution graph
        title = name+' - 1d advection, time='+str(t*dt)+', CFL='+str(CFL)+', N='+str(N)
        filename = graphdir+'adv1d_ppm_ic'+str(ic)+'_t'+str(t)+'_N'+str(N)+'.png'
        plot_field_graphs([q_exact, q_parabolic], ['Exact', 'Parabolic'], xplot, ymin, ymax, filename, title)
        print('\nGraphs have been ploted in '+ graphdir)
        print('Error evolution is shown in '+ graphdir + 'adv1d_ppm_ic' + str(ic) + '_error.png')      
   
    else:
        # Compute the parabola
        for i in range(0, N):
            z = (xplot[neighbours==i]-x[i])/dx # Maps to [0,1]
            q_parabolic[neighbours==i] = Q_L[i+2] + da[i+2]*z+ z*(1.0-z)*a6[i+2]

        # Compute exact solution
        q_exact = qexact(xplot, Tf, simulation)
    
        # Relative errors in different metrics
        error_inf, error_1, error_2 = compute_errors(q_parabolic, q_exact)

        # Plot the graph
        title = name+' - 1d advection, time='+str(t*dt)+', CFL='+str(CFL)+', N='+str(N)
        filename = graphdir+'adv1d_ppm_ic'+str(ic)+'_t'+str(t)+'_N'+str(N)+'.png'
        plot_field_graphs([q_exact, q_parabolic], ['Exact', 'Parabolic'], xplot, ymin, ymax, filename, title)
        return error_inf, error_1, error_2

