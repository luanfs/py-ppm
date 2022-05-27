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
from miscellaneous import diagnostics, print_diagnostics, plot_sol_graphs

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
    Q_average = np.zeros(N+5)
    if (simulation.ic == 0 or simulation.ic == 1 or simulation.ic == 3 or simulation.ic == 4):
        Q_average[0:N] = (q0_antiderivative(x[1:N+1], simulation) - q0_antiderivative(x[0:N], simulation))/dx
    elif (simulation.ic == 2):
        Q_average[0:N] = q0(xc,simulation)

    # Periodic boundary conditions
    Q_average[N]   = Q_average[0]
    Q_average[N+1] = Q_average[1]
    Q_average[N+2] = Q_average[3]
    Q_average[-2]  = Q_average[N-2]
    Q_average[-1]  = Q_average[N-1]

    # Plotting variables
    Nplot = 10000
    xplot = np.linspace(x0, xf, Nplot)
    Q_parabolic = np.zeros(Nplot)
    Q_exact = q0(xplot, simulation)
    ymin = np.amin(Q_exact)
    ymax = np.amax(Q_exact)
    dists = abs(np.add.outer(xplot,-xc))
    neighbours = dists.argmin(axis=1)

    # Fluxes
    f_L = np.zeros(N)
    f_R = np.zeros(N)

    # Aux. variables
    abar = np.zeros(N+1)   
    minus1toN = np.linspace(-1, N-2 ,N,dtype=np.int32)

    # Compute initial mass
    total_mass0, mass_change =  diagnostics(Q_average, simulation, 1.0)

    # Errors variable
    error_linf = np.zeros(Nsteps)
    error_l1  = np.zeros(Nsteps)
    error_l2  = np.zeros(Nsteps)

    # Time looping
    for t in range(0, Nsteps):
        # Reconstructs the values of Q using a piecewise parabolic polynomial
        da, a6, aL, aR = rec.ppm_reconstruction(Q_average, N)

        # Compute the fluxes (formula 1.11 from Collela and Woodward 1983)
        y = u*dt/dx
        
        if u>=0:
            f_L = aR[0:N]   - y*0.5*(da[0:N]   - (1.0-2.0/3.0*y)*a6[0:N])
            abar[0:N] = f_L
        else:
            y = -y
            f_R = aL[1:N+1] + y*0.5*(da[1:N+1] + (1.0-2.0/3.0*y)*a6[1:N+1])
            abar[0:N] = f_R

        # Periodic boundary conditions
        abar[-1] = abar[N-1]

        # Update the values of Q_average (formula 1.12 from Collela and Woodward 1983)
        Q_average[0:N] = Q_average[0:N] + (u*dt/dx)*(abar[minus1toN]-abar[0:N])

        # Periodic boundary conditions
        Q_average[N]   = Q_average[0]
        Q_average[N+1] = Q_average[1]
        Q_average[N+2] = Q_average[3]
        Q_average[-2]  = Q_average[N-2]
        Q_average[-1]  = Q_average[N-1]

        # Output and plot
        if plot:
            # Compute the parabola
            for i in range(0, N):
                z = (xplot[neighbours==i]-x[i])/dx # Maps to [0,1]
                Q_parabolic[neighbours==i] = aL[i] + da[i]*z+ z*(1.0-z)*a6[i]

            # Compute exact solution
            Q_exact = qexact(xplot, t*dt, simulation)
    
            # Diagnostic computation
            total_mass, mass_change =  diagnostics(Q_average, simulation, total_mass0)

            # Relative errors in different metrics
            error_linf[t], error_l1[t], error_l2[t] = compute_errors(Q_parabolic, Q_exact)

            if error_linf[t] > 10**(4):
                print('\nStopping due to large errors.')
                print('The CFL number is', CFL)
                exit()

            # Plot the graph and print diagnostic
            title = name+' - 1d advection, time='+str(t*dt)+', CFL='+str(CFL)
            filename = graphdir+'adv1d_ppm_ic'+str(ic)+'_t'+str(t+1)+'.png'
            plot_sol_graphs(Q_exact, Q_parabolic, xplot, ymin, ymax, filename, title)
            print_diagnostics(error_linf[t], error_l1[t], error_l2[t], mass_change, t, Nsteps)

    if plot:
        # Plot the error graph
        title = name+' - 1d advection errors - CFL='+str(CFL)
        filename = graphdir+'adv1d_ppm_ic'+str(ic)+'_error.png'
        plot_errors_evolution(error_linf, error_l1, error_l2, Tf, filename, title)
        
        # Plot the solution graph
        title = name+' - 1d advection, time='+str(t*dt)+', CFL='+str(CFL)
        filename = graphdir+'adv1d_ppm_ic'+str(ic)+'_t'+str(t+1)+'.png'
        plot_sol_graphs(Q_exact, Q_parabolic, xplot, ymin, ymax, filename, title)
        print('\nGraphs have been ploted in '+ graphdir)
        print('Error evolution is shown in '+ graphdir + 'adv1d_ppm_ic' + str(ic) + '_error.png')      
   
    else:
        # Compute the parabola
        for i in range(0, N):
            z = (xplot[neighbours==i]-x[i])/dx # Maps to [0,1]
            Q_parabolic[neighbours==i] = aL[i] + da[i]*z+ z*(1.0-z)*a6[i]
        # Compute exact solution
        Q_exact = qexact(xplot, t*dt, simulation)
    
        #Relative errors in different metrics
        error_inf, error_1, error_2 = compute_errors(Q_parabolic, Q_exact)

        # Plot the graph
        plt.plot(xplot, Q_exact, color='black',label='Exact')
        plt.plot(xplot, Q_parabolic, color='blue',label='PPM')
        plt.ylim(1.1*ymin, 1.1*ymax)
        plt.ylabel('y')
        plt.xlabel('x') 
        plt.legend()
        plt.title(name+' - 1d advection, time='+str(t*dt)+', CFL='+str(CFL))
        plt.savefig(graphdir+'adv1d_ppm_ic'+str(ic)+'_t'+str(t+1)+'_N'+str(N)+'.png', format='png')
        plt.close() 
        return error_inf, error_1, error_2

