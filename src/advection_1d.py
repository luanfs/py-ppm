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
from miscellaneous import diagnostics_adv_1d, print_diagnostics_adv_1d, plot_1dfield_graphs, output_adv
from timestep import time_step_adv1d_ppm
from flux import flux_ppm_stencil_coefficients

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
    u_edges = np.zeros(N+7)
    u_edges[0:N+7] = velocity_adv_1d(x[0:N+7], 0, simulation)

    # CFL number
 
    # CFL at edges - x direction
    c = np.sign(u_edges)*u_edges*dt/dx
    c2 = c*c
    CFL = abs(np.amax(c*dt/dx))

    # Number of time steps
    Nsteps = int(Tf/dt)

    # Compute average values of Q (initial condition)
    Q = np.zeros(N+6)
    if (simulation.ic == 0 or simulation.ic == 1 or simulation.ic == 3 or simulation.ic == 4):
        Q[3:N+3] = (q0_antiderivative_adv(x[4:N+4], simulation) - q0_antiderivative_adv(x[3:N+3], simulation))/dx
    elif (simulation.ic == 2 or simulation.ic == 5):
        Q[3:N+3] = q0_adv(xc[3:N+3],simulation)

    # Periodic boundary conditions
    Q[N+3:N+6] = Q[3:6]
    Q[0:3]     = Q[N:N+3]

    # Compute initial mass
    total_mass0, mass_change = diagnostics_adv_1d(Q[3:N+3], simulation, 1.0)

    # Error variables
    error_linf = np.zeros(Nsteps+1)
    error_l1   = np.zeros(Nsteps+1)
    error_l2   = np.zeros(Nsteps+1)

    # Plot timestep
    plotstep = 100

    # Aux. variables
    F = np.zeros(N+7) # Numerical flux

    # Stencil coefficients
    a = np.zeros((6, N+7))
    flux_ppm_stencil_coefficients(a, c, c2, u_edges, simulation)

    #-------------------Time looping-------------------
    for k in range(1, Nsteps+1):
        # Time
        t = k*dt

        # Applies a PPM time step
        Q, dq, q6, q_L, _ = time_step_adv1d_ppm(Q, u_edges, F, a, simulation)

        # Velocity update
        u_edges[0:N+7] = velocity_adv_1d(x, t*dt, simulation)

        # Output
        output_adv(x, xc, simulation, Q, dq, q6, q_L, error_linf, error_l1, error_l2, plot, k, t, Nsteps, plotstep, total_mass0, CFL)
    # -------------------End of time loop-------------------

    #-------------------Final plot and outputs -------------------
    if plot:
        # Plot the error graph
        title = simulation.title +'- '+icname+', CFL='+str(CFL)+',\n N='+str(N)+', '+simulation.fvmethod+', mono = '+simulation.monot
        filename = graphdir+'1d_adv_tc'+str(tc)+'_ic'+str(ic)+'_N'+str(N)+'_'+simulation.fvmethod+'_mono'+simulation.monot+'_erros.png'
        plot_time_evolution([error_linf, error_l1, error_l2], Tf, ['$L_\infty}$','$L_1$','$L_2$'], 'Error', filename, title)
        print('\nGraphs have been ploted in '+ graphdir)
        print('Error evolution is shown in '+filename)
    else:
        # Return final errors
        return error_linf[Nsteps], error_l1[Nsteps], error_l2[Nsteps]
