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
from errors              import *
from output              import print_diagnostics_adv_1d, output_adv
from diagnostics         import diagnostics_adv_1d
from plot                import plot_1dfield_graphs
from advection_timestep  import time_step_adv1d_ppm
from advection_vars      import adv_vars
from parameters_1d       import graphdir

def adv_1d(simulation, plot):
    dt = simulation.dt   # Time step
    Tf = simulation.Tf   # Total period definition

    # Grid interior indexes
    i0 = simulation.i0
    iend = simulation.iend

    # Number of time steps
    Nsteps = int(Tf/dt)

    # Plot timestep
    plotstep = int(Nsteps/5)

    # Get vars
    Q, U_edges, px, cx, x, xc, CFL = adv_vars(simulation)

    # Compute initial mass
    total_mass0, mass_change = diagnostics_adv_1d(Q[i0:iend], simulation, 1.0)

    # Error variables
    error_linf = np.zeros(Nsteps+1)
    error_l1   = np.zeros(Nsteps+1)
    error_l2   = np.zeros(Nsteps+1)

    #-------------------Time looping-------------------
    for k in range(1, Nsteps+1):
        # Time
        t = k*dt

        # PPM time step
        time_step_adv1d_ppm(Q, U_edges, cx, px, x, t, k, simulation)

        # Output
        output_adv(x, xc, simulation, Q, px, error_linf, error_l1, error_l2, plot, k, t, Nsteps, plotstep, total_mass0, CFL)
    # -------------------End of time loop-------------------

    #-------------------Final plot and outputs -------------------
    if plot:
        CFL = str("{:.2e}".format(CFL))
        # Plot the error graph
        title = simulation.title +'- '+simulation.icname+', CFL='+str(CFL)+',\n N='+str(simulation.N)+', '+simulation.recon_name
        filename = graphdir+'1d_adv_tc'+str(simulation.tc)+'_ic'+str(simulation.ic)+'_vf'+str(simulation.vf)+'_N'+str(simulation.N)+'_'+simulation.recon_name+'_dp'+simulation.dp_name
        plot_time_evolution([error_linf, error_l1, error_l2], Tf, ['$L_\infty}$','$L_1$','$L_2$'], 'Error', filename, title)
        print('\nGraphs have been ploted in '+ graphdir)
        print('Error evolution is shown in '+filename)
    else:
        # Return final errors
        return error_linf[Nsteps], error_l1[Nsteps], error_l2[Nsteps]
