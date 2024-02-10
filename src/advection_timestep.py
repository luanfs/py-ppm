####################################################################################
#
# Piecewise Parabolic Method (PPM) timestep module
# Luan da Fonseca Santos - June 2022
####################################################################################

import numpy as np
from reconstruction_1d import ppm_reconstruction
from flux import flux_ppm
from advection_ic import velocity_adv_1d
from averaged_velocity import time_averaged_velocity

####################################################################################
# Applies a single timestep of PPM for the 1D advection equation
# Q is an array of size [0:N+ng] that store the average values of q.
# The interior indexes are in [i0:iend], the other indexes are used for
# periodic boundary conditions.
####################################################################################
def time_step_adv1d_ppm(t, k, simulation):
    N = simulation.N

    # Ghost cells
    ngl = simulation.ngl
    ngr = simulation.ngr
    ng  = simulation.ng

    # Grid interior indexes
    i0 = simulation.i0
    iend = simulation.iend

    # Time step
    dt = simulation.dt

    # Grid size
    dx = simulation.dx

    # Periodic boundary conditions
    simulation.Q[iend:N+ng] = simulation.Q[i0:i0+ngr]
    simulation.Q[0:i0]      = simulation.Q[N:N+ngl]

    if simulation.vf>=2:
        simulation.U_edges.u_timecenter[:] = velocity_adv_1d(simulation.x, t-dt*0.5, simulation)

    # Compute the time averaged velocity (needed for departure point)
    time_averaged_velocity(simulation.U_edges, simulation, t)

    # CFL number
    simulation.cx[:] = simulation.U_edges.u_averaged[:]*(simulation.dt/simulation.dx) #cfl number

    # Compute the time averaged velocity (needed for departure point)
    # Reconstructs the values of Q using a piecewise parabolic polynomial
    ppm_reconstruction(simulation.Q, simulation.px, simulation)

    # Compute the fluxes
    flux_ppm(simulation.px, simulation.cx, simulation.U_edges, simulation)

    # divergence
    simulation.div[i0:iend] = (simulation.px.f_upw[i0+1:iend+1]-simulation.px.f_upw[i0:iend])/dx

    # Update the values of Q_average (formula 1.12 from Collela and Woodward 1984)
    simulation.Q[i0:iend] = simulation.Q[i0:iend] - dt*simulation.div[i0:iend] 

    # Velocity and CFL update for next time step
    if simulation.vf>=2:
        simulation.U_edges.u_old[:] = simulation.U_edges.u[:]
        simulation.U_edges.u[:] = velocity_adv_1d(simulation.x, t, simulation)

