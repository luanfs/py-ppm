####################################################################################
#
# Piecewise Parabolic Method (PPM) time setup module
# Luan da Fonseca Santos - June 2022
####################################################################################

import numpy as np
from reconstruction_1d import ppm_reconstruction
from flux import flux_ppm
from advection_ic import velocity_adv_1d
####################################################################################
# Applies a single timestep of PPM for the 1D advection equation
# Q is an array of size [0:N+ng] that store the average values of q.
# The interior indexes are in [i0:iend], the other indexes are used for
# periodic boundary conditions.
####################################################################################
def time_step_adv1d_ppm(Q, u_edges, cx, px, x, t, simulation):
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

    # CFL number
    cx[:] = u_edges*(simulation.dt/simulation.dx) #cfl number

    # Reconstructs the values of Q using a piecewise parabolic polynomial
    ppm_reconstruction(Q, px, simulation)

    # Compute the fluxes
    flux_ppm(Q, cx, px, simulation)

    # Update the values of Q_average (formula 1.12 from Collela and Woodward 1984)
    Q[i0:iend] = Q[i0:iend] - (simulation.dt/simulation.dx)*(u_edges[i0+1:iend+1]*px.f_upw[i0+1:iend+1] - u_edges[i0:iend]*px.f_upw[i0:iend])

    # Periodic boundary conditions
    Q[iend:N+ng] = Q[i0:i0+ngr]
    Q[0:i0]      = Q[N:N+ngl]

    # Velocity and CFL update for next time step
    if simulation.vf>=2:
        u_edges[0:N+ng+1] = velocity_adv_1d(x, t, simulation)
