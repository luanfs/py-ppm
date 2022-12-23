####################################################################################
#
# Piecewise Parabolic Method (PPM) time setp module
# Luan da Fonseca Santos - June 2022
####################################################################################

import numpy as np
from reconstruction_1d import ppm_reconstruction
from flux import flux_ppm

####################################################################################
# Applies a single timestep of PPM for the 1D advection equation
# Q is an array of size [0:N+ng] that store the average values of q.
# The interior indexes are in [i0:iend], the other indexes are used for
# periodic boundary conditions.
####################################################################################
def time_step_adv1d_ppm(Q, u_edges, F, simulation):
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

    # Reconstructs the values of Q using a piecewise parabolic polynomial
    dq, q6, q_L, q_R = ppm_reconstruction(Q, simulation)

    # Compute the fluxes
    flux_ppm(Q, q_R, q_L, dq, q6, u_edges, F, simulation)

    # Update the values of Q_average (formula 1.12 from Collela and Woodward 1984)
    Q[i0:iend] = Q[i0:iend] - (simulation.dt/simulation.dx)*(u_edges[i0+1:iend+1]*F[i0+1:iend+1] - u_edges[i0:iend]*F[i0:iend])

    # Periodic boundary conditions
    Q[iend:N+ng] = Q[i0:i0+ngr]
    Q[0:i0]      = Q[N:N+ngl]

    return Q, dq, q6, q_L, F
