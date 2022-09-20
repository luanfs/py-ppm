####################################################################################
#
# Piecewise Parabolic Method (PPM) time setp module
# Luan da Fonseca Santos - June 2022
####################################################################################

import numpy as np
import reconstruction_1d as rec
from monotonization_1d import monotonization_1d
from flux import numerical_flux

####################################################################################
# Applies a single timestep of PPM for the 1D advection equation
# Q is an array of size [0:N+5] that store the average values of q.
# The interior indexes are in [2:N+2], the other indexes are used for
# periodic boundary conditions.
####################################################################################
def time_step_adv1d_ppm(Q, u_edges, N, simulation):
    i0    = simulation.i0         # Interior of grid indexes
    iend  = simulation.iend
    ng    = simulation.ng         # Total number of ghost cells
    ngr   = simulation.ng_right   # Number of ghost cells at right
    ngl   = simulation.ng_left    # Number of ghost cells at left

    # Reconstructs the values of Q using a piecewise parabolic polynomial
    dq, q6, q_L, q_R = rec.ppm_reconstruction(Q, N, simulation)

    # Applies monotonization on the parabolas
    monotonization_1d(Q, q_L, q_R, dq, q6, N, simulation)

    # Compute the fluxes
    F = numerical_flux(Q, q_R, q_L, dq, q6, u_edges, simulation, N)

    # Update the values of Q_average (formula 1.12 from Collela and Woodward 1984)
    Q[i0:iend] = Q[i0:iend] - (simulation.dt/simulation.dx)*(u_edges[i0+1:iend+1]*F[1:N+1] - u_edges[i0:iend]*F[0:N])

    # Periodic boundary conditions
    Q[iend:N+ng+1] = Q[i0:i0+ngr]
    Q[0:i0]        = Q[iend-ngl:iend]

    return Q, dq, q6, q_L, F
