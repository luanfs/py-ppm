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
    # Reconstructs the values of Q using a piecewise parabolic polynomial
    dq, q6, q_L, q_R = rec.ppm_reconstruction(Q, N, simulation)

    # Applies monotonization on the parabolas
    monotonization_1d(Q, q_L, q_R, dq, q6, N, simulation.mono)

    # Compute the fluxes
    F = numerical_flux(Q, q_R, q_L, dq, q6, u_edges, simulation, N)

    # Update the values of Q_average (formula 1.12 from Collela and Woodward 1984)
    Q[3:N+3] = Q[3:N+3] - (simulation.dt/simulation.dx)*(u_edges[1:N+1]*F[1:N+1] - u_edges[0:N]*F[0:N])

    # Periodic boundary conditions
    Q[N+3:N+6] = Q[3:6]
    Q[0:3]     = Q[N:N+3]

    return Q, dq, q6, q_L, F
