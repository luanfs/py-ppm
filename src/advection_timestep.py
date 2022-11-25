####################################################################################
#
# Piecewise Parabolic Method (PPM) time setp module
# Luan da Fonseca Santos - June 2022
####################################################################################

import numpy as np
import reconstruction_1d as rec
from monotonization_1d import monotonization_1d
from flux import numerical_flux, flux_ppm_stencil_coefficients

####################################################################################
# Applies a single timestep of PPM for the 1D advection equation
# Q is an array of size [0:N+ng] that store the average values of q.
# The interior indexes are in [i0:iend], the other indexes are used for
# periodic boundary conditions.
####################################################################################
def time_step_adv1d_ppm(Q, u_edges, F, a, simulation):
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

    # Reconstructs the values of Q using a piecewise parabolic polynomial (for monotonic case only)
    dq, q6, q_L, q_R = rec.ppm_reconstruction(Q, simulation)

    # Applies monotonization on the parabolas
    monotonization_1d(Q, q_L, q_R, dq, q6, simulation)

    # CFL at edges - x direction
    c = np.sign(u_edges)*u_edges*dt/dx
    c2 = c*c

    # Get the stencil coefs
    flux_ppm_stencil_coefficients(a, c, c2, u_edges, simulation)

    # Compute the fluxes
    numerical_flux(Q, q_R, q_L, dq, q6, u_edges, F, a, simulation)

    # Update the values of Q_average (formula 1.12 from Collela and Woodward 1984)
    Q[i0:iend] = Q[i0:iend] - (simulation.dt/simulation.dx)*(u_edges[i0+1:iend+1]*F[i0+1:iend+1] - u_edges[i0:iend]*F[i0:iend])

    # Periodic boundary conditions
    Q[iend:N+ng] = Q[i0:i0+ngr]
    Q[0:i0]      = Q[N:N+ngl]

    return Q, dq, q6, q_L, F
