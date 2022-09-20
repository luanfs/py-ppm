####################################################################################
#
# Module for PPM numerical flux computation
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
# Luan da Fonseca Santos - June 2022
# (luan.santos@usp.br)
####################################################################################
import numpy as np
def numerical_flux(Q, q_R, q_L, dq, q6, u_edges, simulation, N):    
    i0    = simulation.i0         # Interior of grid indexes
    iend  = simulation.iend
    ng    = simulation.ng         # Total number of ghost cells
    ngr   = simulation.ng_right   # Number of ghost cells at right
    ngl   = simulation.ng_left    # Number of ghost cells at left

    # Numerical fluxes at edges
    f_L = np.zeros(N+1) # Left
    f_R = np.zeros(N+1) # Right

    # Compute the fluxes (formula 1.12 from Collela and Woodward 1984)
    c = u_edges*(simulation.dt/simulation.dx) #cfl number
    c2 = c*c

    # Flux from left at edges
    f_L[0:N+1] = q_R[i0-1:iend] - c[i0-1:iend]*0.5*(dq[i0-1:iend] - (1.0-(2.0/3.0)*c[i0-1:iend])*q6[i0-1:iend])

    # Flux from right at edges
    c = -c
    f_R[0:N+1] = q_L[i0:iend+1] + c[i0:iend+1]*0.5*(dq[i0:iend+1] + (1.0-2.0/3.0*c[i0:iend+1])*q6[i0:iend+1])

    # F - Formula 1.13 from Collela and Woodward 1984)
    F = np.zeros(N+1) # Numerical flux
    mask = u_edges[i0:iend+1] >=0

    F[u_edges[i0:iend+1] >= 0] = f_L[u_edges[i0:iend+1] >= 0]
    F[u_edges[i0:iend+1] <= 0] = f_R[u_edges[i0:iend+1] <= 0]
    return F
