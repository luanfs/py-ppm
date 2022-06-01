####################################################################################
#
# Piecewise Parabolic Method (PPM) polynomial reconstruction module
# Luan da Fonseca Santos - March 2022
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

####################################################################################
# Given the average values of a scalar field Q (Q), this routine constructs 
# a piecewise parabolic aproximation of Q using its average value.
####################################################################################
def ppm_reconstruction(Q, N):
    # Compute the slopes dQ0 (centered finite difference)
    # Formula 1.7 from Collela and Woodward 1983 and Figure 2 from Carpenter et al 1989.
    # interior = np.linspace(2, N+1, N, dtype=np.int32)
    dQ0 = np.zeros(N+5)
    dQ0[1:N+4] = 0.5*(Q[2:N+5] - Q[0:N+3]) # Interior values are in 2:N+2

    # Compute the slopes dQ (1-sided finite difference)
    # Formula 1.8 from Collela and Woodward 1983 and Figure 2 from Carpenter et al 1989.
    dQ1 = np.zeros(N+5) # Right 1-sided difference
    dQ2 = np.zeros(N+5) # Left  1-sided difference

    # Right 1-sided finite difference
    dQ1[0:N+4] = Q[1:N+5] - Q[0:N+4]
    dQ1 = 2.0*dQ1

    # Left  1-sided finite difference
    dQ2[1:N+5] = Q[1:N+5] - Q[0:N+4]
    dQ2 = 2.0*dQ2

    # Final slope - Formula 1.8 from Collela and Woodward 1983
    dQ = np.minimum(abs(dQ0), abs(dQ1))
    dQ = np.minimum(dQ, abs(dQ2))*np.sign(dQ0)
    dQ = dQ0

    # Values of Q at right edges (Q_(j+1/2)) - Formula 1.6 from Collela and Woodward 1983
    Q_edges = np.zeros(N+5)
    Q_edges[0:N+4] = 0.5*(Q[0:N+4] + Q[1:N+5]) - (dQ[1:N+5] - dQ[0:N+4])/6.0

    # Assign values of Q_R and Q_L
    q_R = np.zeros(N+5)
    q_R[2:N+2] = Q_edges[2:N+2]
    q_L = np.zeros(N+5)
    q_L[2:N+2] = Q_edges[1:N+1]

    # Compute the polynomial coefs
    # q(x) = q_L + z*(dq + q6*(1-z)) z in [0,1]
    dq = np.zeros(N+5)
    q6 = np.zeros(N+5)
    dq[2:N+2] = q_R[2:N+2] - q_L[2:N+2]
    q6[2:N+2] = 6*Q[2:N+2] - 3*(q_R[2:N+2] + q_L[2:N+2])

    # Periodic boundary conditions
    q_L[N+2:N+5] = q_L[2:5]
    q_L[0:2]     = q_L[N:N+2]

    q_R[N+2:N+5] = q_R[2:5]
    q_R[0:2]     = q_R[N:N+2]

    q6[N+2:N+5] = q6[2:5]
    q6[0:2]     = q6[N:N+2]

    dq[N+2:N+5] = dq[2:5]
    dq[0:2]     = dq[N:N+2]

    return dq, q6, q_L, q_R
