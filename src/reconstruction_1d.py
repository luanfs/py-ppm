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
# Given the average values of a scalar field Q, this routine constructs 
# a piecewise parabolic aproximation of Q using its average value.
####################################################################################
def ppm_reconstruction(Q, simulation):
    N = simulation.N

    # Compute the slopes dQ0 (centered finite difference)
    # Formula 1.7 from Collela and Woodward 1984 and Figure 2 from Carpenter et al 1990.
    dQ0 = np.zeros(N+6)
    dQ0[1:N+5] = 0.5*(Q[2:N+6] - Q[0:N+4]) # Interior values are in 3:N+3

    if simulation.monot != 'none': #Avoid overshoot
        # Compute the slopes dQ (1-sided finite difference)
        # Formula 1.8 from Collela and Woodward 1984 and Figure 2 from Carpenter et al 1990.
        dQ1 = np.zeros(N+6) # Right 1-sided difference
        dQ2 = np.zeros(N+6) # Left  1-sided difference

        # Right 1-sided finite difference
        dQ1[1:N+5] = Q[2:N+6] - Q[1:N+5]
        dQ1 = dQ1*2.0

        # Left  1-sided finite difference
        dQ2[1:N+5] = Q[1:N+5] - Q[0:N+4]
        dQ2 = dQ2*2.0

        # Final slope - Formula 1.8 from Collela and Woodward 1984
        dQ = np.zeros(N+6)
        dQ[1:N+5] = np.minimum(abs(dQ0[1:N+5]), abs(dQ1[1:N+5]))
        dQ[1:N+5] = np.minimum(dQ[1:N+5], abs(dQ2[1:N+5]))*np.sign(dQ0[1:N+5])
        mask = ( (Q[2:N+6] - Q[1:N+5]) * (Q[1:N+5] - Q[0:N+4]) > 0.0 ) # Indexes where (Q_{j+1}-Q_{j})*(Q_{j}-Q{j-1}) > 0
        dQ[1:N+5][~mask] = 0.0
    else:
        dQ = dQ0

    # Values of Q at right edges (q_(j+1/2)) - Formula 1.6 from Collela and Woodward 1984
    Q_edges = np.zeros(N+7)
    Q_edges[2:N+5] = 0.5*(Q[2:N+5] + Q[1:N+4]) - (dQ[2:N+5] - dQ[1:N+4])/6.0

    # Assign values of Q_R and Q_L
    q_R = np.zeros(N+6)
    q_R[2:N+4] = Q_edges[3:N+5]
    q_L = np.zeros(N+6)
    q_L[2:N+4] = Q_edges[2:N+4]

    # Compute the polynomial coefs
    # q(x) = q_L + z*(dq + q6*(1-z)) z in [0,1]
    dq = np.zeros(N+6)
    q6 = np.zeros(N+6)
    dq[2:N+4] = q_R[2:N+4] - q_L[2:N+4]
    q6[2:N+4] = 6*Q[2:N+4] - 3*(q_R[2:N+4] + q_L[2:N+4])

    return dq, q6, q_L, q_R
