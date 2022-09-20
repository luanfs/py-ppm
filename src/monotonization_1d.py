####################################################################################
#
# Piecewise Parabolic Method (PPM) monotonization module
# Luan da Fonseca Santos - June 2022
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
# Apply the 1D PPM monotonization
####################################################################################
def monotonization_1d(Q, q_L, q_R, dq, q6, N, simulation):
    if simulation.mono == 0:
        return
    elif simulation.mono == 1:
        i0    = simulation.i0         # Interior of grid indexes
        iend  = simulation.iend
        ng    = simulation.ng         # Total number of ghost cells
        ngr   = simulation.ng_right   # Number of ghost cells at right
        ngl   = simulation.ng_left    # Number of ghost cells at left

        # In each cell, check if Q is a local maximum
        # See First equation in formula 1.10 from Collela and Woodward 1984
        local_maximum = (q_R[i0-1:iend+1]-Q[i0-1:iend+1])*(Q[i0-1:iend+1]-q_L[i0-1:iend+1])<=0

        # In this case (local maximum), the interpolation is a constant equal to Q
        q_R[i0-1:iend+1][local_maximum] = Q[i0-1:iend+1][local_maximum]
        q_L[i0-1:iend+1][local_maximum] = Q[i0-1:iend+1][local_maximum]
       
        # Check overshot
        overshoot  = (abs(dq[i0-1:iend+1]) < abs(q6[i0-1:iend+1]))

        # Move left
        move_left  = (q_R[i0-1:iend+1]-q_L[i0-1:iend+1])*(Q[i0-1:iend+1]-0.5*(q_R[i0-1:iend+1]+q_L[i0-1:iend+1])) > ((q_R[i0-1:iend+1]-q_L[i0-1:iend+1])**2)/6.0

        # Move right
        move_right = (-((q_R[i0-1:iend+1]-q_L[i0-1:iend+1])**2)/6.0 > (q_R[i0-1:iend+1]-q_L[i0-1:iend+1])*(Q[i0-1:iend+1]-0.5*(q_R[i0-1:iend+1]+q_L[i0-1:iend+1])) )

        overshoot_move_left  = np.logical_and(overshoot, move_left)
        overshoot_move_right = np.logical_and(overshoot, move_right)

        q_L[i0-1:iend+1][overshoot_move_left]  = 3.0*Q[i0-1:iend+1][overshoot_move_left]  - 2.0*q_R[i0-1:iend+1][overshoot_move_left]
        q_R[i0-1:iend+1][overshoot_move_right] = 3.0*Q[i0-1:iend+1][overshoot_move_right] - 2.0*q_L[i0-1:iend+1][overshoot_move_right]

        # Update the polynomial coefs
        dq[i0-1:iend+1] = q_R[i0-1:iend+1] - q_L[i0-1:iend+1]
        q6[i0-1:iend+1] = 6*Q[i0-1:iend+1] - 3*(q_R[i0-1:iend+1] + q_L[i0-1:iend+1])

