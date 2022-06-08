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
# Apply the PPM monotonization
####################################################################################
def monotonization(Q, q_L, q_R, dq, q6, N, mono):
    if mono == 0:
        return 
    elif mono == 1:
        # In each cell, check if Q is a local maximum
        # See First equation in formula 1.10 from Collela and Woodward 1984
        local_maximum = (q_R[2:N+2]-Q[2:N+2])*(Q[2:N+2]-q_L[2:N+2])<=0

        # In this case (local maximum), the interpolation is a constant equal to Q
        q_R[2:N+2][local_maximum] = Q[2:N+2][local_maximum]
        q_L[2:N+2][local_maximum] = Q[2:N+2][local_maximum]

        # Check overshot
        overshoot  = (abs(dq[2:N+2]) < abs(q6[2:N+2]))

        # Move left
        move_left  = (q_R[2:N+2]-q_L[2:N+2])*(Q[2:N+2]-0.5*(q_R[2:N+2]+q_L[2:N+2])) > ((q_R[2:N+2]-q_L[2:N+2])**2)/6.0

        # Move right
        move_right = (-((q_R[2:N+2]-q_L[2:N+2])**2)/6.0 > (q_R[2:N+2]-q_L[2:N+2])*(Q[2:N+2]-0.5*(q_R[2:N+2]+q_L[2:N+2])) )
 
        overshoot_move_left  = np.logical_and(overshoot, move_left)
        overshoot_move_right = np.logical_and(overshoot, move_right)

        q_L[2:N+2][overshoot_move_left]  = 3.0*Q[2:N+2][overshoot_move_left]  - 2.0*q_R[2:N+2][overshoot_move_left]
        q_R[2:N+2][overshoot_move_right] = 3.0*Q[2:N+2][overshoot_move_right] - 2.0*q_L[2:N+2][overshoot_move_right]
       #Auxiliary variables
        #a0 = 1.5*Q[2:N+2] - 0.5*(q_R[2:N+2]+q_L[2:N+2])*0.5 
        #a1 = q_R[2:N+2] - q_L[2:N+2]
        #a2 = 6.0*((q_R[2:N+2]+q_L[2:N+2])*0.5 - Q[2:N+2])

        # Monotonization
        #x_extreme = np.zeros(N)
        #mask_a2not0 = abs(a2)>=10**(-12)
        #x_extreme[mask_a2not0==True] = -a1[mask_a2not0==True]/(2*a2[mask_a2not0==True])
        #x_extreme[mask_a2not0==False] =  float('inf')

        #mask1 = np.logical_and(x_extreme>-0.5, x_extreme<0.0)
        #q_R[2:N+2][mask1==True] = 3.0*Q[2:N+2][mask1==True]-2.0*q_L[2:N+2][mask1==True] 
        #mask2 = np.logical_and(x_extreme>0.0, x_extreme<0.5)  
        #q_L[2:N+2][mask2==True] = 3.0*Q[2:N+2][mask2==True]-2.0*q_R[2:N+2][mask2==True]

        #for i in range(0, N):
        #    if local_maximum[i]==False:
        #        if abs(a2[i])>=10**(-12):
        #            x_extreme = -a1[i]/(2*a2[i])
        #        else:
        #            x_extreme = float('inf')
        #        if (x_extreme>-0.5 and x_extreme<0.0):
        #            q_R[i+2] = 3.0*Q[i+2]-2.0*q_L[i+2] 
        #        elif (x_extreme>0.0 and x_extreme<0.5):
        #            q_L[i+2] = 3.0*Q[i+2]-2.0*q_R[i+2] 

        # Update the polynomial coefs 
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

