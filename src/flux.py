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
def numerical_flux(F, f_R, f_L, Q, q_R, q_L, dq, q6, u_edges, simulation, N):
    # Compute the fluxes (formula 1.12 from Collela and Woodward 1984)
    c = u_edges*(simulation.dt/simulation.dx) #cfl number
    c2 = c*c
    # Flux at left edges
    f_L[1:N+1] = q_R[2:N+2] - c[0:N]*0.5*(dq[2:N+2] - (1.0-(2.0/3.0)*c[0:N])*q6[2:N+2])
    #f_L[1:N+1] =  q_R[2:N+2]  + c[0:N]*0.5*(q6[2:N+2] -dq[2:N+2]) - q6[2:N+2]*c[0:N]**2/3.0
    f_L[0] = f_L[N]
    #f_L[1:N+1] =       (        1.0*c[0:N] -  1.0*c2[0:N])*Q[0:N]
    #f_L[1:N+1] = f_L[1:N+1] + (-1.0 -  5.0*c[0:N] +  6.0*c2[0:N])*Q[1:N+1]
    #f_L[1:N+1] = f_L[1:N+1] + ( 7.0 + 15.0*c[0:N] - 10.0*c2[0:N])*Q[2:N+2]
    #f_L[1:N+1] = f_L[1:N+1] + ( 7.0 - 13.0*c[0:N] +  6.0*c2[0:N])*Q[3:N+3]
    #f_L[1:N+1] = f_L[1:N+1] + (-1.0 +  2.0*c[0:N] -  1.0*c2[0:N])*Q[4:N+4]
    #f_L[1:N+1] = f_L[1:N+1]/12.0

    # Flux at right edges
    c = -c
    f_R[1:N+1] = q_L[3:N+3] + c[1:N+1]*0.5*(dq[3:N+3] + (1.0-2.0/3.0*c[1:N+1])*q6[3:N+3])
    #f_R[1:N+1]= q_L[3:N+3] + c[1:N+1]*0.5*(dq[3:N+3] + q6[3:N+3]) - q6[3:N+3]*c[1:N+1]**2/3.0
    f_R[0] = f_R[N] # Periodic bc

    #fR = (-1.0 + 2.0*c[1:N+1] -c[1:N+1]*c[1:N+1])*Q[1:N+1]
    #fR = fR + (7.0 -13.0*c[1:N+1] +6.0*c[1:N+1]*c[1:N+1])*Q[2:N+2]
    #fR = fR + (7.0 + 15.0*c[1:N+1] - 10.0*c[1:N+1]*c[1:N+1])*Q[3:N+3]
    #fR = fR + (-1.0 - 5.0*c[1:N+1] + 6.0*c[1:N+1]*c[1:N+1])*Q[4:N+4]
    #fR = fR + (1.0*c[1:N+1]-1.0*c[1:N+1]*c[1:N+1])*Q[5:N+5]
    #fR = fR/12.0
    #print(np.amax(abs(fR-f_R[1:N+1])))
    #exit()
    # F - Formula 1.13 from Collela and Woodward 1984)
    F[u_edges >= 0] = f_L[u_edges >= 0]
    F[u_edges <= 0] = f_R[u_edges <= 0]
