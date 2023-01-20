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

####################################################################################
# Compute the flux operator from PPM using the parabola coefficients
# Inputs: Q (average values),  u_edges (velocity at edges)
####################################################################################

def flux_ppm(px, cx, simulation):
    N = simulation.N
    ng = simulation.ng
    i0 = simulation.i0
    iend = simulation.iend

    # Compute the fluxes (formula 1.12 from Collela and Woodward 1984)
    # Flux at left edges
    px.f_L[i0:iend+1] = px.q_R[i0-1:iend] - cx[i0:iend+1]*0.5*(px.dq[i0-1:iend] - (1.0-(2.0/3.0)*cx[i0:iend+1])*px.q6[i0-1:iend])

    # Flux at right edges
    px.f_R[i0:iend+1] = px.q_L[i0:iend+1] - cx[i0:iend+1]*0.5*(px.dq[i0:iend+1] + (1.0+2.0/3.0*cx[i0:iend+1])*px.q6[i0:iend+1])

    # Upwind flux - Formula 1.13 from Collela and Woodward 1984)
    px.f_upw[cx >= 0] = px.f_L[cx >= 0]
    px.f_upw[cx <= 0] = px.f_R[cx <= 0]

"""
####################################################################################
# Compute the flux operator from PPM using its stencil
# Inputs: Q (average values),  u_edges (velocity at edges)
####################################################################################
def flux_ppm_stencil(Q, u_edges, F, a, simulation):
    N = simulation.N

    F[3:N+4] =  a[0,3:N+4]*Q[0:N+1] +\
                a[1,3:N+4]*Q[1:N+2] +\
                a[2,3:N+4]*Q[2:N+3] +\
                a[3,3:N+4]*Q[3:N+4] +\
                a[4,3:N+4]*Q[4:N+5] +\
                a[5,3:N+4]*Q[5:N+6]


####################################################################################
# Compute the flux operator PPM stencil coefficients
# Inputs: c (cfl at egdes), c2 (cfl^2 at edges),  u_edges (velocity at edges)
####################################################################################
def flux_ppm_stencil_coefficients(a, c, c2, u_edges, simulation):
    # Stencil coefficients
    upositive = u_edges>=0
    unegative = ~upositive

    if simulation.flux_method_name == 'PPM':
        a[0, upositive] =  c[upositive] - c2[upositive]
        a[0, unegative] =  0.0

        a[1, upositive] = -1.0 - 5.0*c[upositive] + 6.0*c2[upositive]
        a[1, unegative] = -1.0 + 2.0*c[unegative] - c2[unegative]

        a[2, upositive] =  7.0 + 15.0*c[upositive] - 10.0*c2[upositive]
        a[2, unegative] =  7.0 - 13.0*c[unegative] + 6.0*c2[unegative]

        a[3, upositive] =  7.0 - 13.0*c[upositive] + 6.0*c2[upositive]
        a[3, unegative] =  7.0 + 15.0*c[unegative] - 10.0*c2[unegative]

        a[4, upositive] = -1.0 + 2.0*c[upositive] - c2[upositive]
        a[4, unegative] = -1.0 - 5.0*c[unegative] + 6.0*c2[unegative]

        a[5, upositive] =  0.0
        a[5, unegative] =  c[unegative] - c2[unegative]

        a[:,:] = a[:,:]/12.0

    elif simulation.flux_method_name == 'PPM_hybrid':
        a[0, upositive] =  2.0 - c[upositive] - c2[upositive]
        a[0, unegative] =  0.0

        a[1, upositive] = -13.0 -      c[upositive] + 14.0*c2[upositive]
        a[1, unegative] =  -3.0 +  4.0*c[unegative] -      c2[unegative]

        a[2, upositive] =  47.0 + 39.0*c[upositive] - 26.0*c2[upositive]
        a[2, unegative] =  27.0 - 41.0*c[unegative] + 14.0*c2[unegative]

        a[3, upositive] =  27.0 - 41.0*c[upositive] + 14.0*c2[upositive]
        a[3, unegative] =  47.0 + 39.0*c[unegative] - 26.0*c2[unegative]

        a[4, upositive] =  -3.0 +  4.0*c[upositive] -      c2[upositive]
        a[4, unegative] = -13.0 -      c[unegative] + 14.0*c2[unegative]

        a[5, upositive] =  0.0
        a[5, unegative] =  2.0 - c[unegative] - c2[unegative]

        a[:,:] = a[:,:]/60.0
"""
