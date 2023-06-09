####################################################################################
#
# Module for computing the time-averaged velocity needed
# for departure point calculation
# Luan Santos 2023
####################################################################################
from advection_ic import velocity_adv_1d
import numpy as np
import numexpr as ne

def time_averaged_velocity(U_edges, simulation, t):
    # Interior grid indexes
    i0   = simulation.i0
    iend = simulation.iend
    N    = simulation.N
    ng   = simulation.ng
    u = U_edges.u[i0:iend+1]
    U_edges.upos = ne.evaluate('u>=0')
    U_edges.uneg = ~U_edges.upos

    # Compute the velocity needed for the departure point
    if simulation.vf == 1: # constant velocity
        U_edges.u_averaged[:] = U_edges.u[:]

    if simulation.vf>=2:
        if simulation.dp == 1:
            U_edges.u_averaged[:] = U_edges.u[:]

        elif simulation.dp == 2:
            x = simulation.x
            dt = simulation.dt
            dto2 = simulation.dto2
            twodt = simulation.twodt

            # Velocity data at edges used for interpolation
            u_interp = ne.evaluate('1.5*u - 0.5*u_old', local_dict=vars(U_edges)) # extrapolation for time at n+1/2

            # Linear interpolation
            upos, uneg = U_edges.upos, U_edges.uneg
            a = (U_edges.u[i0:iend+1]*dto2)/simulation.dx
            ap, an = a[upos], a[uneg]
            u1, u2 = u_interp[i0-1:iend][upos], u_interp[i0:iend+1][upos] 
            u3, u4 = u_interp[i0:iend+1][uneg], u_interp[i0+1:iend+2][uneg]
            U_edges.u_averaged[i0:iend+1][upos] = ne.evaluate('(1.0-ap)*u2 + ap*u1')
            U_edges.u_averaged[i0:iend+1][uneg] = ne.evaluate('-an*u4 + (1.0+an)*u3')

        elif simulation.dp == 3:
            x = simulation.x
            dt = simulation.dt
            dto2 = simulation.dto2
            twodt = simulation.twodt

            simulation.K1 = velocity_adv_1d(x, t, simulation)
            simulation.K2 = velocity_adv_1d(x-dto2*simulation.K1, t-dto2, simulation)
            simulation.K3 = velocity_adv_1d(x-twodt*simulation.K2+dt*simulation.K1, t-dt, simulation)
            U_edges.u_averaged[:] = (simulation.K1 + 4.0*simulation.K2 + simulation.K3)/6.0
