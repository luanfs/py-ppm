####################################################################################
#
# Module for computing the time-averaged velocity needed
# for departure point calculation
# Luan Santos 2023
####################################################################################
from advection_ic import velocity_adv_1d
import numpy as np

def time_averaged_velocity(U_edges, simulation, t):
    # Interior grid indexes
    i0   = simulation.i0
    iend = simulation.iend
    N    = simulation.N
    ng   = simulation.ng

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

            # First departure point estimate
            xd = x-dto2*U_edges.u[:]

            # Velocity data at edges used for interpolation
            u_interp = 1.5*U_edges.u[:] - 0.5*U_edges.u_old[:] # extrapolation for time at n+1/2

            # Linear interpolation
            #U_edges.u_averaged[i0:iend+1] = np.interp(xd[i0:iend+1], x[i0-1:iend+2], u_interp[i0-1:iend+2])
            a = (x[i0:iend+1]-xd[i0:iend+1])/simulation.dx
            upos = U_edges.u[i0:iend+1]>=0
            uneg = ~upos
            U_edges.u_averaged[i0:iend+1][upos] = (1.0-a[upos])*u_interp[i0:iend+1][upos] + a[upos]*u_interp[i0-1:iend][upos]
            U_edges.u_averaged[i0:iend+1][uneg] = -a[uneg]*u_interp[i0+1:iend+2][uneg] + (1.0+a[uneg])*u_interp[i0:iend+1][uneg]

        elif simulation.dp == 3:
            x = simulation.x
            dt = simulation.dt
            dto2 = simulation.dto2
            twodt = simulation.twodt

            simulation.K1 = velocity_adv_1d(x, t, simulation)
            simulation.K2 = velocity_adv_1d(x-dto2*simulation.K1, t-dto2, simulation)
            simulation.K3 = velocity_adv_1d(x-twodt*simulation.K2+dt*simulation.K1, t-dt, simulation)
            U_edges.u_averaged[:] = (simulation.K1 + 4.0*simulation.K2 + simulation.K3)/6.0
