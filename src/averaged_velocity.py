####################################################################################
#
# Module for computing the time-averaged velocity needed
# for departure point calculation
# Luan Santos 2023
####################################################################################
from advection_ic import velocity_adv_1d

def time_averaged_velocity(u_edges, k, t, simulation):
    # Compute the velocity needed for the departure point
    if simulation.vf>=2:
        if simulation.dp == 2:
            x = simulation.x
            dt = simulation.dt
            dto2 = simulation.dto2
            twodt = simulation.twodt

            simulation.K1 = velocity_adv_1d(x, t, simulation)
            simulation.K2 = velocity_adv_1d(x-dto2*simulation.K1, t-dto2, simulation)
            simulation.K3 = velocity_adv_1d(x-twodt*simulation.K2+dt*simulation.K1, t-dt, simulation)
            u_edges[:] = (simulation.K1 + 4.0*simulation.K2 + simulation.K3)/6.0

            # Fourth-order RK
            #K1 = velocity_adv_1d(x       , t     , simulation)
            #K2 = velocity_adv_1d(x-dto2*K1, t-dto2, simulation)
            #K3 = velocity_adv_1d(x-dto2*K2, t-dto2, simulation)
            #K4 = velocity_adv_1d(x-  dt*K3, t-dt  , simulation)
            #u_edges[:] = (K1 + 2.0*K2 + 2.0*K3 + K4)/6.0

