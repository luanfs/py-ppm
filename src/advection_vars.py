####################################################################################
# This module contains the routine that initializates the advection routine variables
# Luan da Fonseca Santos - 2023
####################################################################################

import numpy as np
from parameters_1d       import ppm_parabola, velocity
from advection_ic        import q0_adv, qexact_adv, Qexact_adv, q0_antiderivative_adv, velocity_adv_1d

def adv_vars(simulation):
    N  = simulation.N    # Number of cells
    ic = simulation.ic   # Initial condition
    x  = simulation.x    # Grid
    xc = simulation.xc
    x0 = simulation.x0
    xf = simulation.xf
    dx = simulation.dx   # Grid spacing
    dt = simulation.dt   # Time step
    Tf = simulation.Tf   # Total period definition
    tc = simulation.tc
    recon = simulation.recon # Reconstruction scheme
    dp = simulation.dp # Departure point scheme

    # Ghost cells
    ngl = simulation.ngl
    ngr = simulation.ngr
    ng  = simulation.ng

    # Grid interior indexes
    i0 = simulation.i0
    iend = simulation.iend

    # Velocity at edges
    simulation.U_edges = velocity(simulation)
    simulation.U_edges.u[:] = velocity_adv_1d(x[0:N+ng+1], 0, simulation)
    simulation.U_edges.u_old[:] = simulation.U_edges.u[:]
    simulation.U_edges.u_timecenter[:] = simulation.U_edges.u[:]

    # CFL at edges - x direction
    simulation.cx = simulation.U_edges.u[:]*dt/dx
    simulation.CFL = np.amax(abs(simulation.cx))

    # Compute average values of Q (initial condition)
    # PPM parabola
    simulation.px = ppm_parabola(simulation)

    if (simulation.ic == 0 or simulation.ic == 1 or simulation.ic == 3 or simulation.ic == 4):
        simulation.Q[i0:iend] = (q0_antiderivative_adv(x[i0+1:iend+1], simulation) - q0_antiderivative_adv(x[i0:iend], simulation))/dx
    elif (simulation.ic == 2 or simulation.ic >= 5):
        simulation.Q[i0:iend] = q0_adv(xc[i0:iend],simulation)

    # Periodic boundary conditions
    simulation.Q[iend:N+ng] = simulation.Q[i0:i0+ngr]
    simulation.Q[0:i0]      = simulation.Q[N:N+ngl]

    return
