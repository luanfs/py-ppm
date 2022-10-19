####################################################################################
#
# Module for diagnostic computation routines
#
# Luan da Fonseca Santos - October 2022
# (luan.santos@usp.br)
####################################################################################

import numpy as np

####################################################################################
# Diagnostic variables computation
####################################################################################
def diagnostics_adv_1d(Q_average, simulation, total_mass0):
    total_mass =  np.sum(Q_average[0:simulation.N]*simulation.dx)  # Compute new mass
    if abs(total_mass0)>10**(-10):
        mass_change = abs(total_mass0-total_mass)/abs(total_mass0)
    else:
        mass_change = abs(total_mass0-total_mass)
    return total_mass, mass_change


