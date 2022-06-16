####################################################################################
#
# Piecewise Parabolic Method (PPM)
# Luan da Fonseca Santos - March 2022
#
####################################################################################
# Source code directory
srcdir = "src/"

import sys
import os.path
sys.path.append(srcdir)

# Imports
import configuration as conf
from miscellaneous           import createDirs

from parameters_1d           import simulation_par_1d
from advection_1d            import adv_1d
from error_adv_1d            import error_analysis_adv1d
from error_reconstruction_1d import error_analysis_recon_1d

# Create directories
createDirs()


# 1D advection test cases - parameters from par/configuration.par
# Get parameters
N, dt, Tf, tc, ic, mono = conf.get_test_parameters_1d('configuration.par')
simulation = simulation_par_1d(N, dt, Tf, ic, tc, mono)

# Select test case
if tc == 1:
    # Advection routine
    adv_1d(simulation, True)
elif tc == 2:
    # Advection error analysis
    error_analysis_adv1d(simulation)
elif tc == 3:
    # Reconstruction error analysis
    error_analysis_recon_1d(simulation)
else:
    print('Invalid test case.')
    exit()
