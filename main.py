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

#Imports
import configuration
from miscellaneous        import createDirs
from parameters           import simulation_par
from advection            import adv_1d
from error_adv            import error_analysis_adv1d
from error_reconstruction import error_analysis_recon

# Create directories
createDirs()

# Get parameters
N, dt, Tf, tc, ic, mono = configuration.get_parameters()
simulation = simulation_par(N, dt, Tf, ic, tc, mono)

# Select test case
if tc == 1:
    # Advection routine
    adv_1d(simulation, True)
elif tc == 2:
    # Advection error analysis
    error_analysis_adv1d(simulation)
elif tc == 3:
    # Advection error analysis
    error_analysis_recon(simulation)
else:
    print('Invalid test case.')
    exit()
