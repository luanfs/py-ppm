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
from miscellaneous            import createDirs
from parameters_1d            import simulation_adv_par_1d, simulation_recon_par_1d
from advection_1d             import adv_1d
from advection_error          import error_analysis_adv1d
from reconstruction_1d_error  import error_analysis_recon_1d
from von_neumman              import stability_analysis

# Create directories
createDirs()


# 1D advection test cases - parameters from par/configuration.par
# Get parameters
N, problem = conf.get_parameters()

# Select problem to be solved
if problem == 1:
    # Reconstruction error analysis
    ic, recon = conf.get_recon_parameters_1d('reconstruction.par')
    simulation = simulation_recon_par_1d(N, ic, recon)
    error_analysis_recon_1d(simulation)

elif problem == 2:
    # Advection equation
    dt, Tf, tc, ic, vf, recon, dp = conf.get_adv_parameters_1d('advection.par')
    simulation = simulation_adv_par_1d(N, dt, Tf, ic, vf, tc, recon, dp)
    if tc == 1:
        # Advection routine
        adv_1d(simulation, True)
    elif tc == 2:
        # Advection error analysis
        error_analysis_adv1d(simulation)

# Shallow water equations
elif problem == 3:
    print('Not implemented yet!')

# Von Neumman stability analysis
elif problem == 4:
    stability_analysis()

else:
    print('Invalid test case.')
    exit()
