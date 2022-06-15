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

from advection_2d            import adv_2d
from parameters_2d           import simulation_par_2d
from error_adv_2d            import error_analysis_adv2d

# Create directories
createDirs()

dim, equation = conf.get_parameters()

# One-dimensional tests
if dim == 1:
    # 1D advection test cases - parameters from par/adv1d.par
    if equation == 'adv':
        # Get parameters
        N, dt, Tf, tc, ic, mono = conf.get_test_parameters_1d('adv_1d.par')
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

    elif equation == 'sw':
        print('Shallow-water is not implemented yet :(')

    else:
        print('Invalid equation!\n')
        exit()

# Two-dimensional tests
elif dim == 2:
    # 2D advection test cases - parameters from par/adv1d.par
    if equation == 'adv':
        # Get parameters
        N, M, dt, Tf, tc, ic, mono = conf.get_test_parameters_2d('adv_2d.par')
        simulation = simulation_par_2d(N, M, dt, Tf, ic, tc, mono)

        # Select test case
        if tc == 1:
            # Advection routine
            adv_2d(simulation, True)
        else:
            # Advection error analysis
            error_analysis_adv2d(simulation)

    elif equation == 'sw':
        print('Shallow-water is not implemented yet :(')
else:
    print('Invalid dimension!\n')
    exit()