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
from miscellaneous        import createDirs
from parameters           import simulation_par_1d
from advection            import adv_1d
from error_adv            import error_analysis_adv1d
from error_reconstruction import error_analysis_recon_1d

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
    print('2d tests are not implemented yet :(')

else:
    print('Invalid dimension!\n')
    exit()