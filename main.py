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
from miscellaneous import createDirs
from parameters import simulation_par
from advection import adv_1d
from error_convergence import error_convergence

# Create directories
createDirs()

# Get parameters
N, dt, Tf, tc, ic = configuration.get_parameters()
simulation = simulation_par(N, dt, Tf, ic)


# Select test case
if tc == 1:
   # Advection routine
   adv_1d(simulation, True)
elif tc == 2:
   # Error analysis
   error_convergence(simulation)
else:
   print('Invalid test case.')
   exit()
