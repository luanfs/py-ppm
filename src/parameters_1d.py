####################################################################################
#
# Module for 1D test case set up (grid, initial condition, exact solution and etc)
#
# Luan da Fonseca Santos - April 2022
# (luan.santos@usp.br)
####################################################################################
import numpy as np

# Directory parameters
graphdir = "graphs/"            # Graphs directory
pardir   = "par/"               # Parameter files directory

####################################################################################
# Create the 1d grid
####################################################################################
def grid_1d(x0, xf, N, ngl, ngr, ng):
    dx  = (xf-x0)/N                # Grid length
    x   = np.linspace(x0-ngl*dx, xf+ngr*dx, N+1+ng) # Cell edges
    xc  = (x[0:N+ng] + x[1:N+1+ng])/2    # Cell centers
    return x, xc, dx

####################################################################################
#  Advection simulation class
####################################################################################
class simulation_adv_par_1d:
    def __init__(self, N, dt, Tf, ic, tc, flux_method):
        # Number of cells
        self.N  = N

        # Initial condition
        self.ic = ic

        # Test case
        self.tc = tc

        # Time step
        self.dt = dt

        # Total period definition
        self.Tf = Tf

        # Flux method
        self.flux_method = flux_method

        # Define the interval extremes, advection velocity, etc
        if ic == 1:
            x0 = -1.0
            xf =  1.0
            name = 'Sine wave'

        elif ic == 2:
            x0 = -0.5
            xf =  0.5
            name = 'Gaussian wave'

        elif ic == 3:
            x0 = 0
            xf = 40
            name = 'Triangular wave'

        elif ic == 4:
            x0 = 0
            xf = 40
            name = 'Rectangular wave'

        elif ic == 5:
            x0 = -1.0
            xf =  1.0
            name = 'Gaussian wave - variable speed'
        else:
            print("Error - invalid initial condition")
            exit()

        # Flux scheme
        if flux_method == 1:
            flux_method_name = 'PPM'
        elif flux_method == 2:
            flux_method_name = 'PPM_mono_CW84' #Monotonization from Collela and Woodward 84 paper
        elif flux_method == 3:
            flux_method_name = 'PPM_hybrid' #Monotonization from Collela and Woodward 84 paper
        elif flux_method == 4:
            flux_method_name = 'PPM_mono_L04' #Monotonization from Lin 04 paper

        else:
           print("Error - invalid flux method", flux_method)
           exit()

        # Interval endpoints
        self.x0 = x0
        self.xf = xf

        # Ghost cells variables
        if flux_method <= 3:
            self.ngl = 3
            self.ngr = 3
        elif flux_method == 4:
            self.ngl = 4
            self.ngr = 4
        self.ng  = self.ngl + self.ngr

        # Grid interior indexes
        self.i0   = self.ngl
        self.iend = self.ngl + N

        # Grid
        self.x, self.xc, self.dx = grid_1d(x0, xf, N, self.ngl, self.ngr, self.ng)

        # IC name
        self.icname = name

        # Finite volume method
        self.flux_method_name = flux_method_name

        # Simulation title
        if tc == 1:
            self.title = '1D Advection '
        elif tc == 2:
            self.title = '1D advection errors '
        else:
            print("Error - invalid test case")
            exit()


####################################################################################
#  Reconstruction simulation class
####################################################################################
class simulation_recon_par_1d:
    def __init__(self, N, ic, flux_method):
        # Number of cells
        self.N  = N

        # Initial condition
        self.ic = ic

        # Monotonization
        self.flux_method = flux_method

        # Define the interval extremes, advection velocity, etc
        if ic == 1:
            x0 = -1.0
            xf =  1.0
            name = 'Sine wave'

        elif ic == 2:
            x0 = -0.5
            xf =  0.5
            name = 'Gaussian wave'

        elif ic == 3:
            x0 = 0
            xf = 40
            name = 'Triangular wave'

        elif ic == 4:
            x0 = 0
            xf = 40
            name = 'Rectangular wave'

        elif ic == 5:
            x0 = -1.0
            xf =  1.0
            name = 'Gaussian wave - variable speed'
        else:
            print("Error - invalid initial condition")
            exit()

        # Flux scheme
        if flux_method == 1:
            flux_method_name = 'PPM'
        elif flux_method == 2:
            flux_method_name = 'PPM_mono_CW84' #Monotonization from Collela and Woodward 84 paper
        elif flux_method == 3:
            flux_method_name = 'PPM_hybrid'    #Quasi-fifth order from Putman and Lin 07 paper
        elif flux_method == 4:
            flux_method_name = 'PPM_mono_L04' #Monotonization from Lin 04 paper


        else:
           print("Error - invalid flux method")
           exit()

        # Interval endpoints
        self.x0 = x0
        self.xf = xf

        # Ghost cells variables
        self.ngl = 3
        self.ngr = 3
        self.ng  = self.ngl + self.ngr

        # Grid interior indexes
        self.i0   = self.ngl
        self.iend = self.ngl + N

        # Grid
        self.x, self.xc, self.dx = grid_1d(x0, xf, N, self.ngl, self.ngr, self.ng)

        # IC name
        self.icname = name

        # Monotonization method
        self.flux_method_name = flux_method_name

        self.title = '1D reconstruction errors '
