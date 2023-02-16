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
    def __init__(self, N, dt, Tf, ic, vf, tc, recon, dp):
        # Number of cells
        self.N  = N

        # Initial condition
        self.ic = ic

        # Velocity field
        self.vf = vf

        # Test case
        self.tc = tc

        # Time step
        self.dt    = dt
        self.dto2  = dt*0.5
        self.twodt = dt*2.0

        # Total period definition
        self.Tf = Tf

        # Reconstruction method
        self.recon = recon

        # Departure point method
        self.dp = dp

        # Define the interval extremes, advection velocity, etc
        x0 = 0.0
        xf = 1.0

        if ic == 1:
           name = 'Sine wave'
        elif ic == 2:
            name = 'Gaussian wave'
        elif ic == 3:
            name = 'Triangular wave'
        elif ic == 4:
            name = 'Rectangular wave'
        elif ic == 5:
            name = 'Constant field'
        elif ic == 10:
            name = 'Constant field'
        else:
            print("Error in simulation_adv_par_1d - invalid initial condition:", ic)
            exit()

        # IC name
        self.icname = name


        if vf == 1:
           name = 'constant velocity'
        elif vf == 2:
            name = 'variable velocity 1'
        else:
            print("Error in simulation_adv_par_1d- invalid velocity:", vf)
            exit()

        # vf name
        self.vfname = name

        # Reconstruction scheme
        if recon == 1:
            recon_name = 'PPM'
        elif recon == 2:
            recon_name = 'PPM_mono_CW84' #Monotonization from Collela and Woodward 84 paper
        elif recon == 3:
            recon_name = 'PPM_hybrid' #Hybrid PPM from Putman and Lin 07 paper
        elif recon == 4:
            recon_name = 'PPM_mono_L04' #Monotonization from Lin 04 paper

        else:
           print("Error in simulation_adv_par_1d - invalid reconstruction method", recon)
           exit()

        # Departure point scheme
        if dp == 1:
            dp_name = 'Euler'
        elif dp == 2:
            dp_name = 'RK3'
        else:
           print("Error in simulation_adv_par_1d - invalid departure point scheme", dp)
           exit()

        # Interval endpoints
        self.x0 = x0
        self.xf = xf

        # Ghost cells variables
        if recon <= 3:
            self.ngl = 3
            self.ngr = 3
        elif recon == 4:
            self.ngl = 4
            self.ngr = 4
        self.ng  = self.ngl + self.ngr

        # Grid interior indexes
        self.i0   = self.ngl
        self.iend = self.ngl + N

        # Grid
        self.x, self.xc, self.dx = grid_1d(x0, xf, N, self.ngl, self.ngr, self.ng)

        # RK vars
        if dp == 2:
            self.K1 = np.zeros(np.shape(self.x))
            self.K2 = np.zeros(np.shape(self.x))
            self.K3 = np.zeros(np.shape(self.x))

        # Reconstruction method
        self.recon_name = recon_name

        # Departure point method
        self.dp_name = dp_name

        # Simulation title
        if tc == 1:
            self.title = '1D Advection '
        elif tc == 2:
            self.title = '1D advection errors '
        else:
            print("Error in simulation_adv_par_1d- invalid test case")
            exit()


####################################################################################
#  Reconstruction simulation class
####################################################################################
class simulation_recon_par_1d:
    def __init__(self, N, ic, recon):
        # Number of cells
        self.N  = N

        # Initial condition
        self.ic = ic

        # Monotonization
        self.recon = recon

        # Define the interval extremes, advection velocity, etc
        x0 = 0.0
        xf = 1.0
        if ic == 1:
            name = 'Sine wave'
        elif ic == 2:
            name = 'Gaussian wave'
        elif ic == 3:
            name = 'Triangular wave'
        elif ic == 4:
            name = 'Rectangular wave'
        elif ic == 5:
            name = 'Constant field'
        else:
            print("Error - invalid initial condition")
            exit()

        # Flux scheme
        if recon == 1:
            recon_name = 'PPM'
        elif recon == 2:
            recon_name = 'PPM_mono_CW84' #Monotonization from Collela and Woodward 84 paper
        elif recon == 3:
            recon_name = 'PPM_hybrid'    #Quasi-fifth order from Putman and Lin 07 paper
        elif recon == 4:
            recon_name = 'PPM_mono_L04' #Monotonization from Lin 04 paper


        else:
           print("Error - invalid flux method")
           exit()

        # Interval endpoints
        self.x0 = x0
        self.xf = xf

        # Ghost cells variables
        if recon <= 3:
            self.ngl = 3
            self.ngr = 3
        elif recon == 4:
            self.ngl = 4
            self.ngr = 4
        self.ng  = self.ngl + self.ngr

        # Grid interior indexes
        self.i0   = self.ngl
        self.iend = self.ngl + N

        # vf - needed for exact solution
        self.vf = 1

        # Grid
        self.x, self.xc, self.dx = grid_1d(x0, xf, N, self.ngl, self.ngr, self.ng)

        # IC name
        self.icname = name

        # Monotonization method
        self.recon_name = recon_name

        self.title = '1D reconstruction errors '

####################################################################################
#  Parabola class
####################################################################################
class ppm_parabola:
    def __init__(self, simulation):
        # Number of cells
        N  = simulation.N
        ng = simulation.ng

        # parabola coefficients
        # Notation from Colella and  Woodward 1984
        # q(x) = q_L + z*(dq + q6*(1-z)) z in [0,1]
        self.q_L = np.zeros(N+ng)
        self.q_R = np.zeros(N+ng)
        self.dq  = np.zeros(N+ng)
        self.q6 = np.zeros(N+ng)

        # parabola fluxes
        self.f_L = np.zeros(N+ng+1)   # flux from left
        self.f_R = np.zeros(N+ng+1)   # flux from right
        self.f_upw = np.zeros(N+ng+1) # upwind flux

        # reconstruction name
        self.recon_name = simulation.recon_name

        # Extra variables for each schem
        if simulation.recon_name == 'PPM' or simulation.recon_name == 'PPM_mono_CW84' or simulation.recon_name == 'PPM_mono_L04':
            self.Q_edges =  np.zeros(N+ng+1)

        if simulation.recon_name == 'PPM_mono_CW84':
            self.dQ  = np.zeros(N+ng)
            self.dQ0 = np.zeros(N+ng)
            self.dQ1 = np.zeros(N+ng)
            self.dQ2 = np.zeros(N+ng)

        if simulation.recon_name == 'PPM_mono_L04':
            self.dQ      = np.zeros(N+ng)
            self.dQ_min  = np.zeros(N+ng)
            self.dQ_max  = np.zeros(N+ng)
            self.dQ_mono = np.zeros(N+ng)
