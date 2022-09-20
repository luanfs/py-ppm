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
# Create the grid
####################################################################################
def grid_1d(x0, xf, N, ng_left, ng_right):
    ng = ng_left + ng_right         # Total of ghost cells
    dx = (xf-x0)/N                  # Grid length
    x  = np.linspace(x0-ng_left*dx, xf+ng_right*dx, N+ng+1)   # Cell edges
    xc = (x[0:N+ng] + x[1:N+ng+1])/2   # Cell centers
    return x, xc, dx

####################################################################################
#  Simulation class
####################################################################################
class simulation_par_1d:
    def __init__(self, N, dt, Tf, ic, tc, mono):
        # Number of cells
        self.N  = N

        # Number of ghost cells
        self.ng_left  = 3
        self.ng_right = 3
        self.ng = self.ng_left + self.ng_right

        # Interior indexes
        self.i0, self.iend = self.ng_left, self.ng_left + N

        # Initial condition
        self.ic = ic

        # Test case
        self.tc = tc

        # Time step
        self.dt = dt

        # Total period definition
        self.Tf = Tf

        # Monotonization
        self.mono = mono

        # Define the interval extremes, advection velocity, etc
        if ic == 1:
            x0 = -1.0
            xf =  1.0
            name = 'Sine wave'

        elif ic == 2:
            x0 = -1.0
            xf =  1.0
            name = 'Gaussian wave'

        elif ic == 3:
            x0 = 0
            xf = 40
            name = 'Triangular wave'

        elif ic == 4:
            x0 = 0
            xf = 40
            name = 'Rectangular wave'

        else:
            print("Error - invalid test case")
            exit()

        # Monotonization:
        if mono == 0:
            monot = 'none'
        elif mono == 1:
            monot = 'CW84' # Collela and Woodward 84 paper
        else:
           print("Error - invalid monotization method")
           exit()

        # Interval endpoints
        self.x0 = x0
        self.xf = xf

        # Grid
        self.x, self.xc, self.dx = grid_1d(x0, xf, N, self.ng_left, self.ng_right)

        # IC name
        self.icname = name

        # Monotonization method
        self.monot = monot

        # Finite volume method
        self.fvmethod = 'PPM'

        # Simulation title
        if tc == 1:
            self.title = '1D Advection '
        elif tc == 2:
            self.title = '1D advection errors '
        elif tc == 3:
            self.title = '1D reconstruction errors '
        else:
            print("Error - invalid test case")
            exit()

####################################################################################
# Initial condition
####################################################################################
def q0_adv(x, simulation):
    y = qexact_adv(x, 0, simulation)
    return y

####################################################################################
# Initial condition antiderivative
####################################################################################
def q0_antiderivative_adv(x, simulation):
    if simulation.ic == 1:
        y = -np.cos(2.0*np.pi*x)/(2.0*np.pi) + 1.0*x

    elif simulation.ic == 3:
        mask1 = np.logical_and(x>=15.0,x<=20.0)
        mask2 = np.logical_and(x>=20.0,x<=25.0)
        y = x*0
        y[mask1==True] = (x[mask1==True]-15.0)**2/10
        y[mask2==True] = 5.0 - (25.0-x[mask2==True])**2/10.0
        mask3 = x>25.0
        y[mask3==True] = 5

    elif simulation.ic == 4:
        mask = np.logical_and(x>=15.0, x<=25.0)
        y = x*0
        y[mask==True] = x[mask==True] - 15.0
        mask2 = x>25.0
        y[mask2==True] = 10

    return y

####################################################################################
# Exact solution to the advection problem
####################################################################################
def qexact_adv(x, t, simulation):
    x0 = simulation.x0
    xf = simulation.xf
    ic = simulation.ic

    if simulation.ic >= 1 and simulation.ic <= 4 : # constant speed
        u = velocity_adv_1d(x, t, simulation)
        X = x-u*t
        mask = (X != xf)
        X[mask] = (X[mask]-x0)%(xf-x0) + x0 # maps back to [x0,xf]

        if simulation.ic == 1:
            y = np.sin(2.0*np.pi*X) + 1.0

        elif simulation.ic == 2:
            y = np.exp(-10*(np.sin(-np.pi*X))**2)

        elif simulation.ic == 3:
            mask1 = np.logical_and(X>=15.0,X<=20.0)
            mask2 = np.logical_and(X>=20.0,X<=25.0)
            y = x*0
            y[mask1==True] = ( X[mask1==True] - 15.0)/5.0
            y[mask2==True] = (-X[mask2==True] + 25.0)/5.0

        elif simulation.ic == 4:
            mask = np.logical_and(X>=15.0,X<=25.0)
            y = x*0
            y[mask==True] = 1.0
    return y

####################################################################################
# Exact average solution to the advection problem
####################################################################################
def Qexact_adv(x, t, simulation):
    x0 = simulation.x0
    xf = simulation.xf
    ic = simulation.ic
    N  = simulation.N
    ng = simulation.ng       # Interior of grid indexes

    #exit()
    if simulation.ic >= 1 and simulation.ic <= 4 : # constant speed
        u = velocity_adv_1d(x, t, simulation)
        X = x-u*t
        #mask = (X != xf)
        #X[mask] = (X[mask]-x0)%(xf-x0) + x0 # maps back to [x0,xf]
        #X[N] = xf
        if simulation.ic == 1:
            y = (q0_antiderivative_adv(X[1:N+ng+1], simulation)-q0_antiderivative_adv(X[0:N+ng], simulation))/simulation.dx

        elif simulation.ic == 2:
            y = qexact_adv(simulation.xc, t, simulation)

        elif simulation.ic == 3:
            y = qexact_adv(simulation.xc, t, simulation)

        elif simulation.ic == 4:
            y = qexact_adv(simulation.xc, t, simulation)
    return y

####################################################################################
# Velocity field
####################################################################################
def velocity_adv_1d(x, t, simulation):
    if simulation.ic == 1:
        u = 0.1
    elif simulation.ic == 2:
        u = 0.1
    elif simulation.ic == 3:
        u = 0.5
    elif simulation.ic == 4:
        u = 0.5
    return u
