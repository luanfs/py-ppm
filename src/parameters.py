####################################################################################
#
# Module for test case set up (grid, initial condition, exact solution and etc)
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
def grid(x0, xf, N):
    x  = np.linspace(x0, xf, N+1) # Cell edges
    xc = (x[0:N] + x[1:N+1])/2    # Cell centers
    dx = (xf-x0)/N                # Grid length
    return x, xc, dx

####################################################################################
#  Simulation class
####################################################################################      
class simulation_par:
    def __init__(self, N, dt, Tf, ic):
        # Number of cells
        self.N  = N

        # Initial condition
        self.ic = ic

        # Time step
        self.dt = dt

        #Total period definition
        self.Tf = Tf

        # Define the interval extremes, advection velocity, etc
        if ic == 1:
            x0 = -1
            xf =  1
            u  =  0.5
            name = 'Sine wave'

        elif ic == 2:
            x0 = 0
            xf = 80
            u  = 0.5
            name = 'Gaussian wave'

        elif ic == 3:
            x0 = 0
            xf = 40
            u  = 0.5
            name = 'Triangular wave'

        elif ic == 4:
            x0 = 0
            xf = 40
            u  = 0.5
            name = 'Rectangular wave'

        elif ic == 5:
            x0 = 0
            xf = 1
            u  = 0.1
            name = 'Test wave'

        else:
            print("Error - invalid test case")
            exit()

        # Interval endpoints
        self.x0 = x0
        self.xf = xf

        # Advection velocity
        self.u = u

        # Grid
        self.x, self.xc, self.dx = grid(x0, xf, N)

        # IC name
        self.name = name

####################################################################################
# Initial condition
####################################################################################
def q0(x, simulation):
    y = qexact(x, 0, simulation)
    return y

####################################################################################
# Initial condition antiderivative
####################################################################################
def q0_antiderivative(x, simulation):
    if simulation.ic == 1:
        y = -np.cos(np.pi*x)/(np.pi) + 1.0*x

    elif simulation.ic == 2:
        x0 = 40
        sigma = 1
        y = np.exp(-((x-x0)/sigma)**2)

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

    #elif simulation.ic == 5:
    #    n = 2
    #    y = x**(n+1)/(n+1.0)
    return y

####################################################################################
# Exact solution to the advection problem 
####################################################################################
def qexact(x, t, simulation):
    u  = simulation.u
    x0 = simulation.x0
    xf = simulation.xf
    ic = simulation.ic  
    X = x-u*t
    mask = (X != xf)
    X[mask] = (X[mask]-x0)%(xf-x0) + x0 # maps back to [x0,xf]

    if simulation.ic == 1:
        y = np.sin(np.pi*X) + 1.0

    elif simulation.ic == 2:
        x0 = 40
        sigma = 1
        y = np.exp(-((X-x0)/sigma)**2)

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
    #elif simulation.ic == 5:
    #    #y = np.ones(np.shape(x))
    #    n = 2
    #    y = x**n
    return y
