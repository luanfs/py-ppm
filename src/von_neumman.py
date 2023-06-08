####################################################################################
#
# Module for Von Neumann stabilty analysis of PPM
#
# Luan da Fonseca Santos - 2023
# (luan.santos@usp.br)
####################################################################################
import numpy as np
from parameters_1d import simulation_adv_par_1d, graphdir, ppm_parabola, velocity
from advection_timestep import time_step_adv1d_ppm
import matplotlib.pyplot as plt
import matplotlib.ticker as tck

def stability_analysis():
    # Flux method
    recons = (1, 3)

    # CFL number for all simulations
    CFL = (1.0,0.8,0.60,0.5, 0.3, 0.1)
    #CFL = (1.0,)

    # Number of tests
    Ntest = 10

    # Number of cells
    N = 100

    # Test case
    ic = 10
    tc = 1
    dp = 1

    # Angles
    dtheta = 2.0*np.pi/N
    theta = np.linspace(dtheta, 2.0*np.pi, N)

    # amplification factor
    rho = np.zeros((len(CFL),N))

    # constant velocity field
    vf = 1

    for recon in recons:
        l = 0
        for cfl in CFL:
            dt = cfl/N

            # Update simulation parameters
            simulation = simulation_adv_par_1d(N, dt, 1.0, ic, vf, tc, recon, dp)

            x = simulation.x
            # Ghost cells
            ng  = simulation.ng
            ngr  = simulation.ngr
            ngl  = simulation.ngl

            # PPM parabola
            px = ppm_parabola(simulation)

            # Grid interior indexes
            i0 = simulation.i0
            iend = simulation.iend

            # PPM variables
            Q = np.zeros(N+ng, dtype = np.complex)
            Qreal = np.zeros(N+ng)
            Qimag = np.zeros(N+ng)
            Q_old = np.zeros(N+ng, dtype = np.complex)
            u_edges = np.ones((N+ng+1))
            cx = np.ones(N+ng+1)

            for k in range(1,N+1):
                # k is the wavenumber
                Q_old[i0:iend] = np.exp(1j*k*theta)

                # Periodic boundary conditions
                Q_old[iend:N+ng] = Q_old[i0:i0+ngr]
                Q_old[0:i0]      = Q_old[N:N+ngl]

                # real and imaginary parts
                Qreal[:] = Q_old.real[:]
                Qimag[:] = Q_old.imag[:]

                # apply ppm operator
                simulation.U_edges = velocity(simulation)
                simulation.U_edges.u[:] = u_edges[:]
                simulation.U_edges.u_old[:] = u_edges[:]
                simulation.U_edges.u_averaged[:] = u_edges[:]
                simulation.px = ppm_parabola(simulation)
                simulation.Q[:] = Qreal[:]
                time_step_adv1d_ppm(0.0, 1, simulation)
                Qreal[:] = simulation.Q[:] 

                simulation.Q[:] = Qimag[:]
                time_step_adv1d_ppm(0.0, 1, simulation)
                Qimag[:] = simulation.Q[:]

                Q = Qreal + 1j*Qimag

                # compute amplification factor
                amplification = abs(Q[i0]/Q_old[i0])
                error = np.amax(abs((Q/Q_old)[i0:iend])-amplification)
                if error > 0.00000000001:
                    print('ERROR on von neumann analysis: Fourier mode ',k,'is not an eigenvector for CFL = ', cfl)
                    exit()
                rho[l, k-1] = amplification

            l = l+1


        # Plot the graph
        for l in range(0, len(CFL)):
            plt.plot(theta[0:N//2], rho[l,0:N//2], label = str(CFL[l]))
        plt.ylim(-0.1, 1.1)
        plt.xlabel(r'$k \Delta \theta$ - dimensionless wavenumber')
        plt.ylabel(r'$\rho(k)$ - Amplification factor')
        plt.title('Scheme: '+simulation.recon_name)
        ax = plt.gca()
        ax.grid(True)
        ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 12))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))
        plt.legend(title="CFL")
        filename = 'stability_'+simulation.recon_name
        plt.savefig(graphdir+filename+'.pdf', format='pdf')
        plt.close()

#--------------------------------------------------------
# src: https://stackoverflow.com/questions/40642061/how-to-set-axis-ticks-in-multiples-of-pi-python-matplotlib
#--------------------------------------------------------
def multiple_formatter(denominator=2, number=np.pi, latex='\pi'):
    def gcd(a, b):
        while b:
            a, b = b, a%b
        return a
    def _multiple_formatter(x, pos):
        den = denominator
        num = np.int(np.rint(den*x/number))
        com = gcd(num,den)
        (num,den) = (int(num/com),int(den/com))
        if den==1:
            if num==0:
                return r'$0$'
            if num==1:
                return r'$%s$'%latex
            elif num==-1:
                return r'$-%s$'%latex
            else:
                return r'$%s%s$'%(num,latex)
        else:
            if num==1:
                return r'$\frac{%s}{%s}$'%(latex,den)
            elif num==-1:
                return r'$\frac{-%s}{%s}$'%(latex,den)
            else:
                return r'$\frac{%s%s}{%s}$'%(num,latex,den)
    return _multiple_formatter

class Multiple:
    def __init__(self, denominator=2, number=np.pi, latex='\pi'):
        self.denominator = denominator
        self.number = number
        self.latex = latex

    def locator(self):
        return plt.MultipleLocator(self.number / self.denominator)

    def formatter(self):
        return plt.FuncFormatter(multiple_formatter(self.denominator, self.number, self.latex))
