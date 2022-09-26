####################################################################################
#
# Module for miscellaneous routines
#
# Luan da Fonseca Santos - April 2022
# (luan.santos@usp.br)
####################################################################################

import os
import numpy as np
from parameters_1d import qexact_adv, Qexact_adv, graphdir
import matplotlib.pyplot as plt
from reconstruction_1d import ppm_reconstruction
from errors import *

####################################################################################
# Create a folder
# Reference: https://gist.github.com/keithweaver/562d3caa8650eefe7f84fa074e9ca949
####################################################################################
def createFolder(dir):
   try:
      if not os.path.exists(dir):
         os.makedirs(dir)
   except OSError:
      print ('Error: Creating directory. '+ dir)

####################################################################################
# Create the needed directories
####################################################################################
def createDirs():
   print("--------------------------------------------------------")
   print("PPM python implementation by Luan Santos - 2022\n")
   # Check directory graphs does not exist
   if not os.path.exists(graphdir):
      print('Creating directory ',graphdir)
      createFolder(graphdir)

   print("--------------------------------------------------------")

####################################################################################
# Diagnostic variables computation
####################################################################################
def diagnostics_adv_1d(Q_average, simulation, total_mass0):
    total_mass =  np.sum(Q_average[0:simulation.N]*simulation.dx)  # Compute new mass
    if abs(total_mass0)>10**(-10):
        mass_change = abs(total_mass0-total_mass)/abs(total_mass0)
    else:
        mass_change = abs(total_mass0-total_mass)
    return total_mass, mass_change

####################################################################################
# Print the diagnostics variables on the screen
####################################################################################
def print_diagnostics_adv_1d(error_linf, error_l1, error_l2, mass_change, t, Nsteps):
    print('\nStep', t, 'from', Nsteps)
    print('Error (Linf, L1, L2) :',"{:.2e}".format(error_linf), "{:.2e}".format(error_l1), "{:.2e}".format(error_l2))
    print('Total mass variation:', "{:.2e}".format(mass_change))

####################################################################################
# Plot the 1d graphs given in the list fields
####################################################################################
def plot_1dfield_graphs(fields, labels, xplot, ymin, ymax, filename, title):
    n = len(fields)
    colors = ('black', 'blue', 'green', 'red', 'purple')
    for k in range(0, n):
        plt.plot(xplot, fields[k], color = colors[k], label = labels[k])

    plt.ylim(-0.1, 1.1*ymax)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title(title)
    plt.savefig(filename)
    plt.close()
    
####################################################################################
# Output and plot
####################################################################################
def output_adv(x, xc, simulation, Q, dq, q6, q_L, error_linf, error_l1, error_l2, plot, k, t, Nsteps, plotstep, total_mass0, CFL):
    N = simulation.N
    dx = simulation.dx
    dt = simulation.dt

    if plot or k==Nsteps:
        # Compute exact averaged solution
        Q_exact = Qexact_adv(x, t, simulation)

        # Relative errors in different metrics
        #error_linf[k], error_l1[k], error_l2[k] = compute_errors(q_parabolic, q_exact)
        error_linf[k], error_l1[k], error_l2[k] = compute_errors(Q_exact, Q[3:N+3])
        if error_linf[k] > 10**(4):
            # CFL number
            CFL = abs(np.amax(abs(u_edges)*dt/dx))
            print('\nStopping due to large errors.')
            print('The CFL number is', CFL)
            exit()

        # Diagnostic computation
        total_mass, mass_change = diagnostics_adv_1d(Q[3:N+3], simulation, total_mass0)

        if plot:
            # Print diagnostics on the screen
            print_diagnostics_adv_1d(error_linf[k], error_l1[k], error_l2[k], mass_change, k, Nsteps)

        if (k-1)%plotstep==0 or k==Nsteps:
            # Plotting variables
            x0 = simulation.x0
            xf = simulation.xf
            Nplot = 10000
            xplot = np.linspace(x0, xf, Nplot)
            q_parabolic = np.zeros(Nplot)
            dists = abs(np.add.outer(xplot,-xc[3:N+3]))
            neighbours = dists.argmin(axis=1)

            if k!=Nsteps:
                # the parabola coeffs are from the previous step
                time = t-dt
            else:
                time = t
                dq, q6, q_L,_ = ppm_reconstruction(Q, simulation) #update parabola coeffs for final step

            # Compute exact solution
            q_exact = qexact_adv(xplot, time, simulation) 

            # Compute the parabola
            for i in range(0, N):
                z = (xplot[neighbours==i]-x[i+3])/dx # Maps to [0,1]
                q_parabolic[neighbours==i] = q_L[i+3] + dq[i+3]*z+ z*(1.0-z)*q6[i+3]

            # Additional plotting variables
            ymin = np.amin(q_exact)
            ymax = np.amax(q_exact)
            icname = simulation.icname
            mono   = simulation.mono  # Monotonization scheme
            tc = simulation.tc
            ic = simulation.ic

            # Plot the solution graph
            qmin = str("{:.2e}".format(np.amin(q_parabolic)))
            qmax = str("{:.2e}".format(np.amax(q_parabolic)))
            title = '1D advection - '+icname+' - time='+str(time)+', CFL='+str(CFL)+',\n N='+str(N)+', '+simulation.fvmethod+', mono = '+simulation.monot+ ', Min = '+ qmin +', Max = '+qmax
            filename = graphdir+'1d_adv_tc'+str(tc)+'_ic'+str(ic)+'_t'+str(k-1)+'_N'+str(N)+'_'+simulation.fvmethod+'_mono'+simulation.monot+'.png'
            plot_1dfield_graphs([q_exact, q_parabolic], ['Exact', 'Parabolic'], xplot, ymin, ymax, filename, title)
