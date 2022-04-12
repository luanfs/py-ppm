####################################################################################
# 
# Module to compute the error convergence in L_inf, L1 and L2 norms
# for the advection equation using the Piecewise Parabolic Method (PPM)
# Luan da Fonseca Santos - April 2022
# 
####################################################################################

from advection import adv_1d
import numpy as np
import matplotlib.pyplot as plt
from parameters import simulation_par, graphdir

def error_convergence(simulation):
   # adv velocity
   u = simulation.u

   # Period
   Tf = (simulation.xf-simulation.x0)/u
   
   # Initial condition
   ic = simulation.ic
   
   # CFL number for all simulations
   CFL = 0.5   

   # Interval
   x0 = simulation.x0
   xf = simulation.xf
   
   # Number of tests
   Ntest = 11

   # Number of cells
   N = np.zeros(Ntest)
   N[0] = 10

   # Array of time steps
   dt = np.zeros(Ntest)
   dt[0] = CFL/(N[0]*u)*(xf-x0)

   # Errors array
   error_linf = np.zeros(Ntest)
   error_l1  = np.zeros(Ntest)
   error_l2  = np.zeros(Ntest)
   
   # Compute number of cells and time step for each simulation
   for i in range(1,Ntest):
      N[i]  = N[i-1]*2.0
      dt[i] = dt[i-1]*0.5

   # Let us test and compute the error!
   for i in range(0,Ntest):
      # Update simulation parameters
      simulation = simulation_par(int(N[i]), dt[i], Tf, ic)

      # Run advection routine and get the errors
      error_linf[i], error_l1[i], error_l2[i] = adv_1d(simulation, False)
      print('\nParameters: N = '+str(int(N[i]))+', dt = '+str(dt[i]))
      
      # Output
      if i > 0:
         print('Norms          (Linf,    L1,       L2) ')
         print('Error E_'+str(i)+'    :',"{:.2e}".format(error_linf[i]), "{:.2e}".format(error_l1[i]), "{:.2e}".format(error_l2[i]))
         print('Ratio E_'+str(i)+'/E_'+str(i-1)+':',"{:.2e}".format(error_linf[i-1]/error_linf[i]), "{:.2e}".format(error_l1[i-1]/error_l1[i]), "{:.2e}".format(error_l2[i-1]/error_l2[i]))
      else:
         print('Error (Linf, L1, L2) :',"{:.2e}".format(error_linf[i]), "{:.2e}".format(error_l1[i]), "{:.2e}".format(error_l2[i]))
   # Plot the error graph
   plt.loglog(N, error_linf, color='green', marker='x', label = '$L_\infty$')
   plt.loglog(N, error_l1 , color='blue',  marker='o', label = '$L_1$')
   plt.loglog(N, error_l2 , color='red',   marker='D', label = '$L_2$')
   
   # Reference lines
   Norder = [N[Ntest-3],N[Ntest-2], N[Ntest-1]]
   ref = 0.001
   order1 = [ref, ref/2.0, ref/4.0]
   order2 = [ref, ref/4.0, ref/16.0]
   order3 = [ref, ref/8.0, ref/64.0]
   plt.loglog(Norder, order1 , ':' , color='black', label = 'Order 1')
   plt.loglog(Norder, order2 , '--', color='black', label = 'Order 2')
   plt.loglog(Norder, order3 , '-.', color='black', label = 'Order 3')

   plt.xlabel('N (number of cells)')
   plt.ylabel('Error')
   plt.legend()
   plt.grid(True, which="both")
   plt.title(simulation.name+' - 1d advection errors - CFL='+str(CFL))
   plt.savefig(graphdir+'adv1d_ppm_ic'+str(ic)+'_errors.png', format='png')
   plt.close()
   print('\nGraphs have been ploted in '+graphdir)
   print('Convergence graphs has been ploted in '+graphdir+'adv1d_ppm_ic'+str(ic)+'_errors.png')
