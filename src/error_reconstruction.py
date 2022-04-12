####################################################################################
# 
# Module to compute the error convergence in L_inf, L1 and L2 norms
# for the recontruction of a function using the Piecewise Parabolic Method (PPM)
# Luan da Fonseca Santos - April 2022
# 
####################################################################################


import numpy as np
import matplotlib.pyplot as plt
import reconstruction as rec

from parameters import Ψ0, Ψexact, Ψ0_antiderivative, simulation_par, graphdir
from advection  import compute_errors

def error_analysis_recon(simulation):
   # Initial condition
   ic = simulation.ic
   
   # Interval
   x0 = simulation.x0
   xf = simulation.xf
   
   # Number of tests
   Ntest = 11

   # Number of cells
   Ns = np.zeros(Ntest)
   Ns[0] = 10

   # Errors array
   error_linf = np.zeros(Ntest)
   error_l1  = np.zeros(Ntest)
   error_l2  = np.zeros(Ntest)
   
   # Compute number of cells for each simulation
   for i in range(1,Ntest):
      Ns[i] = Ns[i-1]*2.0

   # Aux. variables
   Nplot = 10000
   ξ0 = simulation.x0
   ξf = simulation.xf
   ξplot = np.linspace(ξ0, ξf, Nplot)

   # Let us test and compute the error!
   for i in range(0,Ntest):
      # Update simulation parameters
      simulation = simulation_par(int(Ns[i]), 1.0, 1.0, ic)
      N  = simulation.N
      ξ  = simulation.x
      ξc = simulation.xc
      Δξ = simulation.Δx
     
      Ψ_parabolic = np.zeros(Nplot)
      dists = abs(np.add.outer(ξplot,-ξc))
      neighbours = dists.argmin(axis=1) 

      # Compute average values of Ψ (initial condition)
      Ψ_average = np.zeros(N+5)
      
      if (simulation.ic == 0 or simulation.ic == 1 or simulation.ic == 3 or simulation.ic == 4):
         Ψ_average[0:N] = (Ψ0_antiderivative(ξ[1:N+1], simulation) - Ψ0_antiderivative(ξ[0:N], simulation))/Δξ
      
      elif (simulation.ic == 2):
         Ψ_average[0:N] = Ψ0(ξc,simulation)

      # Periodic boundary conditions
      Ψ_average[N]   = Ψ_average[0]
      Ψ_average[N+1] = Ψ_average[1]
      Ψ_average[N+2] = Ψ_average[3]
      Ψ_average[-2]  = Ψ_average[N-2]
      Ψ_average[-1]  = Ψ_average[N-1]

      # Reconstructs the values of Ψ using a piecewise parabolic polynomial
      δa, a6, aL, aR = rec.ppm_reconstruction(Ψ_average, N)

      # Compute the parabola
      for k in range(0,N):
         x = (ξplot[neighbours==k]-ξ[k])/Δξ # Maps to [0,1]
         Ψ_parabolic[neighbours==k] = aL[k] + δa[k]*x+ x*(1.0-x)*a6[k]

      # Compute exact solution
      Ψ_exact = Ψexact(ξplot, 0, simulation)

      # Relative errors in different metrics
      error_linf[i], error_l1[i], error_l2[i] = compute_errors(Ψ_parabolic, Ψ_exact)

      print('\nParameters: N = '+str(N))
      
      # Output
      if i > 0:
         print('Norms          (Linf,    L1,       L2) ')
         print('Error E_'+str(i)+'    :',"{:.2e}".format(error_linf[i]), "{:.2e}".format(error_l1[i]), "{:.2e}".format(error_l2[i]))
         print('Ratio E_'+str(i)+'/E_'+str(i-1)+':',"{:.2e}".format(error_linf[i-1]/error_linf[i]), "{:.2e}".format(error_l1[i-1]/error_l1[i]), "{:.2e}".format(error_l2[i-1]/error_l2[i]))
      else:
         print('Error (Linf, L1, L2) :',"{:.2e}".format(error_linf[i]), "{:.2e}".format(error_l1[i]), "{:.2e}".format(error_l2[i]))
  
   # Plot the error graph
   plt.loglog(Ns, error_linf, color='green', marker='x', label = '$L_\infty$')
   plt.loglog(Ns, error_l1  , color='blue' , marker='o', label = '$L_1$')
   plt.loglog(Ns, error_l2  , color='red'  , marker='D', label = '$L_2$')
   
   # Reference lines
   Norder = [Ns[Ntest-3],Ns[Ntest-2], Ns[Ntest-1]]
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
   plt.title(simulation.name+' - 1d PPM reconstruction errors')
   plt.savefig(graphdir+'adv1d_ppm_ic'+str(ic)+'_errors.png', format='png')
   plt.close()
   print('\nGraphs have been ploted in '+graphdir)
   print('Convergence graphs has been ploted in '+graphdir+'recon_ppm_ic'+str(ic)+'_errors.png')
