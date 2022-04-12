####################################################################################
#
# Piecewise Parabolic Method (PPM) advection module
# Luan da Fonseca Santos - April 2022
# Solves the PDE ∂Ψ_t+ c*∂Ψ_x = 0 with periodic boundary conditions
# The initial condition Ψ(ξ,0) is given in the module parameters.py
# 
# References:
# -  Phillip Colella, Paul R Woodward, The Piecewise Parabolic Method (PPM) for gas-dynamical simulations,
# Journal of Computational Physics, Volume 54, Issue 1, 1984, Pages 174-201, ISSN 0021-9991,
# https://doi.org/10.1016/0021-9991(84)90143-8.
#
# -  Carpenter , R. L., Jr., Droegemeier, K. K., Woodward, P. R., & Hane, C. E. (1990). 
# Application of the Piecewise Parabolic Method (PPM) to Meteorological Modeling, Monthly Weather Review, 118(3),
# 586-612. Retrieved Mar 31, 2022, 
# from https://journals.ametsoc.org/view/journals/mwre/118/3/1520-0493_1990_118_0586_aotppm_2_0_co_2.xml
#
####################################################################################

import numpy as np
import matplotlib.pyplot as plt
from parameters import Ψ0, Ψexact, Ψ0_antiderivative, graphdir
import reconstruction as rec

def adv_1d(simulation, plot):
   N  = simulation.N    # Number of cells
   ic = simulation.ic   # Initial condition
   ξ  = simulation.x    # Grid
   ξc = simulation.xc
   ξ0 = simulation.x0
   ξf = simulation.xf
   Δξ = simulation.Δx # Grid spacing
   Δt = simulation.Δt # Time step
   Tf = simulation.Tf # Total period definition
   u  = simulation.u  # Advection velocity
   name = simulation.name

   # CFL number
   CFL = u*Δt/Δξ

   # Number of time steps
   Nsteps = int(Tf/Δt)

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

   # Plotting variables
   Nplot = 10000
   ξplot = np.linspace(ξ0, ξf, Nplot)
   Ψ_parabolic = np.zeros(Nplot)
   Ψ_exact = Ψ0(ξplot, simulation)
   ymin = np.amin(Ψ_exact)
   ymax = np.amax(Ψ_exact)
   dists = abs(np.add.outer(ξplot,-ξc))
   neighbours = dists.argmin(axis=1)
   
   # Fluxes
   f_L = np.zeros(N)
   f_R = np.zeros(N)

   # Aux. variables
   abar = np.zeros(N+1)   
   minus1toN = np.linspace(-1, N-2 ,N,dtype=np.int32)

   # Compute initial mass
   total_mass0 = np.sum(Ψ_average[0:N]*Δξ)
   
   # Errors variable
   error_linf = np.zeros(Nsteps)
   error_l1  = np.zeros(Nsteps)
   error_l2  = np.zeros(Nsteps)

   # Time looping
   for t in range(0, Nsteps):
      # Reconstructs the values of Ψ using a piecewise parabolic polynomial
      δa, a6, aL, aR = rec.ppm_reconstruction(Ψ_average, N)

      # Compute the fluxes (formula 1.11 from Collela and Woodward 1983)
      y = u*Δt/Δξ
      if u>=0:
         f_L = aR[0:N]   - y*0.5*(δa[0:N]   - (1.0-2.0/3.0*y)*a6[0:N])
         abar[0:N] = f_L
      else:
         y = -y
         f_R = aL[1:N+1] + y*0.5*(δa[1:N+1] + (1.0-2.0/3.0*y)*a6[1:N+1])
         abar[0:N] = f_R

      # Periodic boundary conditions
      abar[-1] = abar[N-1]

      # Update the values of Ψ_average (formula 1.12 from Collela and Woodward 1983)
      Ψ_average[0:N] = Ψ_average[0:N] + (u*Δt/Δξ)*(abar[minus1toN]-abar[0:N])
      
      # Periodic boundary conditions
      Ψ_average[N]   = Ψ_average[0]
      Ψ_average[N+1] = Ψ_average[1]
      Ψ_average[N+2] = Ψ_average[3]
      Ψ_average[-2]  = Ψ_average[N-2]
      Ψ_average[-1]  = Ψ_average[N-1]

      # Output and plot
      if plot==True:
         # Compute the parabola
         for i in range(0,N):
            x = (ξplot[neighbours==i]-ξ[i])/Δξ # Maps to [0,1]
            Ψ_parabolic[neighbours==i] = aL[i] + δa[i]*x+ x*(1.0-x)*a6[i]

         # Compute exact solution
         Ψ_exact = Ψexact(ξplot, t*Δt, simulation)

         # Diagnostic computation
         total_mass =  np.sum(Ψ_average[0:N]*Δξ)  # Compute new mass
         if abs(total_mass0)>10**(-10):
            mass_change = abs(total_mass0-total_mass)/abs(total_mass0)
         else:
            mass_change = abs(total_mass0-total_mass)

         # Relative errors in different metrics
         error_linf[t], error_l1[t], error_l2[t] = compute_errors(Ψ_parabolic, Ψ_exact)
         if error_linf[t] > 10**(4):
            print('\nStopping due to large errors.')
            print('The CFL number is', CFL)
            exit()
         
         # Plot the graph
         plt.plot(ξplot, Ψ_exact, color='black',label='Exact')
         plt.plot(ξplot, Ψ_parabolic, color='blue',label='PPM')
         plt.ylim(1.1*ymin, 1.1*ymax)
         plt.ylabel('y')
         plt.xlabel('x')
         plt.legend()
         plt.title(name+' - 1d advection, time='+str(t*Δt)+', CFL='+str(CFL))
         plt.savefig(graphdir+'adv1d_ppm_ic'+str(ic)+'_t'+str(t+1)+'.png', format='png')
         plt.close()
         
         print('\nStep',t+1, 'from', Nsteps)
         print('Error (Linf, L1, L2) :',"{:.2e}".format(error_linf[t]), "{:.2e}".format(error_l1[t]), "{:.2e}".format(error_l2[t]))
         print('Total mass variation:', "{:.2e}".format(mass_change))

   if plot==True:
      # Plot the error graph
      times = np.linspace(0,Tf,Nsteps)
      plt.semilogy(times, error_linf, color='green', label = '$L_\infty$')
      plt.semilogy(times, error_l1  , color='blue', label = '$L_1$')
      plt.semilogy(times, error_l2  , color='red', label = '$L_2$')
      plt.ylabel('Error')
      plt.xlabel('Time (seconds)')
      plt.legend()
      plt.grid(True, which="both")
      plt.title(name+' - 1d advection errors - CFL='+str(CFL))
      plt.savefig(graphdir+'adv1d_ppm_ic'+str(ic)+'_error.png', format='png')
      plt.close()
      print('\nGraphs have been ploted in '+graphdir)
      print('Error evolution is shown in '+graphdir+'adv1d_ppm_ic'+str(ic)+'_error.png')      

   else:
      # Compute the parabola
      for i in range(0,N):
         x = (ξplot[neighbours==i]-ξ[i])/Δξ # Maps to [0,1]
         Ψ_parabolic[neighbours==i] = aL[i] + δa[i]*x+ x*(1.0-x)*a6[i]
      
      # Compute exact solution
      Ψ_exact = Ψexact(ξplot, t*Δt, simulation)
      
      # Relative errors in different metrics
      error_inf, error_1, error_2 = compute_errors(Ψ_parabolic, Ψ_exact)

      return error_inf, error_1, error_2

####################################################################################
# Returns the L_inf, L_1 and L_2 errors
####################################################################################
def compute_errors(Ψ_parabolic, Ψ_exact):
   # Relative errors in different metrics
   E = abs(Ψ_exact-Ψ_parabolic)

   # L_inf error
   error_inf = np.amax(abs(E))/np.amax(abs(Ψ_exact))
      
   # L_1 error
   error_1 = np.sum(E)/len(E)
      
   # L_2 error
   error_2 = np.sqrt(np.sum(E*E))/len(E)

   return error_inf, error_1, error_2
