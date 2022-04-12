####################################################################################
# 
# Piecewise Parabolic Method (PPM) polynomial reconstruction module
# Luan da Fonseca Santos - March 2022
# Solveswith periodic boundary conditions 
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

####################################################################################
# Given the average values of a scalar field Ψ (Ψ_average), this routine constructs 
# a piecewise parabolic aproximation of Ψ using its average value.
####################################################################################
def ppm_reconstruction(Ψ_average, N):
   # Compute the slopes δΨ0 (centered finite difference)
   # Formula 1.7 from Collela and Woodward 1983 and Figure 2 from Carpenter et al 1989.
   minus1toNm1   = np.linspace(-1, N-2 ,N,dtype=np.int32)
   δΨ0 = np.zeros(N+3)
   δΨ0[0:N] =  0.5*(Ψ_average[1:N+1] - Ψ_average[minus1toNm1])
   δΨ0[N]   =  0.5*(Ψ_average[N+1] - Ψ_average[N-1])
   δΨ0[N+1] =  0.5*(Ψ_average[N+2] - Ψ_average[N])
   δΨ0[-1]  =  0.5*(Ψ_average[0] - Ψ_average[-2])

   # Compute the slopes δΨ (1-sided finite difference)
   # Formula 1.8 from Collela and Woodward 1983 and Figure 2 from Carpenter et al 1989.
   δΨ1 = np.zeros(N+3) # Right 1-sided difference
   δΨ2 = np.zeros(N+3) # Left  1-sided difference

   # Right 1-sided finite difference
   δΨ1[0:N+2] = (Ψ_average[1:N+3]-Ψ_average[0:N+2])
   δΨ1[-1] = (Ψ_average[0]-Ψ_average[-1])
   δΨ1 = 2.0*δΨ1

   # Left  1-sided finite difference
   minus1toNp1 = np.linspace(-1, N ,N+2,dtype=np.int32)
   δΨ2[0:N+2] = (Ψ_average[0:N+2]-Ψ_average[minus1toNp1])
   δΨ2[-1] = (Ψ_average[-2]-Ψ_average[-1])
   δΨ2 = 2.0*δΨ2

   # Final slope - Formula 1.8 from Collela and Woodward 1983
   δΨ = np.minimum(abs(δΨ0),abs(δΨ1))
   δΨ = np.minimum(δΨ, abs(δΨ2))*np.sign(δΨ0)
   
   # Values of Ψ at right edges (Ψ_(j+1/2)) - Formula 1.6 from Collela and Woodward 1983
   Ψ_box = np.zeros(N+2)
   Ψ_box[0:N] = 0.5*(Ψ_average[0:N] + Ψ_average[1:N+1]) - (δΨ[1:N+1] - δΨ[0:N])/6.0
   Ψ_box[N]  = 0.5*(Ψ_average[N] + Ψ_average[N+1]) - (δΨ[N+1] - δΨ[N])/6.0
   Ψ_box[-1] = 0.5*(Ψ_average[-1] + Ψ_average[0]) - (δΨ[0] - δΨ[-1])/6.0

   # Assign values of Ψ_R and Ψ_L
   Ψ_R = np.zeros(N+2)
   Ψ_R[0:N] = Ψ_box[0:N]
   Ψ_L = np.zeros(N+2)
   Ψ_L[0:N] = Ψ_box[minus1toNm1]

   # In each cell, check if Ψ_average is a local maximum
   # See First equation in formula 1.10 from Collela and Woodward 1983
   local_maximum = (Ψ_R[0:N]-Ψ_average[0:N])*(Ψ_average[0:N]-Ψ_L[0:N])<=0
   
   # In this case (local maximum), the interpolation is a constant equal to Ψ_average
   Ψ_R[0:N][local_maximum==True] = Ψ_average[0:N][local_maximum==True]
   Ψ_L[0:N][local_maximum==True] = Ψ_average[0:N][local_maximum==True]

   # Compute the polynomial coefs (we are using the formulation from Carpenter el al 89)
   # a(x)  = <a> + δa*x + a6*(1/12-x*x), x in [-0.5,0.5]
   δa = np.zeros(N+2)
   a6 = np.zeros(N+2)
   δa[0:N] = Ψ_R[0:N] - Ψ_L[0:N]
   a6[0:N] = 6*Ψ_average[0:N] - 3*(Ψ_R[0:N] + Ψ_L[0:N])
   
   # Auxiliary variables
   a0 = 1.5*Ψ_average[0:N] - 0.5*(Ψ_R[0:N]+Ψ_L[0:N])*0.5 
   a1 = Ψ_R[0:N] - Ψ_L[0:N]
   a2 = 6.0*((Ψ_R[0:N]+Ψ_L[0:N])*0.5 - Ψ_average[0:N])
   
   # Monotonization
   x_extreme = np.zeros(N)
   mask_a2not0 = abs(a2)>=10**(-12)
   x_extreme[mask_a2not0==True] = -a1[mask_a2not0==True]/(2*a2[mask_a2not0==True])
   x_extreme[mask_a2not0==False] =  float('inf')
   
   mask1 = np.logical_and(x_extreme>-0.5, x_extreme<0.0)
   Ψ_R[0:N][mask1==True] = 3.0*Ψ_average[0:N][mask1==True]-2.0*Ψ_L[0:N][mask1==True] 
   mask2 = np.logical_and(x_extreme>0.0, x_extreme<0.5)  
   Ψ_L[0:N][mask2==True] = 3.0*Ψ_average[0:N][mask2==True]-2.0*Ψ_R[0:N][mask2==True]
   #exit()
   #for i in range(0,N):
   #   if local_maximum[i]==False:
   #      if abs(a2[i])>=10**(-12):
   #         x_extreme = -a1[i]/(2*a2[i])
   #      else:
   #         x_extreme = float('inf')
   #      if (x_extreme>-0.5 and x_extreme<0.0):
   #         Ψ_R[i] = 3.0*Ψ_average[i]-2.0*Ψ_L[i] 
   #      elif (x_extreme>0.0 and x_extreme<0.5):
   #         Ψ_L[i] = 3.0*Ψ_average[i]-2.0*Ψ_R[i] 

   # Update the polynomial coefs 
   δa[0:N] = Ψ_R[0:N] - Ψ_L[0:N]
   a6[0:N] = 6*Ψ_average[0:N] - 3*(Ψ_R[0:N] + Ψ_L[0:N])
   
   # Periodic boundary conditions
   Ψ_L[N] = Ψ_L[0]
   Ψ_L[-1] = Ψ_L[N-1]

   Ψ_R[N] = Ψ_R[0]
   Ψ_R[-1] = Ψ_R[N-1]

   δa[N] = δa[0]
   δa[-1] = δa[N-1]
   
   a6[N] = a6[0]
   a6[-1] = a6[N-1]
   
   return δa, a6, Ψ_L, Ψ_R
