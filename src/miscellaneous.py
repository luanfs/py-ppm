####################################################################################
# 
# Module for miscellaneous routines
#
# Luan da Fonseca Santos - April 2022
# (luan.santos@usp.br)
####################################################################################

import os
import numpy as np
from parameters_1d import graphdir
import matplotlib.pyplot as plt

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
# Diagnostic variables computation
#################################################################################### 
def diagnostics_adv_2d(Q_average, simulation, total_mass0):
    total_mass =  np.sum(Q_average[2:simulation.N+2, 2:simulation.M+2])*simulation.dx*simulation.dy  # Compute new mass
    if abs(total_mass0)>10**(-10):
        mass_change = abs(total_mass0-total_mass)/abs(total_mass0)
    else:
        mass_change = abs(total_mass0-total_mass)
    return total_mass, mass_change

#################################################################################### 
# Print the diagnostics variables on the screen
#################################################################################### 
def print_diagnostics_adv_2d(error_linf, error_l1, error_l2, mass_change, t, Nsteps):
    print('\nStep', t, 'from', Nsteps)
    print('Error (Linf, L1, L2) :',"{:.2e}".format(error_linf), "{:.2e}".format(error_l1), "{:.2e}".format(error_l2))
    print('Total mass variation:', "{:.2e}".format(mass_change))

####################################################################################
# Plot the 2d graphs given in the list fields 
####################################################################################
def plot_2dfield_graphs(fields, xplot, yplot, filename, title):
    n = len(fields)
    for k in range(0, n):
        plt.contourf(xplot, yplot, fields[k], cmap='jet', levels=100)
        plt.colorbar(orientation='vertical', fraction=0.046, pad=0.04)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(title)
        plt.savefig(filename)
        plt.close()