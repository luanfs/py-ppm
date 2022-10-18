####################################################################################
#
# This module contains all the routines that get the needed
# parameters from the /par directory.
#
# Luan da Fonseca Santos - April 2022
# (luan.santos@usp.br)
####################################################################################

import os.path
from parameters_1d import pardir

def get_parameters():
#    # The standard file configuration.par must exist in par/ directory
    file_path = pardir+"configuration.par"

    if os.path.exists(file_path): # The file exists
        # Open the grid file
        confpar = open(file_path, "r")

        # Read the grid file lines
        confpar.readline()
        confpar.readline()
        N = confpar.readline()
        confpar.readline()
        problem = confpar.readline()
        confpar.readline()

        # Convert from str to int
        N = int(N)
        problem = int(problem)

        # Problem name
        if problem == 1:
            problem_name = 'reconstruction'
        elif problem == 2:
            problem_name = 'advection'
        elif problem == 3:
            problem_name = 'shallow water'

        # Close the file
        confpar.close()

        #Print the parameters on the screen
        print("\n--------------------------------------------------------")
        print("Parameters from file", file_path,"\n")
        print("Number of cells: ", N)
        print("Problem: ", problem_name)
        print("--------------------------------------------------------\n")

    else:   # The file does not exist
        print("ERROR in get_grid_parameters: file configuration.par not found in /par.")
        exit()
    return N, problem


def get_adv_parameters_1d(filename):
    # The standard file filename.par must exist in par/ directory
    file_path = pardir+filename

    if os.path.exists(file_path): # The file exists
        # Open the grid file
        confpar = open(file_path, "r")

        # Read the grid file lines
        confpar.readline()
        confpar.readline()
        Tf = confpar.readline()
        confpar.readline()
        dt = confpar.readline()
        confpar.readline()
        ic = confpar.readline()
        confpar.readline()
        tc = confpar.readline()
        confpar.readline()
        mono = confpar.readline()
        confpar.readline()

        # Close the file
        confpar.close()

        # Convert from str to int
        Tf = float(Tf)
        dt = float(dt)
        ic = int(ic)
        tc = int(tc)
        mono = int(mono)

        #Print the parameters on the screen
        print("\n--------------------------------------------------------")
        print("Parameters from file", file_path,"\n")
        print("Test case", tc, "\n")
        print("Initial condition", ic,"\n")
        print("Time step: ", dt)
        print("Mononization scheme: ", mono)
        print("--------------------------------------------------------\n")

    else:   # The file does not exist
        print("ERROR in get_grid_parameters: file "+ filename +" not found in /par.")
        exit()
    return dt, Tf, tc, ic, mono

def get_recon_parameters_1d(filename):
    # The standard file filename.par must exist in par/ directory
    file_path = pardir+filename

    if os.path.exists(file_path): # The file exists
        # Open the grid file
        confpar = open(file_path, "r")

        # Read the grid file lines
        confpar.readline()
        confpar.readline()
        ic = confpar.readline()
        confpar.readline()
        mono = confpar.readline()
        confpar.readline()

        # Close the file
        confpar.close()

        # Convert from str to int
        ic = int(ic)
        mono = int(mono)

        #Print the parameters on the screen
        print("\n--------------------------------------------------------")
        print("Parameters from file", file_path,"\n")
        print("Function: ", ic,"\n")
        print("Mononization scheme: ", mono)
        print("--------------------------------------------------------\n")

    else:   # The file does not exist
        print("ERROR in get_grid_parameters: file "+ filename +" not found in /par.")
        exit()
    return  ic, mono
