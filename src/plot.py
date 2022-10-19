####################################################################################
#
# Module for plotting routines.
#
# Luan da Fonseca Santos - October 2022
# (luan.santos@usp.br)
####################################################################################

import matplotlib.pyplot as plt

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


