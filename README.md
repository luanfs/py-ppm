# py-ppm
Python implementation of PPM (piecewise parabolic method)

Solves the linear advection equation using PPM 

Luan F. Santos
(luan.santos@usp.br)

-------------------------------------------------------
Requires:

- numpy
- matplotlib

-------------------------------------------------------

1) Configuration parameters (initial condition, number of grid cells, time step, testcase, etc) must be set in par/configuration.par

2) Choose the simulation to be run in configuration.par (1, 2...)

3) Run using "python3 main.py" in terminal. 

4) Output is written in graphs/.
 
----------------------------------------------------------
References:
 -  Phillip Colella, Paul R Woodward, The Piecewise Parabolic Method (PPM) for gas-dynamical simulations, Journal of Computational Physics, Volume 54, Issue 1, 1984, Pages 174-201, ISSN 0021-9991, https://doi.org/10.1016/0021-9991(84)90143-8.

 -  Carpenter , R. L., Jr., Droegemeier, K. K., Woodward, P. R., & Hane, C. E. (1990).  Application of the Piecewise Parabolic Method (PPM) to Meteorological Modeling, Monthly Weather Review, 118(3), 586-612. Retrieved Mar 31, 2022,  from https://journals.ametsoc.org/view/journals/mwre/118/3/1520-0493_1990_118_0586_aotppm_2_0_co_2.xml

----------------------------------------------------------

This code is freely available under the MIT license.
