================================================

Title: 

Sensor network localization with noisy distance measurements

Abstract:

A sensor network consists of many sensor nodes whose positions are unknown and a small number of anchor nodes with known positions. The nodes in the network can communicate wirelessly to determine approximate distances to nearby nodes. We consider a multiplicative noise model and a related least-squares problem. We can obtain good initial positions by solving a Euclidean distance matrix (EDM) problem using semidefinite programming and aligning the anchor positions. To obtain a low-rank solution, we investigate both minimizing and maximizing the trace of the semidefinite Gram matrix. Different tolerances on the distance approximation error lead to different solutions on the Pareto frontier. We consider an alternative parametrization of the Pareto frontier in which we minimize the distance approximation error subject to a trace constraint. This leads to an inexact Newton method to determine the value of the trace that gives us a solution having the desired distance error. Finally we refine the resulting initial positions by solving a nonlinear least-squares problem.

================================================
