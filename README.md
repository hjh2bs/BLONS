# BLONS

The scripts here demonstrate implementations of algorithms in the following papers

a) Csato and Opper: "Sparse on-line Gaussian processes"
b) He, Koppel, Bedi, Farhood, and Stilwell: "Bi-Level Nonstationary Kernels for Online Gaussian Process Regression"

PREREQUISITES

The following toolboxes are needed to run our code:
1. Mapping Toolbox (for displaying contour plots)
2. Optimization Toolbox

GETTING STARTED

A few functions from the GPML toolbox are used. Please run 
gpml-matlab-master\gpml-matlab-master\startup.m 
first before running the scripts.

cl_data.mat contains the Claytor Lake (CL) north-east-down data.

The following two scripts can be run to demonstrate GP regression on the CL dataset

- SOGP.m is our implementation of sparse online Gaussian process applied to the CL dataset
- BLONS_NN.m is our implementation of BLONS with neural network kernel for the latent GP applied to the CL dataset


NOTES
- For SOGP, the performance is very good when increasing the dictionary size
- In our implementation of BLONS, the noise and function variance for f are optimized separately from the 
log length scale values. The optimization is done on each batch of data to speed up computation.
- BLONS_NN.m can be modified to use different kernels for the latent GP, see gpml documentation for different kernels
- Performance of BLONS_NN varies widely based on how well log length scale values are fit, so multiple runs are recommended
- BLONS appears more stable when less log length scale data is kept, as the data is very noisy (e.g. keeping around 200 LLS values)
- get_lls.m contains the optimization code for discovering log length scale values
- get_sparse_dictionary.m implements Csato and Opper's algorithm for selecting subset of training data to keep
