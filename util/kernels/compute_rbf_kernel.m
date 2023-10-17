function cov = compute_rbf_kernel(x,xp,rbfhp)

% sigma_f- scalar signal variance
% len_scale- mxm diagonal matrix of length scale parameters
% a - mx1 vector, n is dimension of inputs
% b- mx1 vector, location of second 

cov=rbfhp(1)*exp(-0.5*(x-xp).' *inv(rbfhp(2))*(x-xp));