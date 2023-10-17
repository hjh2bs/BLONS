
% Assume all locations are 2 dimensional, this calculation is incorrect for
% dimensions not equal to 2

function kernel = compute_nsrbf_kernel(x, xp, variances, ls)

    d=x-xp;
    sls=sum(ls);
    kernel= variances(2)*sqrt(ls(1)*ls(2))*(2/sls)*exp(-d.'*d*(2/sls));
