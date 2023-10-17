function kov = compute_rbf_matrix(x1, x2, rbfhp)

% x1 - mxn matrix, n number of samples, m dimension of input
% x2 - mxl matrix, matrix of other locations
% rbfhp - log [variance, lengthscale]

% returns nxl covariance matrix
rbfhp=exp(rbfhp);
n1=size(x1,2);
n2=size(x2,2);
kov=zeros(n1,n2);
for ii=1:n1
    for jj=1:n2
        kov(ii,jj)= compute_rbf_kernel(x1(:,ii),x2(:,jj),rbfhp);
    end
end


