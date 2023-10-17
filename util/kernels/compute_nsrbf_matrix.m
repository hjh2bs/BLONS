
% Assume x1, x2 are 2xm matrices
% lls is nx1 vector
% variances contains sigma_n, sigma_f


function kov = compute_nsrbf_matrix(x1,x2,variances,lls1,lls2)

    n1=size(x1,2);
    n2=size(x2,2);
    variances=exp(variances);
    kov=zeros(n1,n2);
    ls1=exp(lls1);
    ls2=exp(lls2);
    for ii=1:n1
        for jj=1:n2
            kov(ii,jj)=compute_nsrbf_kernel(x1(:,ii), x2(:,jj), variances, [ls1(ii);ls2(jj)]);
        end
    end
    

























