% computes log posterior and gradient, specifically with neural network as
% the kernel 

function [lp, gradient]= log_posterior_and_gradient(locations,measurements,variances,covfunc, nnhyp,lls,mode)
    n1 = size(locations,1);
    
    % Compute K^f, K^g, K^f+sig_n*I matrices
    if mode ==2
        Kg_xx=feval(covfunc{:}, nnhyp.cov,locations);
        B=Kg_xx+exp(nnhyp.lik)*eye(size(Kg_xx));
    end
    Kf_xx=compute_nsrbf_matrix(locations.',locations.',variances,lls,lls);
    A=Kf_xx+exp(variances(1))*eye(size(Kf_xx));
    lp = -1*(-n1*log(2*pi) - (1/2)*(measurements.'*inv(A)*measurements+log(det(A))+log(det(B))+lls.'*inv(B)*lls));

    gradient= zeros(n1,1);
    for ii=1:n1
        gradient(ii)=lp_gradient(locations.',measurements,A,B,lls,variances,ii);
    end
    gradient=-gradient;
