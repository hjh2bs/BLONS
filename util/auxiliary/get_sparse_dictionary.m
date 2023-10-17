function r = get_sparse_dictionary(current_set, limit,covfunc, hyp)

% Chooses a subset of training data to retain based on projection error.
% See
% Csató, Lehel, and Manfred Opper. "Sparse on-line Gaussian processes." Neural computation 14.3 (2002): 641-668.

if (size(current_set,1)<=limit)
    r=current_set;
    return;
else
    while size(current_set,1)>limit
        % compute alpha, Q
        k_xx=feval(covfunc{:}, hyp.cov,current_set(:,1:2));
        alpha=inv(k_xx+exp(hyp.lik)*eye(size(k_xx)))*current_set(:,3); 
        
        % compute inverse Gram matrix
        Q=inv(k_xx+exp(hyp.lik)*eye(size(k_xx)));
    
        % get score for each point
        scores=zeros(length(k_xx));
        for i=1:length(k_xx)
            scores(i)=abs(alpha(i))/Q(i,i);
        end
        [~,minindex]=min(scores);
        current_set=[current_set(1:minindex-1,:); current_set(minindex+1:end,:)];
    end
    r=current_set;
end






