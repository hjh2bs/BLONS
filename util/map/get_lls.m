

function lls = get_lls(locations, measurements, variances, covfunc, nnhyp, mode)
    n1 = size(locations,1);
    
    % Compute K^f, K^g, K^f+sig_n*I matrices
    if mode ==2
        Kg_xx=feval(covfunc{:}, nnhyp.cov,locations);
        B=Kg_xx+exp(nnhyp.lik)*eye(size(Kg_xx));
    end
    % Random starts
    lp=Inf;
    starts=100;
    lls=zeros(n1,1);
    for ii=1:starts
        proposed_lls=7.8+0.7*rand(n1,1); %(10+0.3*rand)*ones(n1,1);
        Kf_xx=compute_nsrbf_matrix(locations.',locations.',variances,proposed_lls,proposed_lls);
        A=Kf_xx+exp(variances(1))*eye(size(Kf_xx));
        temp_lp = -(-n1*log(2*pi) - (1/2)*(measurements.'*inv(A)*measurements+log(det(A))+log(det(B))+proposed_lls.'*inv(B)*proposed_lls));
        if temp_lp< lp && ~isinf(temp_lp)
            lp=temp_lp;
            lls=proposed_lls;
        end
    end


    % Optimize
    options1 =optimoptions('fminunc','Algorithm','trust-region','SpecifyObjectiveGradient',true,...
                'StepTolerance', 3e-2, 'Display', 'iter-detailed','MaxFunctionEvaluations',50); 
    obj_func1=@(x) log_posterior_and_gradient(locations,measurements,variances,covfunc,nnhyp,x,mode);
    try
        lls=fminunc(obj_func1,lls,options1);
    catch
        warning('Non positive definite covariance?\n');
        lls=lls;
    end

    lls(lls>14)=14;
























