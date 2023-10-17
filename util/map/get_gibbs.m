
% Optimize noise and signal variances for the Gibbs kernel using maximum
% likelihood 


function variances=get_gibbs(locations,measurements)

    n1=size(locations,1);
    lml=Inf;
    starts=100;
    irbfhp=zeros(3,1);
    
    for ii=1:starts
        proposed_rbfhp=[-5+7*rand; -2+20*rand; -4+16*rand];
        K_xx=compute_rbf_matrix(locations.',locations.',proposed_rbfhp(2:end));

        B=K_xx+exp(proposed_rbfhp(1))*eye(size(K_xx));
        p_lml=-(-(1/2)*measurements.'*inv(B)*measurements-(1/2)*log(det(B))-(n1/2)*log(2*pi));
        if p_lml<lml && ~isinf(p_lml)
            lml=p_lml;
            irbfhp=proposed_rbfhp;
        end
    end

    obj_func=@(x) lml_rbf_and_gradient(locations, measurements , x);
    options1 =optimoptions('fminunc','Algorithm','trust-region','SpecifyObjectiveGradient',true,...
                     'StepTolerance', 3e-2,'Display', 'final','MaxFunctionEvaluations',100); 
    try
        variances=fminunc(obj_func,irbfhp,options1);
    catch
        warning('Non positive definite covariance');
        variances=irbfhp;
    end








