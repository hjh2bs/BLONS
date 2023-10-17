function partial = rbf_l_partial(locations,measurements,rbfhp,A)

    n1=size(locations,1);
    kov=zeros(n1,n1);
    for ii=1:n1
        for jj=1:n1
            x1=locations(ii,:).';
            x2=locations(jj,:).';
            d12=x1-x2;
            kov(ii,jj)=exp(rbfhp(2))*exp(-(1/2)*d12.'*inv(exp(rbfhp(3)))*d12 )*(1/2)*d12.'*inv(exp(rbfhp(3)))*d12;
        end
    end

    partial=(1/2)*measurements.'*inv(A)*kov*inv(A)*measurements-(1/2)*trace(inv(A)*kov);