

% expect locations to be 2xm
function kov_part = Kf_partial(locations,lls,variances,g_index)

    ls= exp(lls);
    n1=size(locations,2);
    kov_part=zeros(n1,n1);
    
    for ii=1:n1
        for jj=1:n1
            if ii==g_index && jj==g_index
                kov_part(ii,jj)=0;
            elseif ii~=g_index && jj~=g_index
                kov_part(ii,jj)=0;
            else
                dij= locations(:,ii)-locations(:,jj);
                if ii==g_index
                    li=ls(ii);
                    lj=ls(jj);
                elseif jj==g_index
                    li=ls(jj);
                    lj=ls(jj);
                end
                kov_part(ii,jj)=variances(2)*sqrt(li*lj)*exp(-dij.'*dij*(2/(li+lj)))*(1-(2*li/(li+lj))+4*(dij.'*dij)*(1/(li+lj))^2*li  );
            end
        end
    end









