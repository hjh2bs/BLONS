

function [lml,gradient]= lml_rbf_and_gradient(locations,measurements,rbfhp)

    n1=size(locations,1);
    Kf_xx=compute_rbf_matrix(locations.',locations.',rbfhp(2:end));
    A=Kf_xx+exp(rbfhp(1))*eye(size(Kf_xx));
    lml=-(-(1/2)*measurements.'*inv(A)*measurements-(1/2)*log(det(A))-(n1/2)*log(2*pi));

    grad1=exp(rbfhp(1))*((1/2)*measurements.'*inv(A)*inv(A)*measurements-(1/2)*trace(inv(A)));
    grad2=rbf_variance_partial(locations,measurements,rbfhp, A);
    grad3=exp(rbfhp(3))*rbf_l_partial(locations,measurements, rbfhp,A);
%     grad3=0;


    gradient=-[grad1;grad2;grad3];
    
