


function grad = lp_gradient(locations, measurements, A,B, lls, variance, g_index)
    variance=exp(variance);
    Kf_part= Kf_partial(locations, lls,variance,g_index);
    Kg_g= inv(B)*lls;
    iA=inv(A);
    grad=-(1/2)*trace((-iA*(measurements*measurements.')*iA+iA)*Kf_part)-Kg_g(g_index);