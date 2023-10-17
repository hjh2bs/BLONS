

function partial= rbf_variance_partial(locations,values,rbfhp,A)

    variance_partial = exp(rbfhp(2))*compute_rbf_matrix(locations.',locations.',[0;rbfhp(3)]);
    partial=(1/2)*values.'*inv(A)*variance_partial*inv(A)*values-(1/2)*trace(inv(A)*variance_partial);