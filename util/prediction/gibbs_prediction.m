
%fhp contains 2 scalars
function [pred_mean,pred_cov]=gibbs_prediction(testing,training,values,g_set,fhp,meanfunc,covfunc,likfunc,nnhyp)

    [g_training,~] = gp(nnhyp, @infGaussLik, meanfunc,covfunc,likfunc, g_set(:,1:2), g_set(:,3), training(:,1:2));
    [g_testing,~] = gp(nnhyp, @infGaussLik, meanfunc,covfunc,likfunc, g_set(:,1:2), g_set(:,3), testing(:,1:2));
    k_xx=compute_nsrbf_matrix(training.',training.', fhp, g_training, g_training);
    k_tx=compute_nsrbf_matrix(testing.',training.', fhp,g_testing, g_training);

    pred_mean=k_tx*inv(k_xx+exp(fhp(1))*eye(size(k_xx,1)))*values;

    k_tt=compute_nsrbf_matrix(testing.',testing.', fhp, g_testing,g_testing);
    pred_cov=k_tt-(k_tx*inv(k_xx+exp(fhp(1))*eye(size(k_xx,1)))*k_tx.');












