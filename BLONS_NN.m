% This script tests BLONS-GP with the log length scale modeled with secondary
% GP using neural network (NN) kernel on the Claytor Lake dataset. The dataset is 
% partitioned into 200 batches, each with 82 measurements. The first 100 batches 
% are used for training, while the rest are used for testing/validation. The
% parameters are
% 
% meanfunc   - mean function
% covfunc    - covariance function
% likfunc    - likelihood function
% nnhyp      - neural network hyperparameters, see gpml documentation
% gibbshyp   - log noise and signal variance
% gibbst     - how often to train fixed hyperparameters for gibbs kernel
% nnt        - how often to train fixed hyperparameters for nn kernel
% lls_pb     - number of lls values to train per batch
% lls_size   - total number of lls values retained

% The Claytor Lake dataset is a 82x600 matrix where every 3 columns
% contains 82 x,y,z measurement data.
% covfunc and nnhyp can be changed to test different kernels for secondary
% GP. However, modifications may be required in above parameters and
% a few util functions to produce good performance

clear;
clc;
close all;

addpath('./util/auxiliary')
addpath('./util/map/')
addpath('./util/kernels/')
addpath('./util/objectives/')
addpath('./util/objectives/gradients/')
addpath('./util/objectives/partials/')
addpath('./util/prediction/')

%% Load Claytor Lake Data

load('cl_data.mat');
ned_batches=batches;

%% BLONS-NN
mode=2;

lls_dictionary=[];

meanfunc={@meanZero}; 
covfunc = {@covNNone}; 
likfunc = @likGauss; 
nnhyp = struct('mean', [], 'cov', [0 0], 'lik', -1);
gibbshyp=[log(.4); log(50)];
gibbst=6;
nnt=10;
lls_pb=6;
lls_size=200;
batches=100;

dictionary=[];
batch_times=zeros(batches,1);
for ii=1:batches
    timer_start=tic;
    current_batch= ned_batches(:, ii*3-2:ii*3);
    current_batch(:,3)= current_batch(:,3)+ abs(0.5*randn(size(current_batch(:,3))));
    dictionary=[dictionary; current_batch];

    if mod(ii,gibbst)==0
        gibbshyp=get_gibbs(current_batch(:,1:2), current_batch(:,3));
    end

    batch_subset=get_subset(current_batch,lls_pb);         % uniformly sample m amount of datapoints for training length scale
    lls = get_lls(batch_subset(:,1:2), ... 
        batch_subset(:,3), gibbshyp(1:2),covfunc, nnhyp,2);                                % get lls values from log posterior
    lls_dictionary=[lls_dictionary; [batch_subset(:,1:2) lls]];

    lls_sparse=get_sparse_dictionary(lls_dictionary, lls_size, covfunc,nnhyp);
    lls_dictionary=lls_sparse;
    if mod(ii,nnt)==0
        nnhyp = minimize(nnhyp, @gp, -100, @infGaussLik, meanfunc, covfunc, likfunc, lls_dictionary(:,1:2), lls_dictionary(:,3)); 
    end
    batch_times(ii)= toc(timer_start);
    fprintf("Finished processing batch %d in %d seconds\n", ii, batch_times(ii));
end

%%

testing_set=[];
test_num=2000;
for ii=batches+1:200
    testing_set=[testing_set;ned_batches(:, ii*3-2:ii*3)];
end
testing_set=get_subset(testing_set,test_num);
dictionary_subset=get_subset(dictionary,2000);

[mu, ~] = gp(nnhyp, @infGaussLik, meanfunc, covfunc, likfunc,...                          % Predict log length scale values
    lls_dictionary(:,1:2), lls_dictionary(:,3), testing_set(:,1:2));
[f_pred, ~] = gibbs_prediction(testing_set(:,1:2),...
    dictionary_subset(:,1:2),dictionary_subset(:,3),...                                                  % Predict f values
    lls_dictionary,gibbshyp(1:2),meanfunc,covfunc,likfunc,nnhyp);
smse=mean((f_pred-testing_set(:,3)).^2)/var(testing_set(:,3));
fprintf("SMSE is %d\n", smse);

figure(2)
ax1=subplot(1,2,1);
title('Dictionary of Log Length Scale Values')
set(gcf,'color','white')
hold on; grid on;
scatter3(lls_dictionary(:,1),lls_dictionary(:,2),lls_dictionary(:,3),'filled')

ax2=subplot(1,2,2);
title('Prediction of Log Length Scale Values at Testing Points')
set(gcf,'color','white')
hold on; grid on;
scatter3(testing_set(:,1),testing_set(:,2),mu,'filled', 'MarkerFaceColor',[.75 .5 .5])
linkaxes([ax1, ax2]);

figure(4)
ax3=subplot(1,2,1);
title('Testing Data')
set(gcf,'color','white')
hold on; grid on;
scatter3(testing_set(:,1), testing_set(:,2), testing_set(:,3),'filled','MarkerFaceColor',[.25 .5 .5])


ax4=subplot(1,2,2);
title('Prediction at Testing Locations')
set(gcf,'color','white')
hold on; grid on;
scatter3(testing_set(:,1), testing_set(:,2), f_pred,'filled','MarkerFaceColor',[.75 .7 .5])
linkaxes([ax3, ax4]);


%% Contour Plots
xv=min(dictionary(:,1)):max(dictionary(:,1));
yv=min(dictionary(:,2)):max(dictionary(:,2));
[X1,Y1]=ndgrid(xv,yv); 
    

f_pred2=f_pred; f_pred2(f_pred2>22) = 22; f_pred2(f_pred2<0)=0;
Z3=griddata(testing_set(:,1),testing_set(:,2),-f_pred2,X1,Y1);
fig=figure(8);
set(gcf,'Color','white')
hold on;
contourf(X1,Y1,Z3);
% colormap(flipud(parula))
xlabel('X (m)', 'FontSize', 16, Interpreter='latex')
ylabel('Y (m)', 'FontSize', 16, Interpreter='latex')
cb=contourcbar(FontSize=16);
cb.XLabel.Interpreter='latex';
cb.XLabel.String = 'Predictive mean of $$f(\mathbf{x})$$';
set(fig,'Renderer','painters');
set(gca,'clim',[-22 0])













