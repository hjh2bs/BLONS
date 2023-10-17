
% This script tests SOGP on the Claytor Lake dataset. The dataset is 
% partitioned into 200 batches, each with 82 measurements. The first 100 batches 
% are used for training, while the rest are used for testing/validation. The
% parameters are
% 
% meanfunc          - mean function
% covfunc           - covariance function
% likfunc           - likelihood function
% SEhyp             - covSEiso hyperparameters, see gpml documentation
% rbft              - how often to train hyperparameters
% dictionary_size   - number of training points to keep
% 
% The Claytor Lake dataset is a 82x600 matrix where every 3 columns
% contains 82 x,y,z measurement data.


clear;
clc;
close all;

addpath('./util/auxiliary')
addpath('./util/map/')
addpath('./util/kernels/')
addpath('./util/objectives/')
addpath('./util/objectives/gradients/')
addpath('./util/objectives/partials/')

%% Load Claytor Lake Data

load('cl_data.mat');
ned_batches=batches;

meanfunc={@meanZero}; 
covfunc = {@covSEiso}; 
likfunc = @likGauss; 
SEhyp = struct('mean', [], 'cov', [0 8], 'lik', -1);
rbft=5;
dictionary_size=250;
batches=100;

dictionary=[];
batch_times=zeros(batches,1);
for ii=1:batches
    timer_start=tic;
    
    current_batch= ned_batches(:, ii*3-2:ii*3);
    current_batch(:,3)= current_batch(:,3)+ abs(0.5*randn(size(current_batch(:,3))));
    dictionary=[dictionary; current_batch];

    dictionary=get_sparse_dictionary(dictionary, dictionary_size, covfunc,SEhyp);
    if mod(ii,rbft)==0
        SEhyp=minimize(SEhyp, @gp, -100, @infGaussLik, meanfunc, covfunc, likfunc, dictionary(:,1:2), dictionary(:,3));
    end
    batch_times(ii)= toc(timer_start);                          % Record batch times
    fprintf("Finished processing batch %d in %d seconds\n", ii, batch_times(ii));
end


%%
testing_set=[];
test_num=2000;
for ii=batches+1:200
    testing_set=[testing_set;ned_batches(:, ii*3-2:ii*3)];
end
testing_set=get_subset(testing_set,test_num);

[mu, ~] = gp(SEhyp, @infGaussLik, meanfunc, covfunc, likfunc,...            % Predict log length scale values
    dictionary(:,1:2), dictionary(:,3), testing_set(:,1:2));
smse=mean((mu-testing_set(:,3)).^2)/var(testing_set(:,3));
fprintf("SMSE is %d\n", smse);

figure(3)
title('Dictionary')
set(gcf,'color','white')
hold on; grid on;
scatter3(dictionary(:,1), dictionary(:,2), dictionary(:,3),'filled','MarkerFaceColor',[.25 .25 .5])

figure(4)
ax1=subplot(1,2,1);
title('Testing Data')
set(gcf,'color','white')
hold on; grid on;
scatter3(testing_set(:,1), testing_set(:,2), testing_set(:,3),'filled','MarkerFaceColor',[.25 .5 .5])

ax2=subplot(1,2,2);
title('Prediction at Testing Locations')
hold on; grid on;
scatter3(testing_set(:,1), testing_set(:,2), mu,'filled','MarkerFaceColor',[.75 .7 .5])

linkaxes([ax1, ax2]);


%% Contour Plots
xv=min(dictionary(:,1)):max(dictionary(:,1));
yv=min(dictionary(:,2)):max(dictionary(:,2));
[X1,Y1]=ndgrid(xv,yv); 
    

f_pred2=mu; f_pred2(f_pred2>22) = 22; f_pred2(f_pred2<0)=0;
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


































