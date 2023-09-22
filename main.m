clear; 
close all;
addpath(genpath('.'));
rmpath(genpath('.git'));
tic

pars = [];
pars.c = 0.01;
pars.nu = 0.5; % nu in (0, 1)
pars.nu_a = 0.2;
pars.nu_b = pars.nu_a;
pars.g4kerA = 0.5;
pars.g4kerB = 8;
pars.g4view = 0.001;
pars.epsilon = 0.01;

dataset_name = 'NUSWIDEOBJ';
dataset_src = sprintf('.\\%s.mat', dataset_name);
load(dataset_src);
X1 = X{1};
X2 = X{2};
tar_class = 4;
data_amount = 200;

avg_times = 10;
auc_list = zeros(1, avg_times);
for times = 1:avg_times
    mydata = split_dataset(X1, X2, Y, tar_class, data_amount);

    Mdl = train_pocsvm_2v(mydata.tar_x1, mydata.tar_x2, mydata.tar_y, 'rbf', pars.nu_a, pars.nu_b, pars.c, pars.g4kerA, pars.g4kerB, pars.g4view, pars.epsilon);
    [auc, auc1, auc2, auc2v, acc, acc1, acc2, acc2v] = predict_pocsvm_2v(Mdl, mydata.test_x1, mydata.test_x2, mydata.test_y);

    auc_list(times) = auc;
end

auc_avg = mean(auc_list(:));
auc_std = std(auc_list);
[n,~] = size(mydata.test_y);
partition = 1 - sum(mydata.test_y == 1) / n;
fprintf('\t\t auc_avg = %.4f \t std = %.4f \t partition =  %.4f\n', auc_avg, auc_std, partition);
fprintf('\t\t %.3f±%.3f\n', auc_avg, auc_std);
auc_avg_std = sprintf('%.3f±%.3f', auc_avg, auc_std);

t = toc