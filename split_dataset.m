function [mydata] = split_dataset(X1, X2, Y, tar_class, amount)
% To split dataset. Samples partition of taring vs testing is 4 : 1

    % normalize
    X1 = mapminmax(X1',0,1);
    X1 = X1';
    X2 = mapminmax(X2',0,1);
    X2 = X2';

    % training data
    [tar_rol, ~] = find(Y == tar_class);
    tar_n = ceil(numel(tar_rol) * 0.8); % Take 80% for training
    tar_idx = (randperm(numel(tar_rol), tar_n))'; 
    tar_idx = tar_rol(tar_idx);
    tar_y = Y(tar_idx);
    tar_x1 = X1(tar_idx, :);
    tar_x2 = X2(tar_idx, :);

    if amount > 0 && amount < tar_n
        tar_y = tar_y(1:amount);
        tar_x1 = tar_x1(1:amount, :);
        tar_x2 = tar_x2(1:amount, :);
        tar_n = amount;
    end

    % texting data
    rest_idx = setdiff(tar_rol, tar_idx);  % the rest of target class sample idx
    tmp_len = ceil(ceil(tar_n/4)*0.5); % take a half for test, and random draw the rest
%     tmp_len = ceil(numel(rest_idx)*0.5);  
    tmp_idx = rest_idx(1:tmp_len);
    [test_rol, ~] = find(Y ~= tar_class);
    test_rol = [setdiff(rest_idx, tmp_idx); test_rol]; % the random list
    test_idx = (randperm(numel(test_rol), (ceil(tar_n/4) - tmp_len)))';
    test_idx = test_rol(test_idx);
    test_idx = [tmp_idx; test_idx];
%     test_idx = test_rol(test_idx);
    test_y = Y(test_idx);
    test_x1 = X1(test_idx, :);
    test_x2 = X2(test_idx, :);

    % make y into +1 / -1
    if tar_class == -1
        test_y = -test_y;
    else
        test_y(test_y ~= tar_class) = -1;
        test_y(test_y == tar_class) = 1;
    end
    tar_y(:) = 1;

    un_exam = [-1 1];
    un = unique(test_y)';
    if ~isequal(un_exam,un)
        error('test_y contains values other than +1 -1')
    end

    mydata.tar_n = tar_n;
    mydata.test_n = numel(test_y);
    mydata.tar_y = tar_y; mydata.tar_x1 = tar_x1; mydata.tar_x2 = tar_x2;
    mydata.test_y = test_y; mydata.test_x1 = test_x1; mydata.test_x2 = test_x2;
end

