clear all;clc;
% Import data
load data1.mat
Train = dataNF_train;Test = dataNF_test;
P_train = Train;T_train = labelNF_train;
P_test = Test;T_test = labelNF_test;
% Relieff Feature Selection
[idx,weights] = relieff(P_train,T_train,10,'method','classification');
tic
for i = 5
    sel_fea=idx(1:i);
    Y = T_train;
    X = P_train(:,sel_fea);
    testY = T_test;
    testX = P_test(:,sel_fea);
    params = struct;
    params.n_threads = 1;
    params.search = 2;
    params.gap = 5;
    % INGO Algorithm to Optimize Model Parameters
    % 1、Optimized Parameter Settings
    fun = @getObjValue;                 % Objective Function
    dim = 3;                            % Optimize The Number of Parameters
    lb  = [001, 001, 0.001];            % Lower Bound of Parameters
    ub  = [500, 020, 0.100];            % Upper Bound of Parameters
    % 2、Optimization Algorithm
    [Best_pos,best_so_far] = INGO_ABC(dim,lb,ub,params,fun);
    % 3、Getting Parameters
    num_trees = round(Best_pos(1, 1));
    max_depth = round(Best_pos(1, 2));
    eta = Best_pos(1, 3);
    % Training Model
    model = abcboost_train(Y,X,'abcrobustlogit',num_trees,max_depth,eta,params);
    params.test_auc = 1;
    % Testing Model
    res = abcboost_predict(testX,model,params);
    p_testY=res.prediction;
    figure
    stem(1:length(T_test),T_test,'r*','LineStyle', 'none')
    [tu_test_idx1,~,~] = find(p_testY==testY);
    [tu_test_idx2,~,~] = find(p_testY~=testY);
    hold on
    stem(tu_test_idx1,p_testY(tu_test_idx1),'b^','LineStyle', 'none')
    stem(tu_test_idx2,p_testY(tu_test_idx2),'g^','LineStyle', 'none')
    title('Predictive effectiveness of the INGO_ABCLogitBoost algorithm')
    xlabel('Sample Number')
    ylabel('Label')
    legend('Real Test Set','Prediction Correct.','Prediction Error');
    total = length(testY);
    right = sum(p_testY==testY);
    disp('Print test set classification accuracy');
    str = sprintf('Accuracy = %g%% (%d/%d)',right/total*100,right,total);
    disp(str);
    every_acc=right/total*100;
    all_acc(i)=every_acc;
end
toc
% Drawing the confusion matrix
figure
cm = confusionchart(testY, p_testY);
cm.Title = 'Confusion Matrix for Test Data';
cm.ColumnSummary = 'absolute';    % column-normalized
cm.RowSummary = 'absolute';  %  row-normalized
cm.ColumnSummary = 'column-normalized';    % column-normalized
cm.RowSummary = 'row-normalized';  %  row-normalized
