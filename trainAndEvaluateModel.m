function [AUC,ci_low,ci_high,mdl] = trainAndEvaluateModel(X, y)

numSplits = 50; % 进行 50 次不同的数据分割
K = 10; % 十折交叉验证
bestModel = [];
bestAccuracy = -Inf;

for split = 1:numSplits
    % 随机划分数据为训练集和测试集（80% 训练集，20% 测试集）
    cv = cvpartition(y, 'HoldOut', 0.2, 'Stratify', true); % 分层划分，保持类别比例
    
    % 提取训练集和测试集
    X_train = X(training(cv), :);
    y_train = y(training(cv));
    X_test = X(test(cv), :);
    y_test = y(test(cv));
    
    % 使用十折交叉验证调整逻辑回归模型的超参数
    cvFolds = crossvalind('Kfold', y_train, K);
    
    for k = 1:K
        % 训练集和验证集划分
        valIdx = (cvFolds == k);
        trainIdx_fold = ~valIdx;
        
        X_train_fold = X_train(trainIdx_fold, :);
        y_train_fold = y_train(trainIdx_fold);
        X_val_fold = X_train(valIdx, :);
        y_val_fold = y_train(valIdx);
        
        % 训练逻辑回归模型
        model = fitglm(X_train_fold, y_train_fold, 'Distribution', 'binomial', 'Link', 'logit');
        
        % 在验证集上评估模型
        predictions = predict(model, X_val_fold) > 0.5;
        accuracy = sum(predictions == y_val_fold) / length(y_val_fold);
        
        % 如果当前模型在验证集上表现更好，则更新最佳模型
        if accuracy > bestAccuracy
            mdl = model;
            bestAccuracy = accuracy;
        end
    end
end

% 进行预测
y_pred_prob = predict(mdl, X_test);
[fpr, tpr, ~, AUC, opt] = perfcurve(y_test, y_pred_prob, 1);
% 计算AUC的标准误差
se = sqrt(AUC * (1 - AUC) / length(y_test));

% 计算95%置信区间
alpha = 0.05;
z = norminv(1 - alpha/2);
ci_low = AUC - z * se;
ci_high = AUC + z * se;

% 打印AUC值及其95%置信区间
fprintf('AUC: %.3f\n', AUC);
fprintf('95%% CI: [%.3f, %.3f]\n', ci_low, ci_high);

% 绘制ROC曲线
plot(fpr, tpr, 'LineWidth', 2); hold on
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title('ROC Curve');
end