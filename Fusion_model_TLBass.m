close all
clear variables
clc
AUC_=[];optWeights_=[]
for i=1:50
    % 加载模型和输入数据
    load('label.mat'); % 假设标签存储在变量 y

    % 加载模型
    behavioralModel = load('Behavioral_model_1.mat');
    functionalModel = load('Functional_model_1.mat');
    physiologicalModel = load('Pyhsiologial_model_1.mat');

    % 加载输入数据
    behavioralInput = load('Behavioral_input.mat');
    functionalInput = load('Functional_input.mat');
    physiologicalInput = load('Pyhsiologial_input.mat');

    % 进行单个模型的预测
    behavioralScores = predict(behavioralModel.mdl, behavioralInput.X);
    functionalScores = predict(functionalModel.mdl, functionalInput.X);
    physiologicalScores = predict(physiologicalModel.mdl, physiologicalInput.X);

    % 获取各模型的正类概率
    behavioralProbs = behavioralScores(:, 1);
    functionalProbs = functionalScores(:, 1);
    physiologicalProbs = physiologicalScores(:, 1);

    % 定义目标函数
    objFuncPSO = @(w) fusionObjective(w, behavioralProbs, functionalProbs, physiologicalProbs, y);

    % 设置粒子群优化参数
    nvars = 2; % 只需要两个变量
    lb = [0, 0]; % 权重下限
    ub = [1, 1]; % 权重上限

    % 使用粒子群优化
    optWeights = particleswarm(@(w) objFuncPSO(w), nvars, lb, ub);

    % 计算第三个权重
    optWeights(3) = 1 - sum(optWeights); % 确保权重和为1

    % 打印最佳权重
    fprintf('Optimal weights: \n');
    disp(optWeights);

    % 计算融合模型的预测概率
    fusionProbs = optWeights(1) * behavioralProbs + optWeights(2) * functionalProbs + optWeights(3) * physiologicalProbs;

    % 将 fusionProbs 保存到文件
    save('fusionProbs.mat', 'fusionProbs');

    % 计算AUC及其95%置信区间
    [fpr, tpr, ~, AUC] = perfcurve(y, fusionProbs, 1);

    [Bfpr, Btpr, ~, BAUC] = perfcurve(y, behavioralProbs, 1);
    [Ffpr, Ftpr, ~, FAUC] = perfcurve(y, functionalProbs, 1);
    [Pfpr, Ptpr, ~, PAUC] = perfcurve(y, physiologicalProbs, 1);

    se = sqrt(AUC * (1 - AUC) / length(y));
    alpha = 0.05;
    z = norminv(1 - alpha / 2);
    ci_low = AUC - z * se;
    ci_high = AUC + z * se;

    % 打印AUC值及其95%置信区间
    fprintf('Fusion Model AUC: %.3f\n', AUC);
    fprintf('95%% CI: [%.3f, %.3f]\n', ci_low, ci_high);
    AUC_=[AUC_,AUC];
    optWeights_=[optWeights_;optWeights];
end
% 绘制ROC曲线
hf=figure(1);
set(hf,'Position',[100,100,400,400]);
plot(fpr, tpr, 'LineWidth', 2); hold on;
plot(Bfpr, Btpr, 'LineWidth', 2); hold on;
plot(Ffpr, Ftpr, 'LineWidth', 2); hold on;
plot(Pfpr, Ptpr, 'LineWidth', 2); hold on;

xlabel('False Positive Rate', 'FontName', 'Times New Roman', 'FontSize', 18, 'FontWeight', 'bold');
ylabel('True Positive Rate', 'FontName', 'Times New Roman', 'FontSize', 18, 'FontWeight', 'bold');
set(gca, 'YLim', [0, 1], 'FontName', 'Times New Roman', 'FontSize', 18);
% title('TLBass');

lgd =legend(sprintf('TLBass = %.4f', AUC), ...
    sprintf('SBA = %.4f', BAUC), ...
    sprintf('MSA = %.4f', FAUC), ...
    sprintf('MMA = %.4f', PAUC), ...
    'Location', 'SouthEast');
set(lgd, 'FontSize', 16);
grid on; % 增加网格
hold off;
mean(AUC_)
mean(optWeights_)
function obj = fusionObjective(w, behavioralProbs, functionalProbs, physiologicalProbs, y)
% 计算第三个权重
w(3) = 1 - sum(w); % 确保权重和为1

% 计算融合模型的预测概率
fusionProbs = w(1) * behavioralProbs + w(2) * functionalProbs + w(3) * physiologicalProbs;
% 计算融合模型的AUC
[~, ~, ~, AUC] = perfcurve(y, fusionProbs, 1);

% 目标是最大化AUC，因此返回负的AUC作为目标函数值
obj = -AUC;
end