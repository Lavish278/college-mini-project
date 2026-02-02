%% ============================================================
% AI-Based Solar Irradiance Forecasting
% Final Reproducible MATLAB Script (Paper-Ready)
% ============================================================

clear; clc; close all;

%% ================= REPRODUCIBILITY FIX ======================
rng(42,'twister');   % Fix-1: Reproducibility (MANDATORY)

%% ================= PROJECT STRUCTURE ========================
PROJECT_ROOT = pwd;

DIR.data_raw    = fullfile(PROJECT_ROOT,'data','raw');
DIR.results     = fullfile(PROJECT_ROOT,'results');
DIR.tables      = fullfile(DIR.results,'tables');
DIR.predictions = fullfile(DIR.results,'predictions');
DIR.figures     = fullfile(DIR.results,'figures');
DIR.logs        = fullfile(PROJECT_ROOT,'logs');

dirs = struct2cell(DIR);
for i = 1:numel(dirs)
    if ~exist(dirs{i},'dir'), mkdir(dirs{i}); end
end

diary(fullfile(DIR.logs,'Solar_Irradiance_CommandLog.txt'));
diary on
globalTimer = tic;

fprintf('\n========================================\n');
fprintf('Solar Irradiance Forecasting Started\n');
fprintf('Start Time: %s\n', datestr(now));
fprintf('========================================\n\n');

%% ================= LOAD DATA ================================
fprintf('Loading datasets...\n');

files = dir(fullfile(DIR.data_raw,'Solar_Irradiance_*.csv'));
Nfiles = numel(files);

hLoad = waitbar(0,'Loading datasets...');
allData = table();

for i = 1:Nfiles
    waitbar(i/Nfiles,hLoad,...
        sprintf('Loading %s (%d/%d)',files(i).name,i,Nfiles));

    T = readtable(fullfile(DIR.data_raw,files(i).name), ...
        'VariableNamingRule','preserve');
    allData = [allData; T];
end
close(hLoad);

fprintf('Loaded %d files | %d rows\n\n',Nfiles,height(allData));

%% ================= TIME PROCESSING ==========================
allData.Time = datetime(allData.Year,allData.Month,allData.Day,...
                        allData.Hour,allData.Minute,0);

allData = sortrows(allData,'Time');
[~,idx] = unique(allData.Time,'last');
allData = allData(idx,:);

allData(:,{'Year','Month','Day','Hour','Minute'}) = [];
allData = allData(allData.GHI > 0,:);

%% ================= INPUT / OUTPUT ===========================
Y = normalize(allData.GHI);
X = normalize(allData{:,setdiff(allData.Properties.VariableNames,{'GHI','Time'})});

N = size(X,1);
trainSize = floor(0.8*N);

Xtrain = X(1:trainSize,:);
Ytrain = Y(1:trainSize);
Xtest  = X(trainSize+1:end,:);
Ytest  = Y(trainSize+1:end);

%% ================= FEATURE SELECTION ========================
fprintf('Selecting important features...\n');

% Fix-2: NO DATA LEAKAGE (train only)
corrVals = corr(Xtrain,Ytrain);
importantIdx = abs(corrVals) > 0.5;

Xtrain_top = Xtrain(:,importantIdx);
Xtest_top  = Xtest(:,importantIdx);

fprintf('Selected %d features\n\n',sum(importantIdx));

%% ================= METRIC FUNCTION ==========================
metric = @(y,yp) localMetrics(y,yp);

%% ================= MODEL TRAINING ===========================
models = {'LR','SVR','RF','ANN','LSTM'};
nModels = numel(models);

Predictions = struct();
hModel = waitbar(0,'Training models...');

%% ---- LINEAR REGRESSION ----
waitbar(1/nModels,hModel,'Training Linear Regression');
t = tic;
mdlLR = fitlm(Xtrain,Ytrain);
Predictions.LR = predict(mdlLR,Xtest);
fprintf('LR finished in %.2f sec\n',toc(t));

%% ---- SVR ----
waitbar(2/nModels,hModel,'Training SVR');
t = tic;
mdlSVR = fitrsvm(Xtrain_top,Ytrain,'KernelFunction','rbf','Standardize',true);
Predictions.SVR = predict(mdlSVR,Xtest_top);
fprintf('SVR finished in %.2f sec\n',toc(t));

%% ---- RANDOM FOREST ----
waitbar(3/nModels,hModel,'Training Random Forest');
t = tic;
mdlRF = TreeBagger(150,Xtrain_top,Ytrain,'Method','regression');
Predictions.RF = predict(mdlRF,Xtest_top);
fprintf('RF finished in %.2f sec\n',toc(t));

%% ---- ANN ----
waitbar(4/nModels,hModel,'Training ANN');
t = tic;
netANN = fitnet(20);
netANN.trainParam.epochs = 200;
netANN.trainParam.showWindow = true;
netANN = train(netANN,Xtrain_top',Ytrain');
Predictions.ANN = netANN(Xtest_top')';
fprintf('ANN finished in %.2f sec\n',toc(t));

%% ---- LSTM ----
waitbar(5/nModels,hModel,'Training LSTM');
t = tic;

XseqTrain = num2cell(Xtrain_top',1);
YseqTrain = num2cell(Ytrain',1);
XseqTest  = num2cell(Xtest_top',1);

layers = [
    sequenceInputLayer(size(Xtrain_top,2))
    lstmLayer(50,'OutputMode','sequence')
    fullyConnectedLayer(1)
    regressionLayer];

options = trainingOptions('adam', ...
    'MaxEpochs',150, ...
    'MiniBatchSize',64, ...
    'Verbose',true, ...
    'Plots','training-progress');

netLSTM = trainNetwork(XseqTrain,YseqTrain,layers,options);
Predictions.LSTM = cell2mat(predict(netLSTM,XseqTest))';

fprintf('LSTM finished in %.2f sec\n',toc(t));
close(hModel);

Predictions.Ytest = Ytest(:);

%% ================= METRICS ================================
fprintf('\nComputing metrics...\n');

M(1) = metric(Ytest,Predictions.LR);
M(2) = metric(Ytest,Predictions.SVR);
M(3) = metric(Ytest,Predictions.RF);
M(4) = metric(Ytest,Predictions.ANN);
M(5) = metric(Ytest,Predictions.LSTM);

Results = struct2table(M,'RowNames',models);

% Fix-4: Metadata
Results.Samples   = repmat(numel(Ytest),height(Results),1);
Results.TrainFrac = repmat(0.8,height(Results),1);
Results.TestFrac  = repmat(0.2,height(Results),1);

%% ================= SAVE RESULTS ============================
save(fullfile(DIR.predictions,'Final_Model_Predictions.mat'),'Predictions');
save(fullfile(DIR.tables,'Final_Model_Results_Extended.mat'),'Results');
writetable(Results,fullfile(DIR.tables,'Final_Model_Results_Extended.xlsx'),...
           'WriteRowNames',true);

%% ================= FIGURES (ALL MODELS) ====================
fprintf('Saving figures...\n');

idxPlot = 1:min(1000,length(Ytest));
colors = lines(5);   % colors for models

for i = 1:numel(models)

    figure('Visible','off'); hold on

    % 🔵 ACTUAL (very thick, high-contrast)
    plot(Ytest(idxPlot), ...
        'Color',[0.1 0.9 0.9], ...     % bright cyan (visible on dark bg)
        'LineWidth',4);

    % 🟡 PREDICTION
    plot(Predictions.(models{i})(idxPlot), ...
        'Color',colors(i,:), ...
        'LineWidth',2);

    legend({'Actual','Prediction'}, ...
        'Location','best', ...
        'FontSize',11);

    grid on
    xlabel('Time Steps (15-min)','FontWeight','bold')
    ylabel('Normalized GHI','FontWeight','bold')
    title(['Actual vs ' models{i}], ...
        'FontWeight','bold')

    exportgraphics(gcf, ...
        fullfile(DIR.figures, ...
        ['Actual_vs_' models{i} '.png']), ...
        'Resolution',300);

    close
end

%% ================= END =====================================
totalTime = toc(globalTimer);

fprintf('\n========================================\n');
fprintf('SCRIPT COMPLETED SUCCESSFULLY\n');
fprintf('Total Time: %.2f minutes\n',totalTime/60);
fprintf('Results saved in /results\n');
fprintf('========================================\n');

diary off

%% ================= LOCAL FUNCTION ==========================
function S = localMetrics(y,yp)
    y = y(:); yp = yp(:);
    n = min(length(y),length(yp));
    y = y(1:n); yp = yp(1:n);

    S.RMSE = sqrt(mean((y-yp).^2));
    S.MAE  = mean(abs(y-yp));
    S.MAPE = mean(abs((y-yp)./y))*100;
    S.nRMSE = S.RMSE/mean(y);
    S.MBE  = mean(yp-y);
    S.R2   = 1 - sum((y-yp).^2)/sum((y-mean(y)).^2);
    S.PearsonR = corr(y,yp);
    S.NSE  = S.R2;
end
