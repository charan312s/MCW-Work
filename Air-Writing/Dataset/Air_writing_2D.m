%% Define Data path and Image datastore
clear;close all;
% Load Data
[filepath,~,~] = fileparts(which('Air_writing_2D.m'));
Letter_images = strcat(filepath,'\','2D-Range-Doppler Images');   % Data size 900x23

% image data store
imds = imageDatastore(Letter_images,'IncludeSubfolders',true,'LabelSource','foldernames');
%% Define 6-fold-cross validation, training, validation, and testing data
%%% 6-fold cross-validation "first fold"                          % To choose which fold you are working on from (1 to 6)
[imdsTest,imd]= splitEachLabel(imds,30,150);                      % -1-Test data is the first 30 group 

%%% 6-fold cross-validation for the rest of the folds from (2-5) to choose from
% [imd1,imdsTest,imd2]= splitEachLabel(imds,30,30,120);            % -2-Test data is the second 30 group 
% [imd1,imdsTest,imd2]= splitEachLabel(imds,60,30,90);             % -3-Test data is the third 30 group 
% [imd1,imdsTest,imd2]= splitEachLabel(imds,90,30,60);             % -4-Test data is the fourth 30 group 
% [imd1,imdsTest,imd2]= splitEachLabel(imds,120,30,30);            % -5-Test data is the fifth 30 group

%%% Use the following lines to concatinate the 2 portion of imd1 and imd2 back to imd (for training and validation) for folds from (2-5)
% imd.Files=[imd1.Files;imd2.Files];
% imd.Labels=[imd1.Labels;imd2.Labels];
% imd = imageDatastore(imd.Files,'IncludeSubfolders',true,'LabelSource','foldernames');

%%% 6-fold cross-validation "last fold"
% [imd,imdsTest]= splitEachLabel(imds,150,30);                      % -6-Test data is the sixth 30 group

[imdsTrain,imdsValid] = splitEachLabel(imd,0.8,'randomized');       % Split the training and validation randomly 80/20 %                    

YTrain=imdsTrain.Labels;                  % Training labels
YValid = imdsValid.Labels;                % Validation labels
YTest=imdsTest.Labels;                    % Testing labels
%% CNN Layers               
layers = [imageInputLayer([900 23 1],'Name','input');
  
    convolution2dLayer([5 2],64,'Padding',1,'Name','conv_1')
    batchNormalizationLayer('Name','BN_1')
    reluLayer('Name','relu_1')
    maxPooling2dLayer(2,'Stride',2,'Name','maxp_1')
    
    convolution2dLayer([5 2],128,'Padding',1,'Name','conv_2')
    batchNormalizationLayer('Name','BN_2')
    reluLayer('Name','relu_2')
    maxPooling2dLayer(2,'Stride',2,'Name','maxp_2')
   
    convolution2dLayer([5 2],256,'Padding',1,'Name','conv_3')
    batchNormalizationLayer('Name','BN_3')
    reluLayer('Name','relu_3')
    maxPooling2dLayer(2,'Stride',2,'Name','maxp_3')

    convolution2dLayer([5 2],512,'Padding',1,'Name','conv_4')
    batchNormalizationLayer('Name','BN_4')
    reluLayer('Name','relu_4')
    maxPooling2dLayer(2,'Stride',2,'Name','maxp_4')
    
    fullyConnectedLayer(512,'Name','fc_1')
    reluLayer('Name','relu_5')
    fullyConnectedLayer(128,'Name','fc_2')
    fullyConnectedLayer(10,'Name','fc_3')
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classification')];
% Creat lgraph layer
lgraph = layerGraph;
% Add input and folding layer to graph
lgraph = addLayers(lgraph,layers);
plot(lgraph)
% Check network validation
analyzeNetwork(lgraph)
%% CNN options and Training
options = trainingOptions('adam', ...
    'InitialLearnRate',1e-4, ...
    'SquaredGradientDecayFactor',0.99, ...
    'MaxEpochs',30, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsValid, ...   
    'ValidationFrequency',5, ...    
    'MiniBatchSize',30, ...
    'Plots','training-progress',...
    'Verbose',false);
NeT= trainNetwork(imdsTrain,layers,options);
%% Classification Accracy
YPred = classify(NeT,imdsTest,'MiniBatchSize',30);
accuracy_t = sum(YPred == YTest)/numel(YTest)
plotconfusion(YTest,YPred);