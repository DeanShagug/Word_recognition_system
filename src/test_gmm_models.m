%% Test & Evaluation
% Processes unseen audio samples through the same pipeline and scores each
% sample against the three trained GMMs using the total log-likelihood. The
% sample is assigned to the model with the maximum score, and overall
% performance is reported as accuracy plus a confusion matrix.
%
% Run train_gmm_models.m first (or load an existing model) so that gmmOn,
% gmmOff and gmmOther are available in the workspace.
%
% Requirements: Signal Processing Toolbox, Audio Toolbox,
%               Statistics and Machine Learning Toolbox.

clc; close all;

% Load trained models if not already in the workspace
if ~exist('gmmOn', 'var')
    load(fullfile('models', 'trained_gmm_models.mat'), 'gmmOn', 'gmmOff', 'gmmOther');
end

% Define test paths (edit to match your local dataset location)
testBase      = fullfile('Data', 'Test');
categories    = {'Lighton', 'Lightoff', 'Other'};
numCategories = length(categories);

% Initialize label arrays
trueLabels = [];
predLabels = [];

for c = 1:numCategories
    folderPath = fullfile(testBase, categories{c});
    files = dir(fullfile(folderPath, '*.wav'));

    for f = 1:length(files)
        [audio, fs] = audioread(fullfile(folderPath, files(f).name));

        % Pre-Emphasis
        audio = filter([1, -0.97], 1, audio);

        % VAD
        winLen = round(0.02 * fs);
        env    = movmean(abs(audio), winLen);
        thresh = 0.1 * max(env);
        vadSignal = audio(env > thresh);

        winLenMfcc = round(0.025 * fs);

        % Skip files too short to frame
        if isempty(vadSignal) || length(vadSignal) < winLenMfcc
            continue;
        end

        % Extract MFCC and derivatives -> 39 features
        overlapMfcc = round(0.015 * fs);
        window = hamming(winLenMfcc);
        [coeffs, delta, deltaDelta, ~] = mfcc(vadSignal, fs, ...
            'Window', window, 'OverlapLength', overlapMfcc);
        features = [coeffs, delta, deltaDelta];

        % Log-likelihood scores (1e-10 avoids log(0))
        scoreOn    = sum(log(pdf(gmmOn,    features) + 1e-10));
        scoreOff   = sum(log(pdf(gmmOff,   features) + 1e-10));
        scoreOther = sum(log(pdf(gmmOther, features) + 1e-10));

        % Predicted class = highest score
        [~, predIdx] = max([scoreOn, scoreOff, scoreOther]);

        trueLabels = [trueLabels; c];        %#ok<AGROW>
        predLabels = [predLabels; predIdx];  %#ok<AGROW>
    end
end

% Overall accuracy
accuracy = sum(trueLabels == predLabels) / length(trueLabels) * 100;
fprintf('Overall System Accuracy (39 Features): %.2f%%\n', accuracy);

% Confusion matrix
trueCat = categorical(trueLabels, 1:3, categories);
predCat = categorical(predLabels, 1:3, categories);

figure('Name', 'System Evaluation (39 Features)', 'NumberTitle', 'off');
set(gcf, 'Units', 'centimeters', 'Position', [2, 2, 16, 12]);

cm = confusionchart(trueCat, predCat);
cm.RowSummary    = 'row-normalized';
cm.ColumnSummary = 'column-normalized';
title(sprintf('Confusion Matrix (39 Features) - Accuracy: %.2f%%', accuracy));
