%% GMM Training
% Trains three separate Gaussian Mixture Models (16 mixtures each) to
% statistically represent the 39-dimensional feature distribution of the
% 'Lighton', 'Lightoff' and 'Other' categories. Each training .wav file is
% passed through the same pre-processing + MFCC pipeline used at inference
% time, and the resulting feature frames are pooled per category before
% fitting. The trained models are saved to a .mat file for deployment.
%
% Requirements: Signal Processing Toolbox, Audio Toolbox,
%               Statistics and Machine Learning Toolbox.

clc; clear; close all;

% Define paths (edit to match your local dataset location)
trainBase     = fullfile('Data', 'Train');
categories    = {'Lighton', 'Lightoff', 'Other'};
numCategories = length(categories);

% GMM parameters
numMixtures = 16;
options     = statset('MaxIter', 200, 'Display', 'final');

% Preallocate
allFeatures = cell(1, numCategories);

for c = 1:numCategories
    folderPath = fullfile(trainBase, categories{c});
    files = dir(fullfile(folderPath, '*.wav'));
    currFeatures = [];

    for f = 1:length(files)
        [audio, fs] = audioread(fullfile(folderPath, files(f).name));

        % Pre-Emphasis
        audio = filter([1, -0.97], 1, audio);

        % VAD
        winLen = round(0.02 * fs);
        env    = movmean(abs(audio), winLen);
        thresh = 0.1 * max(env);
        vadSignal = audio(env > thresh);

        % Define MFCC window length
        winLenMfcc = round(0.025 * fs);

        % Skip files too short to frame
        if isempty(vadSignal) || length(vadSignal) < winLenMfcc
            continue;
        end

        % Extract MFCC and derivatives
        overlapMfcc = round(0.015 * fs);
        window = hamming(winLenMfcc);
        [coeffs, delta, deltaDelta, ~] = mfcc(vadSignal, fs, ...
            'Window', window, 'OverlapLength', overlapMfcc);

        % Concatenate to 39 dimensions
        features = [coeffs, delta, deltaDelta];

        % Append
        currFeatures = [currFeatures; features]; %#ok<AGROW>
    end

    allFeatures{c} = currFeatures;
end

% Train one GMM per category
gmmOn    = fitgmdist(allFeatures{1}, numMixtures, 'Options', options, 'RegularizationValue', 0.01);
gmmOff   = fitgmdist(allFeatures{2}, numMixtures, 'Options', options, 'RegularizationValue', 0.01);
gmmOther = fitgmdist(allFeatures{3}, numMixtures, 'Options', options, 'RegularizationValue', 0.01);

disp('GMM Training Completed Successfully.');

% Save the models to a .mat file
savePath = fullfile('models', 'trained_gmm_models.mat');
save(savePath, 'gmmOn', 'gmmOff', 'gmmOther');
disp(['Models saved successfully to: ', savePath]);
