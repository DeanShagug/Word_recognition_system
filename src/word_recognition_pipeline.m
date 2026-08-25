%% Word Recognition System - Full Signal Processing & Analysis Pipeline
%
% This script demonstrates every stage of the MFCC + GMM word-recognition
% pipeline on a single example pair of recordings (one "Light On" and one
% "Light Off"), producing the analysis figures used in the project report:
%
%   1. Configuration & audio loading
%   2. Pre-Emphasis
%   3. Voice Activity Detection (VAD)
%   4. Framing & Hamming windowing
%   5. Feature extraction (MFCC + Delta + Delta-Delta, 39 features/frame)
%
% Model training and evaluation are implemented in separate scripts:
%   train_gmm_models.m   - trains the three GMMs and saves the .mat model
%   test_gmm_models.m    - evaluates accuracy and plots the confusion matrix
%
% Requirements: MATLAB, Signal Processing Toolbox, Audio Toolbox,
%               Statistics and Machine Learning Toolbox.

%% Configuration
clc; clear; close all;

% Define paths (edit to match your local dataset location)
basePath = fullfile('Data', 'Train');
pathOn   = fullfile(basePath, 'LightOn');
pathOff  = fullfile(basePath, 'LightOff');

% Get first wav file from each folder
filesOn  = dir(fullfile(pathOn,  '*.wav'));
filesOff = dir(fullfile(pathOff, '*.wav'));

% Read audio files
[audioOn,  fsOn]  = audioread(fullfile(pathOn,  filesOn(1).name));
[audioOff, fsOff] = audioread(fullfile(pathOff, filesOff(1).name));

%% Pre-Emphasis
% A first-order high-pass filter (alpha = 0.97) amplifies high frequencies,
% compensating for their natural attenuation in human speech and improving
% the quality of the extracted features.

% Time vectors
tOn  = (0:length(audioOn)-1)  / fsOn;
tOff = (0:length(audioOff)-1) / fsOff;

% Freq vectors
N_on = length(audioOn);
fOn  = (0:N_on-1) * (fsOn / N_on);
half_N_on = floor(N_on/2);

N_off = length(audioOff);
fOff  = (0:N_off-1) * (fsOff / N_off);
half_N_off = floor(N_off/2);

% Compute original FFT
fftOn_orig  = abs(fft(audioOn));
fftOff_orig = abs(fft(audioOff));

% Apply pre-emphasis
alpha = 0.97;
preEmphOn  = filter([1, -alpha], 1, audioOn);
preEmphOff = filter([1, -alpha], 1, audioOff);

% Compute pre-emphasized FFT
fftOn_pre  = abs(fft(preEmphOn));
fftOff_pre = abs(fft(preEmphOff));

% Plot results
figure('Name', 'Pre-Emphasis Analysis', 'NumberTitle', 'off');
set(gcf, 'Units', 'centimeters', 'Position', [2, 2, 16, 12]);

subplot(3,2,1); plot(tOn, audioOn);
title('Light On Time Domain');  xlabel('Time (s)'); ylabel('Amp');
subplot(3,2,2); plot(tOff, audioOff);
title('Light Off Time Domain'); xlabel('Time (s)'); ylabel('Amp');

subplot(3,2,3); plot(fOn(1:half_N_on),  fftOn_orig(1:half_N_on));
title('Light On Freq Domain');  xlabel('Freq (Hz)'); ylabel('Mag');
subplot(3,2,4); plot(fOff(1:half_N_off), fftOff_orig(1:half_N_off));
title('Light Off Freq Domain'); xlabel('Freq (Hz)'); ylabel('Mag');

subplot(3,2,5); plot(fOn(1:half_N_on),  fftOn_pre(1:half_N_on));
title('Light On Frequency after Pre-Emphasis');  xlabel('Freq (Hz)'); ylabel('Mag');
subplot(3,2,6); plot(fOff(1:half_N_off), fftOff_pre(1:half_N_off));
title('Light Off Frequency after Pre-Emphasis'); xlabel('Freq (Hz)'); ylabel('Mag');

%% Voice Activity Detection (VAD)
% Isolates speech segments by removing silence and ambient noise. A moving
% average of the signal magnitude gives an energy envelope; frames whose
% envelope exceeds 10% of the peak are kept as active speech. This ensures
% MFCC extraction operates only on relevant acoustic data.

% Define window length for energy envelope
winLenOn  = round(0.02 * fsOn);
winLenOff = round(0.02 * fsOff);

% Calculate signal envelope using moving average
envOn  = movmean(abs(preEmphOn),  winLenOn);
envOff = movmean(abs(preEmphOff), winLenOff);

% Define threshold for VAD
threshOn  = 0.1 * max(envOn);
threshOff = 0.1 * max(envOff);

% Create VAD mask
vadMaskOn  = envOn  > threshOn;
vadMaskOff = envOff > threshOff;

% Apply VAD mask to keep only active speech
vadSignalOn  = preEmphOn(vadMaskOn);
vadSignalOff = preEmphOff(vadMaskOff);

% Create time vectors for VAD signals
tVadOn  = (0:length(vadSignalOn)-1)  / fsOn;
tVadOff = (0:length(vadSignalOff)-1) / fsOff;

% Plot results
figure('Name', 'VAD Analysis', 'NumberTitle', 'off');
set(gcf, 'Units', 'centimeters', 'Position', [2, 2, 16, 12]);

subplot(3,2,1); plot(tOn, preEmphOn);
title('Light On: Time');  xlabel('Time (s)'); ylabel('Amp');
subplot(3,2,2); plot(tOff, preEmphOff);
title('Light Off: Time'); xlabel('Time (s)'); ylabel('Amp');

subplot(3,2,3); plot(tOn, vadMaskOn, 'r', 'LineWidth', 1.5); ylim([-0.2, 1.2]);
title('Light On: VAD Mask');  xlabel('Time (s)'); ylabel('Logic');
subplot(3,2,4); plot(tOff, vadMaskOff, 'r', 'LineWidth', 1.5); ylim([-0.2, 1.2]);
title('Light Off: VAD Mask'); xlabel('Time (s)'); ylabel('Logic');

subplot(3,2,5); plot(tVadOn, vadSignalOn);
title('Light On: After VAD');  xlabel('Time (s)'); ylabel('Amp');
subplot(3,2,6); plot(tVadOff, vadSignalOff);
title('Light Off: After VAD'); xlabel('Time (s)'); ylabel('Amp');

%% Framing & Windowing
% Speech is non-stationary, so it is analysed in short (~25 ms) frames.
% Abrupt frame boundaries cause spectral leakage; a Hamming window tapers
% each frame toward its edges to reduce this before the FFT stage.

frameDuration = 0.025;
winLenOn  = round(frameDuration * fsOn);
winLenOff = round(frameDuration * fsOff);

% Create Hamming window
hamWinOn  = hamming(winLenOn);
hamWinOff = hamming(winLenOff);

% Extract a single frame from the middle of the signals
midIdxOn   = round(length(vadSignalOn) / 2);
frameOrigOn = vadSignalOn(midIdxOn : midIdxOn + winLenOn - 1);

midIdxOff   = round(length(vadSignalOff) / 2);
frameOrigOff = vadSignalOff(midIdxOff : midIdxOff + winLenOff - 1);

% Apply window to the frames
frameWinOn  = frameOrigOn  .* hamWinOn;
frameWinOff = frameOrigOff .* hamWinOff;

% Time vectors for a single frame
tFrameOn  = (0:winLenOn-1)  / fsOn;
tFrameOff = (0:winLenOff-1) / fsOff;

% Plot results
figure('Name', 'Windowing Analysis', 'NumberTitle', 'off');
set(gcf, 'Units', 'centimeters', 'Position', [2, 2, 16, 12]);

subplot(3,2,1); plot(tVadOn, vadSignalOn);
title('Light On: After VAD');  xlabel('Time (s)'); ylabel('Amp');
subplot(3,2,2); plot(tVadOff, vadSignalOff);
title('Light Off: After VAD'); xlabel('Time (s)'); ylabel('Amp');

subplot(3,2,3); plot(tFrameOn, hamWinOn, 'LineWidth', 1.5);
title('Hamming Window (25ms)'); xlabel('Time (s)'); ylabel('Amp');
subplot(3,2,4); plot(tFrameOff, hamWinOff, 'LineWidth', 1.5);
title('Hamming Window (25ms)'); xlabel('Time (s)'); ylabel('Amp');

subplot(3,2,5);
plot(tFrameOn, frameOrigOn, '--', 'Color', [0.5 0.5 0.5]); hold on;
plot(tFrameOn, frameWinOn, 'b', 'LineWidth', 1.2); hold off;
title('Light On: Windowed Frame'); xlabel('Time (s)'); ylabel('Amp');

subplot(3,2,6);
plot(tFrameOff, frameOrigOff, '--', 'Color', [0.5 0.5 0.5]); hold on;
plot(tFrameOff, frameWinOff, 'b', 'LineWidth', 1.2); hold off;
title('Light Off: Windowed Frame'); xlabel('Time (s)'); ylabel('Amp');

%% Feature Extraction & MFCC
% MFCCs represent the vocal-tract characteristics of each command. The
% signal is framed, windowed, transformed with the STFT, passed through a
% Mel-scale filterbank, log-scaled, and reduced with a DCT. The first 13
% coefficients are expanded to 39 features by appending Delta and
% Delta-Delta (velocity and acceleration) coefficients.

winLenOn   = round(0.025 * fsOn);
overlapOn  = round(0.015 * fsOn);
winLenOff  = round(0.025 * fsOff);
overlapOff = round(0.015 * fsOff);

windowOn  = hamming(winLenOn);
windowOff = hamming(winLenOff);

figure('Name', 'Spectrogram and MFCC', 'NumberTitle', 'off');
set(gcf, 'Units', 'centimeters', 'Position', [2, 2, 16, 12]);

subplot(2,2,1);
spectrogram(vadSignalOn, windowOn, overlapOn, 1024, fsOn, 'yaxis');
title('Light On Spectrogram');
subplot(2,2,2);
spectrogram(vadSignalOff, windowOff, overlapOff, 1024, fsOff, 'yaxis');
title('Light Off Spectrogram');

% Extract MFCC features (base coefficients shown here for visualisation)
[coeffsOn,  ~, ~, ~] = mfcc(vadSignalOn,  fsOn,  'Window', windowOn,  'OverlapLength', overlapOn);
[coeffsOff, ~, ~, ~] = mfcc(vadSignalOff, fsOff, 'Window', windowOff, 'OverlapLength', overlapOff);

subplot(2,2,3);
imagesc(coeffsOn'); axis xy;
title('Light On MFCC');  xlabel('Frames'); ylabel('Coefficients'); colorbar;
subplot(2,2,4);
imagesc(coeffsOff'); axis xy;
title('Light Off MFCC'); xlabel('Frames'); ylabel('Coefficients'); colorbar;
