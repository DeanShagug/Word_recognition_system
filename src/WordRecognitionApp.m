classdef WordRecognitionApp < matlab.apps.AppBase
    % WordRecognitionApp  Real-time voice-command GUI.
    %
    % Records 2 seconds of audio from the microphone, runs the same
    % pre-emphasis -> VAD -> MFCC pipeline used in training, scores the
    % features against the three trained GMMs, and drives a simulated
    % light (LED lamp) plus a confidence gauge and a system log.
    %
    % Before running, train the models (train_gmm_models.m) so that
    % 'trained_gmm_models.mat' exists, and set MODEL_PATH below to point
    % at it. This file is the plain-text export of the App Designer app
    % (app1.mlapp) with hard-coded local paths and third-party branding
    % removed.

    % Properties that correspond to app components
    properties (Access = public)
        UIFigure        matlab.ui.Figure
        SystemLog       matlab.ui.control.TextArea
        SystemlogLabel  matlab.ui.control.Label
        ConfGauge       matlab.ui.control.Gauge
        ConfidnceLabel  matlab.ui.control.Label
        PowerSwitch     matlab.ui.control.StateButton
        LCDDisplay      matlab.ui.control.EditField
        TextLabel       matlab.ui.control.Label
        LEDBulb         matlab.ui.control.Lamp
        StatusLabel     matlab.ui.control.Label
        FreqGraph       matlab.ui.control.UIAxes
        TimeGraph       matlab.ui.control.UIAxes
    end

    properties (Access = private)
        fs = 48000;      % Sample rate (match your microphone)
        recObj;          % Microphone object

        % Model variables
        GmmOn;
        GmmOff;
        GmmOther;
    end

    properties (Constant, Access = private)
        % Path to the trained model file (edit as needed)
        MODEL_PATH = fullfile('models', 'trained_gmm_models.mat');
    end

    % Callbacks that handle component events
    methods (Access = private)

        % Code that executes after component creation
        function startupFcn(app)
            % Initialize UI state
            app.LEDBulb.Color = [0.2, 0.2, 0.2];   % Off (dark grey)
            app.LCDDisplay.Value = 'Loading...';
            app.SystemLog.Value = {'System Initializing...'};

            % Load the trained models
            try
                modelData = load(app.MODEL_PATH);
                app.GmmOn    = modelData.gmmOn;
                app.GmmOff   = modelData.gmmOff;
                app.GmmOther = modelData.gmmOther;

                app.LCDDisplay.Value = 'System Ready';
                app.SystemLog.Value = {'Models Loaded Successfully.'; 'Waiting for user...'};
            catch
                app.LCDDisplay.Value = 'Error: No Model';
                app.SystemLog.Value = {'ERROR: trained_gmm_models.mat not found at specified path!'};
                uialert(app.UIFigure, 'Please train and save models first!', 'Error');
            end

            % Initialize recorder
            app.recObj = audiorecorder(app.fs, 16, 1);
        end

        % Value changed function: PowerSwitch
        function PowerSwitchValueChanged(app, ~)
            isPressed = app.PowerSwitch.Value;

            if isPressed == 1
                % 1. Update UI for recording state
                app.LCDDisplay.Value = 'Listening...';
                app.LCDDisplay.FontColor = [0, 0.8, 0];
                app.SystemLog.Value = {'--- New Command ---'; 'Microphone active...'};
                app.ConfGauge.Value = 0;
                app.LEDBulb.Color = [0.2, 0.2, 0.2];
                drawnow;

                % 2. Record audio
                recordDuration = 2;
                recordblocking(app.recObj, recordDuration);
                audioData = getaudiodata(app.recObj);
                app.SystemLog.Value = [app.SystemLog.Value; 'Recording finished.'];
                drawnow;

                % 3. Plot time domain
                t = (0:length(audioData)-1) / app.fs;
                plot(app.TimeGraph, t, audioData, 'b');

                % 4. Plot frequency domain
                Y = abs(fft(audioData));
                L = length(audioData);
                f = app.fs * (0:(L/2)) / L;
                plot(app.FreqGraph, f, Y(1:floor(L/2)+1), 'r');
                drawnow;

                % 5. Classify
                detectedWord = app.processAudio(audioData);

                % 6. Update final UI state
                app.updateSystemState(detectedWord);

                % 7. Reset button
                pause(0.5);
                app.PowerSwitch.Value = 0;
                app.LCDDisplay.FontColor = [0, 0, 0];
            end
        end

        % Process audio: pre-emphasis -> VAD -> MFCC -> GMM scoring
        function resultLabel = processAudio(app, audioIn)
            % Pre-emphasis
            audioPre = filter([1, -0.97], 1, audioIn);

            % VAD
            winLen = round(0.02 * app.fs);
            env    = movmean(abs(audioPre), winLen);
            thresh = 0.1 * max(env);
            vadSignal = audioPre(env > thresh);

            winLenMfcc = round(0.025 * app.fs);

            % Guard against too-short signals
            if isempty(vadSignal) || length(vadSignal) < winLenMfcc
                resultLabel = 'Unrecognized';
                app.ConfGauge.Value = 0;
                return;
            end

            % Extract 39 MFCC features
            overlapMfcc = round(0.015 * app.fs);
            window = hamming(winLenMfcc);
            [coeffs, delta, deltaDelta, ~] = mfcc(vadSignal, app.fs, ...
                'Window', window, 'OverlapLength', overlapMfcc);
            features = [coeffs, delta, deltaDelta];

            % GMM scores (mean log-likelihood per frame)
            scoreOn    = mean(log(pdf(app.GmmOn,    features) + 1e-10));
            scoreOff   = mean(log(pdf(app.GmmOff,   features) + 1e-10));
            scoreOther = mean(log(pdf(app.GmmOther, features) + 1e-10));

            scores = [scoreOn, scoreOff, scoreOther];
            labels = {'Lighton', 'Lightoff', 'Other'};

            % Best match
            [maxScore, idx] = max(scores);
            resultLabel = labels{idx};

            % Confidence via softmax over the three scores
            shiftedScores = scores - maxScore;
            probs = exp(shiftedScores);
            confPercent = (probs(idx) / sum(probs)) * 100;

            app.ConfGauge.Value = confPercent;
            app.SystemLog.Value = [app.SystemLog.Value; ...
                sprintf('Detected: %s (%.1f%%)', resultLabel, confPercent)];
            drawnow;
        end

        % Update UI based on the recognised word
        function updateSystemState(app, word)
            app.LCDDisplay.Value = word;

            if strcmp(word, 'Lighton')
                app.LEDBulb.Color = [1, 1, 0];        % Yellow
                app.SystemLog.Value = [app.SystemLog.Value; 'ACTION: Turning Light ON.'];
            elseif strcmp(word, 'Lightoff')
                app.LEDBulb.Color = [0.2, 0.2, 0.2];  % Grey
                app.SystemLog.Value = [app.SystemLog.Value; 'ACTION: Turning Light OFF.'];
            elseif strcmp(word, 'Unrecognized')
                app.LEDBulb.Color = [0.2, 0.2, 0.2];
                app.SystemLog.Value = [app.SystemLog.Value; 'ACTION: Audio Unrecognized.'];
            else
                app.LEDBulb.Color = [1, 0, 0];        % Red (Other)
                app.SystemLog.Value = [app.SystemLog.Value; 'ACTION: Other word detected.'];
            end
            drawnow;
        end
    end

    % Component initialization
    methods (Access = private)

        % Create UIFigure and components
        function createComponents(app)
            % Create UIFigure and hide until all components are created
            app.UIFigure = uifigure('Visible', 'off');
            app.UIFigure.Position = [100 100 640 480];
            app.UIFigure.Name = 'Word Recognition';

            % TimeGraph
            app.TimeGraph = uiaxes(app.UIFigure);
            title(app.TimeGraph, 'Time Domain');
            xlabel(app.TimeGraph, 'Time [s]');
            ylabel(app.TimeGraph, 'Amp [v]');
            app.TimeGraph.Position = [15 203 280 204];

            % FreqGraph
            app.FreqGraph = uiaxes(app.UIFigure);
            title(app.FreqGraph, 'Freq Domain');
            xlabel(app.FreqGraph, 'Freq');
            ylabel(app.FreqGraph, 'Amp');
            app.FreqGraph.Position = [320 203 307 198];

            % StatusLabel
            app.StatusLabel = uilabel(app.UIFigure);
            app.StatusLabel.HorizontalAlignment = 'right';
            app.StatusLabel.Position = [358 429 39 22];
            app.StatusLabel.Text = 'Status';

            % LEDBulb
            app.LEDBulb = uilamp(app.UIFigure);
            app.LEDBulb.Position = [412 429 20 20];
            app.LEDBulb.Color = [0.651 0.651 0.651];

            % TextLabel
            app.TextLabel = uilabel(app.UIFigure);
            app.TextLabel.HorizontalAlignment = 'right';
            app.TextLabel.Position = [160 429 30 22];
            app.TextLabel.Text = 'Text ';

            % LCDDisplay
            app.LCDDisplay = uieditfield(app.UIFigure, 'text');
            app.LCDDisplay.FontName = 'Courier';
            app.LCDDisplay.FontWeight = 'bold';
            app.LCDDisplay.BackgroundColor = [0.8 0.8 0.8];
            app.LCDDisplay.Position = [205 429 100 22];
            app.LCDDisplay.Value = 'System Ready';

            % PowerSwitch
            app.PowerSwitch = uibutton(app.UIFigure, 'state');
            app.PowerSwitch.ValueChangedFcn = createCallbackFcn(app, @PowerSwitchValueChanged, true);
            app.PowerSwitch.Text = 'START';
            app.PowerSwitch.Position = [332 91 100 22];

            % ConfidnceLabel
            app.ConfidnceLabel = uilabel(app.UIFigure);
            app.ConfidnceLabel.HorizontalAlignment = 'center';
            app.ConfidnceLabel.Position = [513 16 68 30];
            app.ConfidnceLabel.Text = {'Recognition'; 'Confidence'};

            % ConfGauge
            app.ConfGauge = uigauge(app.UIFigure, 'circular');
            app.ConfGauge.Position = [486 61 120 120];

            % SystemlogLabel
            app.SystemlogLabel = uilabel(app.UIFigure);
            app.SystemlogLabel.HorizontalAlignment = 'right';
            app.SystemlogLabel.Position = [15 180 64 22];
            app.SystemlogLabel.Text = 'System log';

            % SystemLog
            app.SystemLog = uitextarea(app.UIFigure);
            app.SystemLog.Position = [89 16 197 174];

            % Show the figure after all components are created
            app.UIFigure.Visible = 'on';
        end
    end

    % App creation and deletion
    methods (Access = public)

        % Construct app
        function app = WordRecognitionApp
            createComponents(app)
            registerApp(app, app.UIFigure)
            runStartupFcn(app, @startupFcn)
            if nargout == 0
                clear app
            end
        end

        % Code that executes before app deletion
        function delete(app)
            delete(app.UIFigure)
        end
    end
end
