
Processing session 20250522, trial 1 (left - SC)
Warning: Found 67 channels for 20250522_tr1 (expected 64). 
> In spectrograms_for_connectivity_analysis (line 149) 
Error using subplot
Index exceeds number of subplots.

Error in spectrograms_for_connectivity_analysis (line 172)
            ax = subplot(plot_grid_rows, plot_grid_cols, chIdx);
 

%% pipeline_eCog_export_per_trial.m
close all; clear; clc;

%% Directories
work_root  = '\\bigdata\Science\Med\Physiology\PTN\Yasmine\IC-stroke_connectivity';
addpath(fullfile(work_root,'function'))  % si besoin
D_i        = NR_Palette;
monkey_dir = 'Lilo';
data_dir   = '\\bigdata\Science\Med\Physiology\PTN\Data\IC_Stroke\LILO_BSI';

%% Parameters
default_fs = 2000;   % si sampleRate absent on prend 2000 Hz
plot_grid_rows = 8;  % grille 8x8 pour 64 canaux
plot_grid_cols = 8;

%% Get sessions to process
sheet_file = 'W:\Students\Yasmine\IC_stroke_decoder\ECoG_alignment\ECoG_Data_preprocessing_matlab\Analysis\NHP_ECoG_Lilo_20250922.csv';
T = readtable(sheet_file, 'VariableNamingRule', 'preserve', 'Delimiter', ',');
disp(T.Properties.VariableNames)

% OPTIONAL: reprendre à partir d'une session/trial spécifique
% start_sess = 20250718; start_trial_for_start_sess = 3;
% si tu veux reprendre à partir d'une session précise, décommente ci-dessus

for iRow = 1:height(T)

    sess_str  = num2str(T.("Date")(iRow));    % garde string pour nom de dossier
    sess_num  = str2double(sess_str);        % pour comparaisons numériques si besoin
    task      = string(T.("Task")(iRow));                  
    trial_str = string(T.("NWB file?")(iRow));         

    if trial_str == "" || ismissing(trial_str)
        continue;
    end
    
    trials = eval(trial_str);  
    
    % if exist start_sess logic uncomment above lines
    % if exist('start_sess','var') && sess_num < start_sess
    %     fprintf('Skipping session %s\n', sess_str); continue;
    % end

    for trial_num = trials
        % if sess_num == start_sess && exist('start_trial_for_start_sess','var') && trial_num < start_trial_for_start_sess
        %     continue;
        % end

        fprintf('Processing session %s, trial %d (%s)\n', sess_str, trial_num, task);

        %% Load ECoG
        input_dir = fullfile(data_dir, sess_str, [sess_str '_processed']);
        ecog_file = fullfile(input_dir, [monkey_dir '_' sess_str '_tr' num2str(trial_num) '_ecog.mat']);
        if ~exist(ecog_file,'file')
            warning('No ecog for the sess %s trial %d', sess_str, trial_num);
            continue
        end
        Sload = load(ecog_file);  % charge ce qui est dispo
        if isfield(Sload,'sampleRate') && ~isempty(Sload.sampleRate)
            sampleRate = Sload.sampleRate;
        else
            sampleRate = default_fs;
            warning('sampleRate missing in %s -> using default %d Hz', ecog_file, default_fs);
        end
        if isfield(Sload,'data')
            data_struct = Sload.data;
        else
            warning('No variable ''data'' inside %s. Skipping.', ecog_file);
            continue
        end

        % startViconNSP peut être absent, vide ou vecteur
        if isfield(Sload,'startViconNSP') && ~isempty(Sload.startViconNSP)
            if numel(Sload.startViconNSP) >= 1
                startViconNSP_val = Sload.startViconNSP(1);
            else
                startViconNSP_val = 0;
            end
        else
            startViconNSP_val = 0;
        end
        % endViconNSP not required here but we keep robust handling
        if isfield(Sload,'endViconNSP') && ~isempty(Sload.endViconNSP)
            if numel(Sload.endViconNSP) >= 1
                endViconNSP_val = Sload.endViconNSP(1);
            else
                endViconNSP_val = [];
            end
        else
            endViconNSP_val = [];
        end

        %% Load Vicon if available
        vicon_file = fullfile(input_dir, [monkey_dir '_' sess_str '_tr' num2str(trial_num) '_vicon.mat']);
        hasVicon   = exist(vicon_file,'file');
        kin_x      = [];
        tsViconCorr = [];

        if hasVicon
            V = load(vicon_file);
            if isfield(V,'kinematic')
                kinematic = V.kinematic;
                ind_WRB = find(strcmp(kinematic.labels,'WRB'));
                if ~isempty(ind_WRB)
                    kin_x     = kinematic.x(:,ind_WRB);
                    fs_video  = kinematic.framerate;
                    durKIN    = length(kin_x);
                    % robust: startViconNSP_val already fixé (0 si absent)
                    tsViconCorr = startViconNSP_val/sampleRate + (0:durKIN-1)/fs_video;
                else
                    warning('No vicon marker WRB for %s trial %d', sess_str, trial_num);
                end
            else
                warning('Vicon file %s exists but no ''kinematic'' variable found', vicon_file);
            end
        end

        %% Create output directories (per trial)
        export_root   = fullfile(work_root, 'data', sess_str);
        trial_root    = fullfile(export_root, sprintf('trial%d', trial_num));
        fig_dir       = fullfile(trial_root, 'Figures');
        trial_dir     = fullfile(trial_root, 'MatFiles');
        if ~exist(fig_dir,'dir');  mkdir(fig_dir);  end
        if ~exist(trial_dir,'dir'); mkdir(trial_dir); end

        %% Assemble signals matrix (channels x time)
        % assume data_struct is array of arrays, with fields Label and Data
        chan_labels = {};
        signals_cell = {};
        for ar = 1:length(data_struct)
            for ch = 1:length(data_struct(ar).Label)
                chan_labels{end+1} = data_struct(ar).Label{ch};
                signals_cell{end+1} = double(data_struct(ar).Data(ch,:));
            end
        end
        % convert to matrix (n_channels x n_samples)
        nChannels = length(signals_cell);
        % find max length (should be same for all, if not we pad/truncate)
        maxlen = max(cellfun(@numel, signals_cell));
        signals = zeros(nChannels, maxlen);
        for k = 1:nChannels
            v = signals_cell{k}(:)';
            if numel(v) < maxlen
                v = [v zeros(1,maxlen-numel(v))]; % pad with zeros (or NaN if preferred)
            end
            signals(k,:) = v;
        end
        % ensure channel count is 64 - warn if not
        if nChannels ~= 64
            warning('Found %d channels for %s_tr%d (expected 64).', nChannels, sess_str, trial_num);
        end

        %% Spectrogram params (time-frequency computed per channel)
        step = round(sampleRate/20);
        fftWinSize = round(sampleRate/4);
        winFunction = hamming(fftWinSize);
        nfft = sampleRate;
        freq_range = [0 200];
        c_range = [0 2];
        sigma = 6;

        % Precompute time/freq axes from first channel
        [s,f,t] = spectrogram(signals(1,:), winFunction, fftWinSize-step, nfft, sampleRate);
        freq2use = find(f>=freq_range(1) & f<=freq_range(2));
        frequencies = f(freq2use);
        winCenter = t;

        %% Create two figures per trial: RAW grid and NORMALIZED grid
        % RAW
        fig_raw = figure('Units','normalized','Position',[0 0.05 1 0.9],'Visible','off');
        colormap(fig_raw, D_i);
        for chIdx = 1:nChannels
            ax = subplot(plot_grid_rows, plot_grid_cols, chIdx);
            [s_ch,~,~] = spectrogram(signals(chIdx,:), winFunction, fftWinSize-step, nfft, sampleRate);
            dataAmp = abs(s_ch);
            imagesc(winCenter, frequencies, dataAmp(freq2use,:));
            set(gca,'ydir','normal'); title(sprintf('ch%d: %s', chIdx, chan_labels{chIdx}));
            if chIdx==nChannels
                xlabel('Time /s'); ylabel('Freq /Hz');
            else
                set(gca,'XTick',[],'YTick',[]);
            end
            clim([min(dataAmp(:)) prctile(dataAmp(:),98)]);
        end
        % add kinematic overlay in a small separate figure if available
        if ~isempty(kin_x)
            % create a little extra figure for kin
            fig_kin = figure('Units','normalized','Position',[0.1 0.1 0.3 0.2],'Visible','off');
            plot(tsViconCorr, kin_x, '-r'); title('Wrist-x VICON');
            xlabel('Time /s'); ylabel('wrist x /mm');
            exportgraphics(fig_kin, fullfile(fig_dir, sprintf('%s_tr%d_kinematic.png', sess_str, trial_num)), 'Resolution',300);
            savefig(fig_kin, fullfile(fig_dir, sprintf('%s_tr%d_kinematic.fig', sess_str, trial_num)));
            close(fig_kin);
        end

        % Save RAW figure: both .fig and .png
        raw_fig_name = fullfile(fig_dir, sprintf('%s_tr%d_raw_grid.fig', sess_str, trial_num));
        savefig(fig_raw, raw_fig_name);
        exportgraphics(fig_raw, fullfile(fig_dir, sprintf('%s_tr%d_raw_grid.png', sess_str, trial_num)), 'Resolution',300);
        close(fig_raw);

        % NORMALIZED (median norm + gaussian smoothing)
        fig_norm = figure('Units','normalized','Position',[0 0.05 1 0.9],'Visible','off');
        colormap(fig_norm, D_i);
        for chIdx = 1:nChannels
            ax = subplot(plot_grid_rows, plot_grid_cols, chIdx);
            [s_ch,~,~] = spectrogram(signals(chIdx,:), winFunction, fftWinSize-step, nfft, sampleRate);
            amp = abs(s_ch);
            ampNorm = amp ./ repmat(median(amp')',1,size(amp,2));
            dataAmpNorm = imgaussfilt(ampNorm(freq2use,:), sigma);
            imagesc(winCenter, frequencies, dataAmpNorm);
            set(gca,'ydir','normal'); title(sprintf('ch%d: %s', chIdx, chan_labels{chIdx}));
            if chIdx==nChannels
                xlabel('Time /s'); ylabel('Freq /Hz');
            else
                set(gca,'XTick',[],'YTick',[]);
            end
            set(gca,'ylim',[0 freq_range(2)], 'clim', c_range);
        end

        % Save NORMALIZED figure
        norm_fig_name = fullfile(fig_dir, sprintf('%s_tr%d_norm_grid.fig', sess_str, trial_num));
        savefig(fig_norm, norm_fig_name);
        exportgraphics(fig_norm, fullfile(fig_dir, sprintf('%s_tr%d_norm_grid.png', sess_str, trial_num)), 'Resolution',300);
        close(fig_norm);

        %% Save .mat per trial (ALL channels in one file)
        out_struct.signals = signals;           % channels x time
        out_struct.time_ecog = (0:size(signals,2)-1) / sampleRate;
        out_struct.channel_labels = chan_labels;
        out_struct.metadata.session = sess_str;
        out_struct.metadata.trial = trial_num;
        out_struct.metadata.task = task;
        out_struct.metadata.sampleRate = sampleRate;
        if ~isempty(kin_x)
            out_struct.kin_x = kin_x;
            out_struct.time_vicon = tsViconCorr;
        else
            out_struct.kin_x = [];
            out_struct.time_vicon = [];
        end

        mat_out_name = fullfile(trial_dir, sprintf('%s_tr%d_allChannels.mat', sess_str, trial_num));
        save(mat_out_name, '-struct', 'out_struct', '-v7.3');

        fprintf('Saved trial %s_tr%d: %s (figs -> %s, mat -> %s)\n', sess_str, trial_num, task, fig_dir, mat_out_name);

    end
end
