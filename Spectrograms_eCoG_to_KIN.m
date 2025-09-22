close all; 
clear; 
clc;

%Directories
work_dir   = '\\bigdata\Science\Med\Physiology\PTN\Yasmine\IC-stroke_connectivity';
addpath(fullfile(work_dir,'function'))
D_i        = NR_Palette;
monkey_dir = 'Lilo';
data_dir   = '\\bigdata\Science\Med\Physiology\PTN\Data\IC_Stroke\LILO_BSI';  

%% Get the sessions to process
sheet_file = 'W:\Students\Yasmine\IC_stroke_decoder\ECoG_alignment\ECoG_Data_preprocessing_matlab\Analysis\NHP_ECoG_Lilo_20250922.csv';
T = readtable(sheet_file, 'VariableNamingRule', 'preserve', 'Delimiter', ',');
disp(T.Properties.VariableNames)

for iRow = 1:height(T)

    sess      = num2str(T.("Date")(iRow));                 
    task      = string(T.("Task")(iRow));                  
    trial_str = string(T.("NWB file?")(iRow));         

    if trial_str == "" || ismissing(trial_str)
        continue;
    end
    
    trials = eval(trial_str);  
    
    for trial_num = trials
        fprintf('Processing session %s, trial %d (%s)\n', sess, trial_num, task);

        %% Load ECoG
        input_dir = fullfile(data_dir, sess, [sess '_processed']);
        ecog_file = fullfile(input_dir, [monkey_dir '_' sess '_tr' num2str(trial_num) '_ecog.mat']);
        if ~exist(ecog_file,'file')
            warning('ECoG manquant pour session %s trial %d', sess, trial_num);
            continue
        end
        load(ecog_file, 'data','sampleRate','startViconNSP','endViconNSP');

        %Load Vicon if available
        vicon_file = fullfile(input_dir, [monkey_dir '_' sess '_tr' num2str(trial_num) '_vicon.mat']);
        hasVicon   = exist(vicon_file,'file');
        kin_x      = [];
        tsViconCorr = [];

        if hasVicon
            load(vicon_file, 'kinematic');
            ind_WRB = find(strcmp(kinematic.labels,'WRB'));
            if ~isempty(ind_WRB)
                kin_x     = kinematic.x(:,ind_WRB);
                fs_video  = kinematic.framerate;
                durKIN    = length(kin_x);
                tsViconCorr = startViconNSP/sampleRate + (1:durKIN)/fs_video;
            else
                warning('No WRB marker for %s trial %d', sess, trial_num);
            end
        end

        %Output dirs (in PTN\Yasmine\IC-stroke_connectivity)
        export_root   = fullfile(work_dir, 'data', sess);
        fig_dir       = fullfile(export_root, 'Figures');
        trial_dir     = fullfile(export_root, 'Trials');
        if ~exist(fig_dir,'dir'); mkdir(fig_dir); end
        if ~exist(trial_dir,'dir'); mkdir(trial_dir); end

        %Loop : array + channels
        for ar = 1:length(data)
            for ch = 1:length(data(ar).Label)
                dataChan  = double(data(ar).Data(ch,:));
                labelChan = data(ar).Label{ch};

                %% Spectrogram
                step = sampleRate/20;
                fftWinSize = sampleRate/4;
                winFunction = hamming(fftWinSize);
                nfft = sampleRate;
                [s,f,t] = spectrogram(dataChan,winFunction,fftWinSize-step,nfft,sampleRate);
                amp = abs(s);
                ampNorm = amp ./ repmat(median(amp')',1,size(amp,2));

                freq_range = [0 200];
                freq2use   = find(f>=freq_range(1) & f<=freq_range(2));
                frequencies = f(freq2use);
                winCenter   = t;

            
                c_range = [0 2];    
                sigma   = 6;

                %Figure simple
                figure('Units','normalized','Position',[0 0.1 1 0.7])

                %RAW spectrum
                subplot(3,1,1)
                dataAmp = amp(freq2use,:);
                imagesc(winCenter,frequencies,dataAmp)
                colormap(D_i)
                set(gca,'ydir','normal')
                set(gca,'ylim',[0 freq_range(2)],...
                        'clim',[min(dataAmp(:)) prctile(dataAmp(:),98)])
                title({['ch' num2str(ch) ':' labelChan],'RAW spectrum'})
                xlabel('Time /s'); ylabel('Frequency /Hz')

                %Normalized spectrum
                subplot(3,1,2)
                dataAmpNorm = imgaussfilt(ampNorm(freq2use,:),sigma);
                imagesc(winCenter,frequencies,dataAmpNorm)
                colormap(D_i)
                set(gca,'ydir','normal')
                set(gca,'ylim',[0 freq_range(2)],'clim',c_range)
                title('Normalized spectrum')
                xlabel('Time /s'); ylabel('Frequency /Hz')

                %Kinematic (if available)
                subplot(3,1,3)
                if ~isempty(kin_x) && ~isempty(tsViconCorr)
                    plot(tsViconCorr,kin_x,'-r')
                    set(gca,'xlim',[tsViconCorr(1) tsViconCorr(end)],...
                            'ylim',[min(kin_x) max(kin_x)])
                    title('Wrist-x VICON results aligned')
                    xlabel('Time /s'); ylabel('wrist x position /mm')
                else
                    text(0.5,0.5,'No Vicon data','Units','normalized','HorizontalAlignment','center')
                    set(gca,'XTick',[],'YTick',[])
                end

                %Save figure
                saveas(gcf, fullfile(fig_dir, sprintf('%s_tr%d_chan%s.png', sess, trial_num, labelChan)));
                close(gcf);

                %Export .mat
                S.signal     = dataChan;
                S.time_ecog  = (0:length(dataChan)-1)/sampleRate;
                S.metadata   = struct('session',sess,'trial',trial_num, ...
                                      'array',ar,'channel',labelChan,'task',task);

                %Add kinematic if available
                if ~isempty(kin_x)
                    S.kin_x      = kin_x;
                    S.time_vicon = tsViconCorr;
                else
                    S.kin_x      = [];
                    S.time_vicon = [];
                end

                save_file = fullfile(trial_dir, sprintf('%s_tr%d_%s.mat', sess, trial_num, labelChan));
                save(save_file, '-struct', 'S');
            end
        end
    end
end
