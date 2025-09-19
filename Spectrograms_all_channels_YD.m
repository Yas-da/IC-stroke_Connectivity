close all
clear all
clc

%% Directories
work_dir = '\\bigdata\Science\Med\Physiology\PTN\Students\Yasmine\IC_stroke_decoder\ECoG_alignment';
addpath(fullfile(work_dir,'function'));  % look for the corresponding folder (with all the functions)
D_i = NR_Palette;                        % add the function NR_Palette to the folder
monkey_dir = 'Lilo_BSI';
data_dir = '\\bigdata\Science\Med\Physiology\PTN\Data\IC_Stroke\LILO_BSI';

%% Parameters
freq_range = [0 200];
sigma = 6;
c_range = [0 2];
isSave = 0;  % 1 = save all figures automatically
time_range_default = [110 140];  % same as reference

%% Get all sessions
all_sessions = dir(fullfile(data_dir,'20*'));
all_sessions = {all_sessions([all_sessions.isdir]).name};

%% Loop over sessions
for s = 1:length(all_sessions)
    the_sess = all_sessions{s};
    input_dir = fullfile(data_dir,the_sess,[the_sess '_processed']);
    save_dir = fullfile(input_dir,'Figures');
    if ~exist(save_dir,'dir')
        mkdir(save_dir);
    end

    %% List all trials
    ecog_files = dir(fullfile(input_dir,[monkey_dir '_' the_sess '_tr*_ecog.mat']));

    for f = 1:length(ecog_files)
        trialName = ecog_files(f).name;
        [~,trialBase] = fileparts(trialName);
        trialNum = regexp(trialBase,'tr(\d+)_ecog','tokens','once');
        trialNum = str2double(trialNum{1});

        %% Load eCoG and Vicon
        load(fullfile(input_dir,[monkey_dir '_' the_sess '_tr' num2str(trialNum) '_ecog.mat']),...
            'data','sampleRate','startViconNSP','endViconNSP');
        load(fullfile(input_dir,[monkey_dir '_' the_sess '_tr' num2str(trialNum) '_vicon.mat']),...
            'kinematic','analog');

        %% Loop over channels
        for ch = 1:max(cellfun(@numel,{data(:).Label}))  % all channels
            array2show = [2 1];  % same as reference
            figure('Units','normalized','Position',[0 0.1 1 0.8], 'Visible', ~isSave);

            for kk = 1:2
                ar = array2show(kk);
                dataChan = double(data(ar).Data(ch,:));
                labelChan = data(ar).Label{ch};

                % Spectrogram
                step = sampleRate/20;
                fftWinSize = sampleRate/4;
                winFunction = hamming(fftWinSize);
                nfft = sampleRate;

                [S,f,t] = spectrogram(dataChan,winFunction,fftWinSize-step,nfft,sampleRate);
                amp = abs(S);
                ampNorm = amp./repmat(median(amp')',1,size(amp,2));
                freq2use = find(f>=freq_range(1) & f<=freq_range(2));
                frequencies = f(freq2use);
                winCenter = t;
                dataAmpRaw = amp(freq2use,:);
                dataAmpNorm = imgaussfilt(ampNorm(freq2use,:),sigma);

                % Kinematics (x,y,z)
                ind_WRB = find(ismember(kinematic.labels,'WRB')==1);
                if isempty(ind_WRB)
                    warning('WRB not found for session %s trial %d', the_sess, trialNum);
                    x = nan(size(t));
                else
                    x = kinematic.x(:,ind_WRB);
                    y = kinematic.y(:,ind_WRB);
                    z = kinematic.z(:,ind_WRB);
                    durKIN = length(x);
                    fs_video = kinematic.framerate;
                    tsViconCorr = startViconNSP/sampleRate + [1:durKIN]/fs_video;
                end

                %% Plot raw spectrogram + kinematics (1ère sous-figure)
                subplot(3,1,1)
                hold on
                imagesc(winCenter,frequencies,dataAmpRaw)
                colormap(D_i)
                if exist('tsViconCorr','var')
                    plot(tsViconCorr,(x-200)/200*50-50,'-r','linewidth',2)
                end
                fill([time_range_default(1) time_range_default(2) ...
                      time_range_default(2) time_range_default(1)], ...
                     [-1 -1 0 0]*50,'g','EdgeColor','none','FaceAlpha',0.3)
                set(gca,'xlim',[winCenter(1) winCenter(end)],'ylim',[-50 freq_range(2)],...
                        'clim',[min(dataAmpRaw(:)),prctile(dataAmpRaw(:),98)],'ydir','normal')
                title({['ch ' num2str(ch) ':' labelChan],'RAW spectrum aligned with kinematic'})
                xlabel('Time /s'); ylabel('Frequency / Hz');

                %% Plot normalized spectrogram + kinematics (2ème sous-figure)
                subplot(3,1,2)
                hold on
                imagesc(winCenter,frequencies,dataAmpNorm)
                colormap(D_i)
                if exist('tsViconCorr','var')
                    plot(tsViconCorr,(x-200)/200*50-50,'-r','linewidth',2)
                end
                set(gca,'xlim',time_range_default,'ylim',[-50 freq_range(2)],'clim',c_range,'ydir','normal')
                title('Normalized spectrum aligned with kinematic')
                xlabel('Time /s'); ylabel('Frequency / Hz');

                %% Plot kinematics only (3ème sous-figure)
                subplot(3,1,3)
                hold on
                if exist('tsViconCorr','var')
                    plot(tsViconCorr,x,'-r')
                end
                set(gca,'xlim',time_range_default,'ylim',[200 400])
                title('Wrist-x VICON results aligned')
                xlabel('Time /s'); ylabel('wrist x position /mm');
            end

            % Save or pause
            if isSave
                fname = sprintf('%s_tr%d_ch%d_%s.png',the_sess,trialNum,ch,labelChan);
                saveas(gcf,fullfile(save_dir,fname))
                saveas(gcf,fullfile(save_dir,strrep(fname,'.png','.fig')))
                close(gcf)
            else
                pause(0.3)  % pour voir les figures, ajustable
            end
        end
    end
end
