close all
clear all
clc

%Directories
work_dir = '\\bigdata\Science\Med\Physiology\PTN\Students\Yasmine\IC_stroke_decoder\ECoG_alignment';
addpath(fullfile(work_dir,'function'));  %look for the correponding folder (with all the functions)
D_i = NR_Palette;                        %add the function NR_Palette to the folder
monkey_dir = 'Lilo_BSI';
data_dir = '\\bigdata\Science\Med\Physiology\PTN\Data\IC_Stroke\LILO_BSI';

%I took the same values as previously
freq_range = [0 200];
sigma = 6;
c_range = [0 2];
isSave = 0;

%Get the list of all the sessions
all_sessions = dir(fullfile(data_dir,'20*'));
all_sessions = {all_sessions([all_sessions.isdir]).name};

for s = 1:length(all_sessions)
    the_sess = all_sessions{s};
    input_dir = fullfile(data_dir,the_sess,[the_sess '_processed']);
    save_dir = fullfile(input_dir,'Figures');
    if ~exist(save_dir,'dir')
        mkdir(save_dir);
    end

    ecog_files = dir(fullfile(input_dir,[monkey_dir '_' the_sess '_tr*_ecog.mat']));

    for f = 1:length(ecog_files)
        trialName = ecog_files(f).name;
        [~,trialBase] = fileparts(trialName);
        trialNum = regexp(trialBase,'tr(\d+)_ecog','tokens','once');
        trialNum = str2double(trialNum{1});

        %Load eCoG and Vicon data
        load(fullfile(input_dir,[monkey_dir '_' the_sess '_tr' num2str(trialNum) '_ecog.mat']),...
            'data','sampleRate','startViconNSP','endViconNSP');
        load(fullfile(input_dir,[monkey_dir '_' the_sess '_tr' num2str(trialNum) '_vicon.mat']),...
            'kinematic','analog');

        %time_range computation from Vicon
        if ~isempty(kinematic.x)
            tsViconCorr = startViconNSP/sampleRate + (1:size(kinematic.x,1))/kinematic.framerate;
            time_range = [tsViconCorr(1) tsViconCorr(end)];
        else
            % fallback sur durée totale ECoG
            nrSamples = length(data(1).Data(1,:));
            tsViconCorr = (0:nrSamples-1)/sampleRate;
            time_range = [tsViconCorr(1) tsViconCorr(end)];
        end
        
        for ar = 1:length(data)
            nrChannels = size(data(ar).Data,1);

            %To get all the channels
            for ch = 1:nrChannels
                dataChan = double(data(ar).Data(ch,:));
                labelChan = data(ar).Label{ch};

                %Spectrogram 
                step = sampleRate/20;
                fftWinSize = sampleRate/4;
                winFunction = hamming(fftWinSize);
                nfft = sampleRate;

                [S,f,t] = spectrogram(dataChan,winFunction,fftWinSize-step,nfft,sampleRate); % S instead of s to avoid conflict with session loop
                amp = abs(S);
                ampNorm = amp ./ repmat(median(amp,2),1,size(amp,2));
                freq2use = f>=freq_range(1) & f<=freq_range(2);
                frequencies = f(freq2use);
                winCenter = t;
                dataAmpRaw = amp(freq2use,:);
                dataAmpNorm = imgaussfilt(ampNorm(freq2use,:),sigma);

                %Kinematics
                ind_WRB = find(strcmp(kinematic.labels,'WRB'));
                if ~isempty(ind_WRB)
                    x = kinematic.x(:,ind_WRB);
                    hasWRB = true;
                else
                    warning('WRB not found for session %s trial %d', the_sess, trialNum);
                    x = nan(length(tsViconCorr),1); % placeholder
                    hasWRB = false;
                end

                %Figure
                if ~isSave
                    fig = figure('Units','normalized','Position',[0 0.1 1 0.8],'Visible','on');
                else
                    fig = figure('Visible','off');
                end

                %Raw spectrogram + kinematics
                subplot(3,1,1)
                hold on
                imagesc(winCenter,frequencies,dataAmpRaw)
                if hasWRB
                    plot(tsViconCorr,(x-200)/200*50-50,'-r','LineWidth',2)
                end
                fill([time_range(1) time_range(2) time_range(2) time_range(1)],...
                     [-1 -1 0 0]*50,'g','EdgeColor','none','FaceAlpha',0.3)
                colormap(D_i)
                set(gca,'xlim',[tsViconCorr(1) tsViconCorr(end)],...
                        'ylim',[-50 freq_range(2)],...
                        'clim',[min(dataAmpRaw(:)),prctile(dataAmpRaw(:),98)],...
                        'ydir','normal')
                title({['ch ' num2str(ch) ':' labelChan],'RAW spectrum aligned with kinematic'})
                xlabel('Time /s'); ylabel('Frequency /Hz');

                %Normalized spectrogram + kinematics
                subplot(3,1,2)
                hold on
                imagesc(winCenter,frequencies,dataAmpNorm)
                if hasWRB
                    plot(tsViconCorr,(x-200)/200*50-50,'-r','LineWidth',2)
                end
                colormap(D_i)
                set(gca,'xlim',time_range,'ylim',[-50 freq_range(2)],'clim',c_range,'ydir','normal')
                title('Normalized spectrum aligned with kinematic')
                xlabel('Time /s'); ylabel('Frequency /Hz');

                %Kinematics only
                subplot(3,1,3)
                hold on
                if hasWRB
                    plot(tsViconCorr,x,'-r')
                end
                set(gca,'xlim',time_range,'ylim',[200 400])
                title('Wrist-x VICON results aligned')
                xlabel('Time /s'); ylabel('wrist x position /mm');

                %Save or pause
                if isSave
                    fname = sprintf('%s_tr%d_ch%d_%s.png',the_sess,trialNum,ch,labelChan);
                    saveas(fig,fullfile(save_dir,fname))
                    saveas(fig,fullfile(save_dir,strrep(fname,'.png','.fig')))
                    close(fig)
                else
                    uiwait(fig) % wait until figure closed manually for inspection
                end

            end
        end
    end
end
