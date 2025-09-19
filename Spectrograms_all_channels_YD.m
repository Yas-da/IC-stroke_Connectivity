close all
clear all
clc

%en vrai mettre ailleurs que dans le PTN comme c'est assez lourd
work_dir = 'C:\Users\physio\Documents\Code\ECoG_Data_preprocessing_matlab';
addpath(fullfile(work_dir,'function'))  
D_i = NR_Palette;
monkey_dir = 'Lilo';
data_dir = '\\bigdata\Science\Med\Physiology\PTN\Data\IC_Stroke\LILO_BSI';

all_sessions = dir(fullfile(data_dir,'20*')); 
all_sessions = {all_sessions([all_sessions.isdir]).name};

freq_range = [0 200];
targetRate = 2000;
sigma = 6;
c_range = [0 2];
isSave = 0;

for s = 1:length(all_sessions)
    the_sess = all_sessions{s};
    input_dir = fullfile(data_dir,the_sess,[the_sess '_processed']);
    save_dir = fullfile(input_dir,'Figures');
    if ~exist(save_dir,'dir')
        mkdir(save_dir)
    end

    ecog_files = dir(fullfile(input_dir,[monkey_dir '_' the_sess '_tr*_ecog.mat']));

    for f = 1:length(ecog_files)
        trialName = ecog_files(f).name;
        [~,trialBase] = fileparts(trialName);
        trialNum = regexp(trialBase,'tr(\d+)_ecog','tokens','once');
        trialNum = str2double(trialNum{1});

        load(fullfile(input_dir,[monkey_dir '_' the_sess '_tr' num2str(trialNum) '_ecog.mat']),...
            'data','comment','sampleRate','startViconNSP','endViconNSP')
        load(fullfile(input_dir,[monkey_dir '_' the_sess '_tr' num2str(trialNum) '_vicon.mat']),...
            'kinematic','analog')

        for ar = 1:length(data)
            nrChannels = size(data(ar).Data,1);

            for ch = 1:nrChannels
                dataChan = double(data(ar).Data(ch,:));
                labelChan = data(ar).Label{ch};

                step = sampleRate/20;
                fftWinSize = sampleRate/4;
                winFunction = hamming(fftWinSize);
                nfft = sampleRate;

                [s,f,t] = spectrogram(dataChan,winFunction,fftWinSize-step,nfft,sampleRate);
                amp = abs(s);
                ampNorm = amp ./ repmat(median(amp,2),1,size(amp,2));
                freq2use = f>=freq_range(1) & f<=freq_range(2);
                frequencies = f(freq2use);
                winCenter = t;

                %change to another marker if needed
                ind_WRB = find(strcmp(kinematic.labels,'WRB'));
                if isempty(ind_WRB)
                    continue
                end
                x = kinematic.x(:,ind_WRB);
                durKIN = length(x);
                fs_video = kinematic.framerate;
                tsViconCorr = startViconNSP/sampleRate + (1:durKIN)/fs_video;

                figure('Visible','off','Units','normalized','Position',[0 0.1 1 0.8])

                subplot(3,1,1)
                imagesc(winCenter,frequencies,amp(freq2use,:))
                set(gca,'ydir','normal')
                title(['RAW spectrum - ' labelChan])

                subplot(3,1,2)
                imagesc(winCenter,frequencies,imgaussfilt(ampNorm(freq2use,:),sigma))
                set(gca,'ydir','normal','clim',c_range)
                title('Normalized spectrum')

                subplot(3,1,3)
                plot(tsViconCorr,x,'-r')
                xlabel('Time (s)')
                ylabel('Wrist X (mm)')
                title('Kinematics')

                if isSave
                    fname = sprintf('%s_tr%d_ch%d_%s.png',the_sess,trialNum,ch,labelChan);
                    saveas(gcf,fullfile(save_dir,fname))
                    close(gcf)
                else
                    close(gcf)
                end
            end
        end
    end
end
