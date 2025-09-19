close all
clear all
clc

work_dir = '\\bigdata\Science\Med\Physiology\PTN\Students\Yasmine\IC_stroke_decoder\ECoG_alignment';
addpath(fullfile(work_dir,'function'));  % others
D_i = NR_Palette;
monkey_dir = 'Lilo';
data_dir = '\\bigdata\Science\Med\Physiology\PTN\Data\IC_Stroke\LILO_BSI';
the_sess = '20250522';

input_dir = fullfile(data_dir,the_sess,[the_sess '_processed']);
save_dir = fullfile(input_dir,'Figures');
mkdir(save_dir);

isSave = 1;  % mettre à 1 pour sauvegarder figures et .mat

%% load 1st trial blackrock, emg, vicon 
trial2show = 1;
load([input_dir filesep monkey_dir '_' the_sess '_tr' num2str(trial2show) '_ecog.mat'],...
    'data','comment','sampleRate','startViconNSP','endViconNSP');
load([input_dir filesep monkey_dir '_' the_sess '_tr' num2str(trial2show) '_vicon.mat'],...
    'kinematic','analog');

sigma = 6;
freq_range = [0 200];
time_range = [110 140];
array2show = [2 1];
c_range = [0 2];

num_channels = 64; % nombre total de channels

for ch = 1:num_channels
    figure('Units','normalized','Position',[0 0.1 1 0.8])
    
    channel_data_struct = struct(); % pour sauvegarder les données dans le .mat

    for kk = 1:2
        ar = array2show(kk);
        dataChan = double(data(ar).Data(ch,:));
        labelChan = data(ar).Label{ch};
        nrSample = length(dataChan);

        step = sampleRate/20;
        fftWinSize = sampleRate/4;
        winFunction = hamming(fftWinSize);
        nfft = sampleRate;

        [s,f,t] = spectrogram(dataChan,winFunction,fftWinSize-step,nfft,sampleRate);
        amp = abs(s);
        ampNorm = amp./repmat(median(amp')',1,size(amp,2));
        freq2use = find(f>=freq_range(1) & f<=freq_range(2));
        frequencies = f(freq2use);
        winCenter = t;

        % coordonnées VICON
        ind_WRB = find(ismember(kinematic.labels,'WRB')==1);
        x = kinematic.x(:,ind_WRB);
        y = kinematic.y(:,ind_WRB);
        z = kinematic.z(:,ind_WRB);
        durKIN = length(x);
        fs_video = kinematic.framerate;
        tsViconCorr = startViconNSP/sampleRate + [1:durKIN]/fs_video;

        % --- Spectrogramme brut ---
        subplot(3,2,kk)
        dataAmp = amp(freq2use,:);
        hold on
        imagesc(winCenter,frequencies,dataAmp)
        colormap(D_i)
        plot(tsViconCorr,(x-200)/200*50-50,'-r','linewidth',2)
        fill([time_range(1) time_range(2) time_range(2) time_range(1)],[-1 -1 0 0]*50,'g',...
            'EdgeColor','none','FaceAlpha',0.3)
        set(gca,'xlim',[tsViconCorr(1) tsViconCorr(end)],'ylim',[-50 freq_range(2)],...
            'clim',[min(dataAmp(:)),prctile(dataAmp(:),98)])
        set(gca,'ydir','normal')
        title({['ch' num2str(ch) ':' labelChan],'RAW spectrum aligned with kinematic'})
        xlabel('Time /s')
        ylabel('Frequency / Hz')

        % --- Spectrogramme normalisé ---
        subplot(3,2,kk+2)
        dataAmpNorm = ampNorm(freq2use,:);
        dataAmpNorm = imgaussfilt(dataAmpNorm,sigma);
        hold on
        imagesc(winCenter,frequencies,dataAmpNorm)
        colormap(D_i)
        plot(tsViconCorr,(x-200)/200*50-50,'-r','linewidth',2)
        set(gca,'xlim',time_range,'ylim',[-50 freq_range(2)],'clim',c_range)
        set(gca,'ydir','normal')
        title('Normalized spectrum aligned with kinematic')
        xlabel('Time /s')
        ylabel('Frequency / Hz')

        % --- VICON wrist-x ---
        subplot(3,2,kk+4)
        hold on
        plot(tsViconCorr,x,'-r')
        set(gca,'xlim',time_range,'ylim',[200 400])
        title('Wrist-x VICON results aligned')
        xlabel('Time /s')
        ylabel('wrist x position /mm')

        % --- sauvegarde des données pour .mat ---
        channel_data_struct(kk).rawAmp = dataAmp;
        channel_data_struct(kk).normAmp = dataAmpNorm;
        channel_data_struct(kk).time = winCenter;
        channel_data_struct(kk).freq = frequencies;
        channel_data_struct(kk).vicon_x = x;
        channel_data_struct(kk).vicon_y = y;
        channel_data_struct(kk).vicon_z = z;
    end

    if isSave
        saveas(gcf,[save_dir filesep 'Trial' num2str(trial2show) '_Ch' num2str(ch) '.fig'])
        saveas(gcf,[save_dir filesep 'Trial' num2str(trial2show) '_Ch' num2str(ch) '.png'])
        save(fullfile(save_dir,['Trial' num2str(trial2show) '_Ch' num2str(ch) '.mat']),...
            'channel_data_struct','labelChan')
    end

    close(gcf)
end
