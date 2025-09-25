import os, glob, h5py
import numpy as np
import scipy.signal as sp
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.animation as animation

# Directories
base_dir = r'\\bigdata\Science\Med\Physiology\PTN\Yasmine\IC-stroke_connectivity\data'
out_dir  = r'\\bigdata\Science\Med\Physiology\PTN\Yasmine\IC-stroke_connectivity\ConnectivityResults'
os.makedirs(out_dir, exist_ok=True)

fs = 2000
bands = {
    'delta': (1,4),
    'theta': (4,8),
    'alpha': (8,13),
    'beta': (13,30),
    'gamma': (30,70),
    'highgamma': (70,200)
}
filt_order = 4
date_cutoff = 20250811

def load_trial_mat(filepath):
    with h5py.File(filepath,'r') as f:
        signals = np.array(f['signals'])
        chan_labels = []
        for ref in f['channel_labels'][0]:
            chan_labels.append(''.join(chr(c[0]) for c in f[ref][:]))
        meta = {}
        for k in f['metadata'].keys():
            try:
                val = f['metadata'][k][()]
                if hasattr(val,'tolist'):
                    val = val.tolist()
                meta[k] = val
            except:
                pass
    return dict(signals=signals, chan_labels=chan_labels, metadata=meta)

def bandpass(data, fs, frange, order=4):
    Wn = [frange[0]/(fs/2), frange[1]/(fs/2)]
    b,a = sp.butter(order, Wn, btype='band')
    return sp.filtfilt(b,a,data,axis=-1)

def custom_cmap():
    return LinearSegmentedColormap.from_list("black_red",
        ["black","dimgray","lightgray","salmon","darkred"])

# Collect signals
pre_trials, post_trials = [], []
sess_dirs = [d for d in glob.glob(os.path.join(base_dir,'*')) if os.path.isdir(d)]
for sess in sess_dirs:
    sess_name = os.path.basename(sess)
    sess_date = int(sess_name)
    trial_dirs = glob.glob(os.path.join(sess,'trial*','MatFiles','*_allChannels.mat'))
    for trial_file in trial_dirs:
        dat = load_trial_mat(trial_file)
        X = dat['signals'][:64]  # keep only 64 channels
        if sess_date < date_cutoff:
            pre_trials.append(X)
        else:
            post_trials.append(X)

pre_mean = np.mean(np.stack(pre_trials,axis=0),axis=0) if len(pre_trials)>0 else None
post_mean= np.mean(np.stack(post_trials,axis=0),axis=0) if len(post_trials)>0 else None

pdf_path = os.path.join(out_dir,"Connectivity_results.pdf")
pdf = PdfPages(pdf_path)

# --- electrode layout (8x8 grid)
coords = np.array([(i//8, i%8) for i in range(64)])

for bname,fr in bands.items():
    pre_env = None
    post_env= None
    if pre_mean is not None:
        Xf = bandpass(pre_mean, fs, fr, order=filt_order)
        pre_env = np.abs(sp.hilbert(Xf,axis=-1))
    if post_mean is not None:
        Xf = bandpass(post_mean, fs, fr, order=filt_order)
        post_env = np.abs(sp.hilbert(Xf,axis=-1))

    if pre_env is not None:
        pre_R = np.corrcoef(pre_env)
    if post_env is not None:
        post_R= np.corrcoef(post_env)

    if pre_env is not None and post_env is not None:
        diff_R = post_R - pre_R

        # --- matrices
        fig,axs = plt.subplots(1,3,figsize=(15,5))
        im0=axs[0].imshow(pre_R,vmin=0,vmax=1,cmap=custom_cmap()); axs[0].set_title(f'{bname} PRE')
        fig.colorbar(im0,ax=axs[0])
        im1=axs[1].imshow(post_R,vmin=0,vmax=1,cmap=custom_cmap()); axs[1].set_title(f'{bname} POST')
        fig.colorbar(im1,ax=axs[1])
        im2=axs[2].imshow(diff_R,cmap=custom_cmap(),vmin=-0.5,vmax=0.5); axs[2].set_title('POST-PRE')
        fig.colorbar(im2,ax=axs[2])
        pdf.savefig(fig); plt.show(); plt.close(fig)

        # --- node strength and significance mapping
        strength_pre  = np.sum(pre_R,axis=1)
        strength_post = np.sum(post_R,axis=1)
        diff_strength = strength_post - strength_pre

        tvals, pvals = ttest_ind(pre_R, post_R, axis=1, equal_var=False)
        sig_mask = pvals < 0.05

        fig,ax = plt.subplots(figsize=(6,6))
        sc = ax.scatter(coords[:,1],-coords[:,0],c=diff_strength,
                        cmap=custom_cmap(),s=200,edgecolor='k')
        for i,(x,y) in enumerate(coords):
            if sig_mask[i]:
                ax.scatter(y,-x,c='none',s=400,edgecolor='yellow',linewidth=2)
        ax.set_title(f'Significant node strength changes ({bname})')
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(sc,ax=ax)
        pdf.savefig(fig); plt.show(); plt.close(fig)

        # --- temporal dynamics animation
        win_sec = 0.2
        step_sec= 0.05
        ds_factor=10
        fs_ds = fs//ds_factor
        win_ds = int(win_sec*fs_ds)
        step_ds= int(step_sec*fs_ds)

        pre_env_ds = pre_env[:,::ds_factor]
        post_env_ds= post_env[:,::ds_factor]

        n_windows = max(1,(pre_env_ds.shape[1]-win_ds)//step_ds+1)
        pre_R_windows = np.zeros((n_windows,64,64))
        for w in range(n_windows):
            seg = pre_env_ds[:,w*step_ds:w*step_ds+win_ds]
            pre_R_windows[w] = np.corrcoef(seg)

        n_windows = max(1,(post_env_ds.shape[1]-win_ds)//step_ds+1)
        post_R_windows = np.zeros((n_windows,64,64))
        for w in range(n_windows):
            seg = post_env_ds[:,w*step_ds:w*step_ds+win_ds]
            post_R_windows[w] = np.corrcoef(seg)

        fig,ax = plt.subplots(figsize=(6,5))
        ims = []
        for i in range(min(len(pre_R_windows),len(post_R_windows))):
            im = ax.imshow(post_R_windows[i]-pre_R_windows[i],animated=True,
                           cmap=custom_cmap(),vmin=-0.5,vmax=0.5)
            ims.append([im])
        ani = animation.ArtistAnimation(fig,ims,interval=300,blit=True,repeat_delay=1000)
        ani.save(os.path.join(out_dir,f'Connectivity_evolution_{bname}.mp4'))
        plt.show(); plt.close(fig)

pdf.close()
print("PDF saved at:", pdf_path)
