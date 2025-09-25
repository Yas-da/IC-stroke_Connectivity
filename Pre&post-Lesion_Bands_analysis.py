import os, glob, h5py
import numpy as np
import scipy.signal as sp
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Directories
base_dir = r'\\bigdata\Science\Med\Physiology\PTN\Yasmine\IC-stroke_connectivity\data'
out_dir  = r'\\bigdata\Science\Med\Physiology\PTN\Yasmine\IC-stroke_connectivity\ConnectivityResults'
os.makedirs(out_dir, exist_ok=True)

fs = 2000  #Sampling frequency
bands = {
    'delta': (1,4),
    'theta': (4,8),
    'alpha': (8,13),
    'beta': (13,30),
    'gamma': (30,70),
    'highgamma': (70,200)
}
filt_order = 4
date_cutoff = 20250811  #Date of the lesion

def load_trial_mat(filepath):
    with h5py.File(filepath,'r') as f:
        signals = np.array(f['signals'])
        return signals  # nChan x nTime

def bandpass(data, fs, frange, order=4):
    Wn = [frange[0]/(fs/2), frange[1]/(fs/2)]
    b,a = sp.butter(order, Wn, btype='band')
    return sp.filtfilt(b,a,data,axis=-1)

# collect signals
pre_signals = []
post_signals = []

sess_dirs = [d for d in glob.glob(os.path.join(base_dir,'*')) if os.path.isdir(d)]
for sess in sess_dirs:
    sess_name = os.path.basename(sess)
    sess_date = int(sess_name)
    trial_dirs = glob.glob(os.path.join(sess,'trial*','MatFiles','*_allChannels.mat'))
    for trial_file in trial_dirs:
        X = load_trial_mat(trial_file)  # (nChan x nTime)
        X = X[:64]  #cause we don't need ch 65:67
        if sess_date < date_cutoff:
            pre_signals.append(X)
        else:
            post_signals.append(X)

mean_pre = np.mean(np.stack(pre_signals,axis=0),axis=0)   # (64, nTime)
mean_post= np.mean(np.stack(post_signals,axis=0),axis=0)  # (64, nTime)

# extraction des bandes
pdf_path = os.path.join(out_dir,"Mean_signals_bands.pdf")
pdf = PdfPages(pdf_path)

for bname,fr in bands.items():
    Xf_pre = bandpass(mean_pre, fs, fr, order=filt_order)
    Xf_post= bandpass(mean_post, fs, fr, order=filt_order)

    env_pre = np.abs(sp.hilbert(Xf_pre,axis=-1))
    env_post= np.abs(sp.hilbert(Xf_post,axis=-1))

    env_pre_ds = env_pre[:,::10]
    env_post_ds= env_post[:,::10]

    fig,axs = plt.subplots(2,1,figsize=(12,6),sharex=True)
    axs[0].plot(env_pre_ds.T)
    axs[0].set_title(f'{bname} PRE (64 channels)')
    axs[1].plot(env_post_ds.T)
    axs[1].set_title(f'{bname} POST (64 channels)')
    pdf.savefig(fig); plt.close(fig)

pdf.close()
print("PDF saved at:", pdf_path)

np.savez(os.path.join(out_dir,"mean_signals_prepost.npz"),
         pre=mean_pre, post=mean_post, fs=fs, bands=list(bands.keys()))
print("NPZ saved.")
