import os, glob, h5py
import numpy as np
import scipy.signal as sp
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

#Directories
base_dir = r'\\bigdata\Science\Med\Physiology\PTN\Yasmine\IC-stroke_connectivity\data'
out_dir  = r'\\bigdata\Science\Med\Physiology\PTN\Yasmine\IC-stroke_connectivity\ConnectivityResults'
os.makedirs(out_dir, exist_ok=True)

fs = 2000  #sampling frequency
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

grid_side = 8
coords = [(i,j) for i in range(grid_side) for j in range(grid_side)]

def load_trial_mat(filepath):
    with h5py.File(filepath,'r') as f:
        signals = np.array(f['signals'])
        time_ecog = np.array(f['time_ecog']).squeeze()
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
        kin_x = np.array(f['kin_x']) if 'kin_x' in f else None
        time_vicon = np.array(f['time_vicon']) if 'time_vicon' in f else None
    return dict(signals=signals, time=time_ecog, chan_labels=chan_labels,
                metadata=meta, kin_x=kin_x, time_vicon=time_vicon)

def bandpass(data, fs, frange, order=4):
    Wn = [frange[0]/(fs/2), frange[1]/(fs/2)]
    b,a = sp.butter(order, Wn, btype='band')
    return sp.filtfilt(b,a,data,axis=-1)

def fisher_z(r):
    return np.arctanh(np.clip(r,-0.999999,0.999999))

def inverse_fisher(z):
    return np.tanh(z)

def cluster_permutation(Z_pre, Z_post, n_perm=1000, alpha=0.05):
    nEdges = Z_pre.shape[0]
    obs_diff = Z_post.mean(1) - Z_pre.mean(1)
    tvals, pvals = ttest_ind(Z_post.T, Z_pre.T, axis=0)
    sig_init = pvals < alpha
    cluster_stat = np.abs(obs_diff) * sig_init
    obs_cluster_mass = cluster_stat.sum()
    all_data = np.hstack([Z_pre, Z_post])
    nA = Z_pre.shape[1]
    null_dist = []
    for _ in range(n_perm):
        perm_idx = np.random.permutation(all_data.shape[1])
        A = all_data[:,perm_idx[:nA]]
        B = all_data[:,perm_idx[nA:]]
        diff = B.mean(1) - A.mean(1)
        tvals, pvals = ttest_ind(B.T,A.T,axis=0)
        sig = pvals<alpha
        null_dist.append(np.abs(diff*sig).sum())
    thresh = np.percentile(null_dist, 100*(1-alpha))
    sig_mask = obs_cluster_mass > thresh
    return sig_init*sig_mask, pvals

all_results = {b:[] for b in bands}

sess_dirs = [d for d in glob.glob(os.path.join(base_dir,'*')) if os.path.isdir(d)]
for sess in sess_dirs:
    sess_name = os.path.basename(sess)
    sess_date = int(sess_name)
    trial_dirs = glob.glob(os.path.join(sess,'trial*','MatFiles','*_allChannels.mat'))
    for trial_file in trial_dirs:
        dat = load_trial_mat(trial_file)
        X = dat['signals']
        nChan,nTime = X.shape
        condition = 'pre' if sess_date<date_cutoff else 'post'
        for bname,fr in bands.items():
            Xf = bandpass(X, fs, fr, order=filt_order)
            env = np.abs(sp.hilbert(Xf,axis=-1))
            env_ds = env[:,::10]
            R = np.corrcoef(env_ds)
            all_results[bname].append(dict(R=R, condition=condition,
                                           session=sess_name, trial=dat['metadata']['trial']))

pdf_path = os.path.join(out_dir,"Connectivity_results.pdf")
pdf = PdfPages(pdf_path)

for bname in bands:
    Rs_pre = [r['R'] for r in all_results[bname] if r['condition']=='pre']
    Rs_post= [r['R'] for r in all_results[bname] if r['condition']=='post']
    Rm_pre = np.mean(Rs_pre,axis=0)
    Rm_post= np.mean(Rs_post,axis=0)
    iu = np.triu_indices(Rm_pre.shape[0],1)
    ZA = np.array([fisher_z(R[iu]) for R in Rs_pre]).T
    ZB = np.array([fisher_z(R[iu]) for R in Rs_post]).T
    sig_mask, pvals = cluster_permutation(ZA,ZB)
    sigMat = np.zeros_like(Rm_pre,dtype=bool)
    sigMat[iu] = sig_mask
    sigMat = sigMat|sigMat.T
    fig,axs = plt.subplots(2,2,figsize=(12,10))
    im0=axs[0,0].imshow(Rm_pre,vmin=0,vmax=1,cmap='viridis'); axs[0,0].set_title(f'{bname} PRE')
    fig.colorbar(im0,ax=axs[0,0])
    im1=axs[0,1].imshow(Rm_post,vmin=0,vmax=1,cmap='viridis'); axs[0,1].set_title(f'{bname} POST')
    fig.colorbar(im1,ax=axs[0,1])
    im2=axs[1,0].imshow(Rm_post-Rm_pre,cmap='bwr',vmin=-0.3,vmax=0.3); axs[1,0].set_title('POST-PRE')
    fig.colorbar(im2,ax=axs[1,0])
    im3=axs[1,1].imshow(sigMat,cmap='gray'); axs[1,1].set_title('Significant edges')
    pdf.savefig(fig); plt.close(fig)

pdf.close()
print("PDF saved at:", pdf_path)

features = []
labels = []
for bname in bands:
    for r in all_results[bname]:
        R = r['R']
        node_strength = R.sum(1)
        bandpower = np.diag(R)
        feat = np.concatenate([node_strength, bandpower])
        features.append(feat)
        labels.append(0 if r['condition']=='pre' else 1)
features = np.array(features)
labels = np.array(labels)
np.savez(os.path.join(out_dir,"features_for_decoder.npz"),
         X=features,y=labels,bandnames=list(bands.keys()))
print("NPZ features saved.")
