import os
import numpy as np
import scipy.io as sio
import scipy.signal as signal
import matplotlib.pyplot as plt
from mne.stats import permutation_cluster_test
from matplotlib.backends.backend_pdf import PdfPages

work_root = r"W:\Students\Yasmine\Projet\Connectivity_YD"
data_dir = os.path.join(work_root, "data")
export_dir = os.path.join(work_root, "Pre_Post_lesion_connectivity")
os.makedirs(export_dir, exist_ok=True)

bands = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 12),
    "beta": (12, 30),
    "lowgamma": (30, 55),
    "highgamma": (70, 150),
}

sessions = sorted(os.listdir(data_dir))
pre_sessions = [s for s in sessions if int(s) < 20250522]
post_sessions = [s for s in sessions if int(s) > 20250522]

def bandpass_filter(data, fs, band):
    nyq = fs / 2
    b, a = signal.butter(4, [band[0]/nyq, band[1]/nyq], btype="band")
    return signal.filtfilt(b, a, data)

def compute_connectivity(signals, fs, band):
    nChan = signals.shape[0]
    env = np.zeros_like(signals)
    for i in range(nChan):
        filtered = bandpass_filter(signals[i, :], fs, band)
        env[i, :] = np.abs(signal.hilbert(filtered))
    env_ds = env[:, ::10].T
    R = np.corrcoef(env_ds, rowvar=False)
    return R

def load_all_trials(sess_dir):
    trial_dir = os.path.join(sess_dir, "trial1", "MatFiles")
    mats = [f for f in os.listdir(trial_dir) if f.endswith("_allChannels.mat")]
    signals = []
    for f in mats:
        d = sio.loadmat(os.path.join(trial_dir, f))
        sig = np.array(d["signals"])
        signals.append(sig)
    return signals

def average_connectivity(signals_list, fs, band):
    Rs = []
    for sig in signals_list:
        Rs.append(compute_connectivity(sig, fs, band))
    return np.mean(Rs, axis=0)

fs = 2000
pdf_path = os.path.join(export_dir, "Connectivity_results.pdf")
pdf = PdfPages(pdf_path)

for band_name, band_range in bands.items():
    pre_conn = []
    post_conn = []

    for sess in pre_sessions:
        sigs = load_all_trials(os.path.join(data_dir, sess))
        if sigs:
            pre_conn.append(average_connectivity(sigs, fs, band_range))

    for sess in post_sessions:
        sigs = load_all_trials(os.path.join(data_dir, sess))
        if sigs:
            post_conn.append(average_connectivity(sigs, fs, band_range))

    pre_conn = np.array(pre_conn)
    post_conn = np.array(post_conn)

    nChan = pre_conn.shape[1]
    T_obs, clusters, p_values, _ = permutation_cluster_test(
        [pre_conn, post_conn], n_permutations=1000, tail=0, n_jobs=1
    )

    sig_mask = np.zeros((nChan, nChan))
    for cl, pval in zip(clusters, p_values):
        if pval < 0.05:
            sig_mask[cl] = 1

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    im0 = axes[0].imshow(np.mean(pre_conn, axis=0), vmin=0, vmax=1)
    axes[0].set_title(f"PRE {band_name}")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(np.mean(post_conn, axis=0), vmin=0, vmax=1)
    axes[1].set_title(f"POST {band_name}")
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(sig_mask, cmap="Reds")
    axes[2].set_title(f"Significant {band_name}")
    plt.colorbar(im2, ax=axes[2])

    pdf.savefig(fig)
    plt.close(fig)

    np.savez(
        os.path.join(export_dir, f"Connectivity_{band_name}.npz"),
        pre=pre_conn,
        post=post_conn,
        sig_mask=sig_mask,
    )

pdf.close()
print(f"Results saved to {pdf_path}")
