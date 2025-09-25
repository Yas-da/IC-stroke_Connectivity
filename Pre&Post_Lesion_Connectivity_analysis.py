import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from scipy.signal import butter, filtfilt
import matplotlib.image as mpimg

n_trials, n_channels, n_timepoints, fs = 20, 67, 2000, 1000
data_pre = np.random.randn(n_trials, n_channels, n_timepoints)
data_post = np.random.randn(n_trials, n_channels, n_timepoints)

bad_channels = [64, 65, 66]
data_pre = np.delete(data_pre, bad_channels, axis=1)
data_post = np.delete(data_post, bad_channels, axis=1)
n_channels = data_pre.shape[1]

bands = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 12),
    "beta": (12, 30),
    "gamma": (30, 70),
    "high_gamma": (70, 150)
}

def bandpass_filter(data, low, high, fs):
    b, a = butter(4, [low/(fs/2), high/(fs/2)], btype="band")
    return filtfilt(b, a, data)

def compute_band_data(data, bands):
    band_data = {}
    for band, (low, high) in bands.items():
        filtered = np.array([
            [bandpass_filter(trial[ch], low, high, fs) for ch in range(data.shape[1])]
            for trial in data
        ])
        band_data[band] = filtered
    return band_data

band_data_pre = compute_band_data(data_pre, bands)
band_data_post = compute_band_data(data_post, bands)

def connectivity_matrix(data):
    return np.corrcoef(data)

win_size = 200
step = 100

def sliding_windows(data, win_size, step):
    n_channels, n_time = data.shape
    windows = []
    for start in range(0, n_time - win_size, step):
        seg = data[:, start:start+win_size]
        windows.append(seg)
    return windows

cmap = "inferno"

for band in bands.keys():
    mean_pre = np.mean(band_data_pre[band], axis=0)
    mean_post = np.mean(band_data_post[band], axis=0)

    windows_pre = sliding_windows(mean_pre, win_size, step)
    windows_post = sliding_windows(mean_post, win_size, step)

    conn_matrices_pre = [connectivity_matrix(w) for w in windows_pre]
    conn_matrices_post = [connectivity_matrix(w) for w in windows_post]

    avg_conn_pre = np.mean(conn_matrices_pre, axis=0)
    avg_conn_post = np.mean(conn_matrices_post, axis=0)
    diff_conn = avg_conn_post - avg_conn_pre

    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.imshow(avg_conn_pre, cmap=cmap, vmin=-1, vmax=1)
    plt.title(f"{band} - Pre")
    plt.colorbar()

    plt.subplot(1,3,2)
    plt.imshow(avg_conn_post, cmap=cmap, vmin=-1, vmax=1)
    plt.title(f"{band} - Post")
    plt.colorbar()

    plt.subplot(1,3,3)
    plt.imshow(diff_conn, cmap=cmap)
    plt.title(f"{band} - Difference")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(f"connectivity_{band}.png")
    plt.show()

    all_conn_pre = np.array(conn_matrices_pre)
    all_conn_post = np.array(conn_matrices_post)

    pvals = np.ones_like(avg_conn_pre)
    sig_diff = np.zeros_like(avg_conn_pre)
    for i in range(avg_conn_pre.shape[0]):
        for j in range(avg_conn_pre.shape[1]):
            _, p = ttest_ind(all_conn_pre[:,i,j], all_conn_post[:,i,j], equal_var=False)
            pvals[i,j] = p
            if p < 0.05:
                sig_diff[i,j] = diff_conn[i,j]

    plt.figure()
    plt.imshow(sig_diff, cmap=cmap)
    plt.title(f"{band} - Significant differences (p<0.05)")
    plt.colorbar()
    plt.savefig(f"significant_diff_{band}.png")
    plt.show()

    layout = mpimg.imread("lilo_electrode_layout_sulci.jpg")
    plt.figure(figsize=(6,6))
    plt.imshow(layout)
    sig_pairs = np.argwhere(pvals < 0.05)
    for (i,j) in sig_pairs:
        if i < j:
            x, y = np.random.randint(0, layout.shape[1]), np.random.randint(0, layout.shape[0])
            plt.plot(x, y, 'o', color="darkred")
    plt.title(f"{band} - Significant connectivity changes")
    plt.savefig(f"electrode_mapping_{band}.png")
    plt.show()

    mean_conn_pre = [np.mean(m) for m in conn_matrices_pre]
    mean_conn_post = [np.mean(m) for m in conn_matrices_post]
    plt.figure()
    plt.plot(mean_conn_pre, label="Pre", color="black")
    plt.plot(mean_conn_post, label="Post", color="darkred")
    plt.xlabel("Window index")
    plt.ylabel("Mean connectivity")
    plt.legend()
    plt.title(f"{band} - Temporal dynamics")
    plt.savefig(f"temporal_{band}.png")
    plt.show()
