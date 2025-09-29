import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from utils import load_signals_from_mat, bandpass_filter

root = "W:/Students/Yasmine/Projet/Connectivity_YD"
monkey = "Lilo"
sessions = ["pre", "post"]
bands = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 12),
    "beta": (13, 30),
    "lowgamma": (30, 60),
    "highgamma": (60, 150),
}
fs = 2000
ds = 10  #downsample factor

save_dir = os.path.join(root, "results", "final_check")
mat_dir = os.path.join(save_dir, "matrices")
diag_dir = os.path.join(save_dir, "diagnostics")
os.makedirs(mat_dir, exist_ok=True)
os.makedirs(diag_dir, exist_ok=True)

def compute_env(sig, fmin, fmax, fs, ds=10, mode="raw"):
    if mode == "CAR":
        sig = sig - sig.mean(axis=0, keepdims=True)

    Xf = bandpass_filter(sig, fmin, fmax, fs)
    env = np.abs(hilbert(Xf, axis=1))
    env = env[:, ::ds]

    if mode == "demean":
        global_mean = env.mean(axis=0, keepdims=True)
        env = env - global_mean

    return env

def compute_corr(env):
    return np.corrcoef(env)

def plot_heatmap(R, title, path):
    plt.figure(figsize=(6, 5))
    plt.imshow(R, vmin=-1, vmax=1, cmap="bwr")
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

def print_stats(name, arr, f):
    stats = np.nanpercentile(arr, [0,1,5,50,95,99,100])
    line = f"{name}: min={stats[0]:.3f}, p1={stats[1]:.3f}, p5={stats[2]:.3f}, median={stats[3]:.3f}, p95={stats[4]:.3f}, p99={stats[5]:.3f}, max={stats[6]:.3f}, mean={np.nanmean(arr):.3f}, std={np.nanstd(arr):.3f}\n"
    f.write(line)
    print(line)

with open(os.path.join(diag_dir, "stats.txt"), "w") as f:
    for sess in sessions:
        arr = load_signals_from_mat(root, monkey, sess)  # (n_trials, n_channels, T)
        print(f"Session {sess}, shape {arr.shape}")

        for band, (fmin, fmax) in bands.items():
            for mode in ["raw", "CAR", "demean"]:
                all_corrs = []

                for tr in range(arr.shape[0]):
                    sig = arr[tr]  
                    env = compute_env(sig, fmin, fmax, fs, ds, mode=mode)
                    R = compute_corr(env)
                    all_corrs.append(R)

                all_corrs = np.array(all_corrs)  #(n_trials, 64, 64)
                mean_R = np.nanmean(all_corrs, axis=0)

                #save matrix
                np.save(os.path.join(mat_dir, f"{sess}_{band}_{mode}.npy"), mean_R)
                plot_heatmap(mean_R, f"{sess} {band} {mode}", os.path.join(mat_dir, f"{sess}_{band}_{mode}.png"))

                #stats
                vals = mean_R[np.triu_indices_from(mean_R, 1)]
                print_stats(f"{sess}-{band}-{mode}", vals, f)

                #extra diag: histogram
                plt.figure()
                plt.hist(vals, bins=50, color="gray", alpha=0.7)
                plt.title(f"Corr distribution {sess} {band} {mode}")
                plt.xlabel("corr")
                plt.ylabel("count")
                plt.tight_layout()
                plt.savefig(os.path.join(diag_dir, f"hist_{sess}_{band}_{mode}.png"), dpi=200)
                plt.close()

print("Results saved in:", save_dir)
