import os
import glob
import h5py
import numpy as np
from scipy.signal import butter, filtfilt, hilbert
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

fs = 2000
lesion_date = 20250811
channels_keep = 64
base_dir = r"\\bigdata\Science\Med\Physiology\PTN\Yasmine\IC-stroke_connectivity\data"
save_dir = r"\\bigdata\Science\Med\Physiology\PTN\Yasmine\IC-stroke_connectivity\ConnectivityResults"
os.makedirs(save_dir, exist_ok=True)

bands = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 12),
    "beta": (13, 30),
    "gamma": (30, 70),
    "highgamma": (70, 150)
}

#Palette : change if necessary
cmap = LinearSegmentedColormap.from_list("gray_red", ["#2c2c2c", "#bfbfbf", "#ffb3b3", "#800000"])

def load_signals_from_mat(path):
    with h5py.File(path, "r") as f:
        # expecting variable 'signals' saved as channels x time
        if "signals" in f:
            signals = np.array(f["signals"])
        elif "data" in f:  # fallback
            signals = np.array(f["data"])
        else:
            raise KeyError(f"'signals' not found in {path}")
    return signals[:channels_keep, :]

def bandpass_filter(data, low, high, fs, order=4):
    b, a = butter(order, [low / (fs / 2), high / (fs / 2)], btype="band")
    return filtfilt(b, a, data, axis=-1)

def fisher_z(mat):
    return np.arctanh(np.clip(mat, -0.999999, 0.999999))

def inverse_fisher(z):
    return np.tanh(z)

def benjamini_hochberg(pvals, alpha=0.05):
    p = np.array(pvals)
    m = p.size
    order = np.argsort(p)
    sorted_p = p[order]
    thresh = (np.arange(1, m+1) / m) * alpha
    below = sorted_p <= thresh
    if not np.any(below):
        mask = np.zeros(m, dtype=bool)
    else:
        max_idx = np.max(np.where(below)[0])
        crit_p = thresh[max_idx]
        mask = p <= crit_p
    return mask

#Pre / post
pre_trials = []
post_trials = []

sess_dirs = [d for d in glob.glob(os.path.join(base_dir, "*")) if os.path.isdir(d)]
for sess in sess_dirs:
    sess_name = os.path.basename(sess)
    try:
        sess_date = int(sess_name)
    except:
        continue
    trial_files = glob.glob(os.path.join(sess, "trial*", "MatFiles", "*_allChannels.mat"))
    for tf in trial_files:
        try:
            S = load_signals_from_mat(tf)
        except Exception:
            continue
        if S.shape[0] < channels_keep:
            continue
        if sess_date < lesion_date:
            pre_trials.append(S)
        else:
            post_trials.append(S)

if len(pre_trials) == 0 or len(post_trials) == 0:
    raise RuntimeError("No pre or post trials found — check base_dir and lesion_date.")

#Length alignment
all_trials = pre_trials + post_trials
min_len = min(t.shape[1] for t in all_trials)
pre_arr = np.stack([t[:, :min_len] for t in pre_trials], axis=0)   # (n_pre, 64, T)
post_arr = np.stack([t[:, :min_len] for t in post_trials], axis=0) # (n_post,64, T)

np.save(os.path.join(save_dir, "pre_data.npy"), pre_arr)
np.save(os.path.join(save_dir, "post_data.npy"), post_arr)

#mean
mean_pre = np.mean(pre_arr, axis=0)   # (64, T)
mean_post = np.mean(post_arr, axis=0) # (64, T)

right_idx_range = range(32, 64)
var_post = np.var(post_arr, axis=2).mean(axis=0)  
example_ch = int(np.argmax(var_post[list(right_idx_range)])) + 32
if example_ch < 32 or example_ch >= 64:
    example_ch = 32
    
plt.figure(figsize=(16, 10))
ax1 = plt.subplot(2,1,1)
for ch in range(channels_keep):
    ax1.plot(np.arange(mean_pre.shape[1]) / fs, mean_pre[ch], color="#333333", alpha=0.6, linewidth=0.8)
ax1.set_title("Pre-lesion: 64 channel means")
ax1.set_ylabel("Amplitude (a.u.)")

ax2 = plt.subplot(2,1,2, sharex=ax1)
for ch in range(channels_keep):
    ax2.plot(np.arange(mean_post.shape[1]) / fs, mean_post[ch], color="#800000", alpha=0.6, linewidth=0.8)
ax2.set_title("Post-lesion: 64 channel means")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Amplitude (a.u.)")

outfn = os.path.join(save_dir, "64channels_mean_pre_post.png")
plt.tight_layout()
plt.savefig(outfn, dpi=300)
plt.show()

sig_pre = mean_pre[example_ch]
sig_post = mean_post[example_ch]

fig, axes = plt.subplots(2, len(bands)+1, figsize=(4*(len(bands)+1), 8), sharex='col')
axes[0,0].plot(np.arange(sig_pre.size)/fs, sig_pre, color='k'); axes[0,0].set_title(f'Ch {example_ch+1} pre (raw)')
axes[1,0].plot(np.arange(sig_post.size)/fs, sig_post, color='k'); axes[1,0].set_title(f'Ch {example_ch+1} post (raw)')

for col, (bname, (fmin, fmax)) in enumerate(bands.items(), start=1):
    filt_pre = bandpass_filter(sig_pre, fmin, fmax, fs)
    filt_post = bandpass_filter(sig_post, fmin, fmax, fs)
    axes[0,col].plot(np.arange(filt_pre.size)/fs, filt_pre, color='#333333')
    axes[0,col].set_title(f'Pre - {bname} ({fmin}-{fmax} Hz)')
    axes[1,col].plot(np.arange(filt_post.size)/fs, filt_post, color='#800000')
    axes[1,col].set_title(f'Post - {bname} ({fmin}-{fmax} Hz)')

for ax in axes.flat:
    ax.set_xlabel('Time (s)')
plt.tight_layout()
outfn = os.path.join(save_dir, f"example_ch{example_ch+1}_decomposition.png")
plt.savefig(outfn, dpi=300)
plt.show()

results = {}
for bname, (fmin, fmax) in bands.items():
    pre_env_trials = []
    for tr in range(pre_arr.shape[0]):
        X = pre_arr[tr]  # (64, T)
        Xf = bandpass_filter(X, fmin, fmax, fs)
        env = np.abs(hilbert(Xf, axis=1))
        pre_env_trials.append(env)
    post_env_trials = []
    for tr in range(post_arr.shape[0]):
        X = post_arr[tr]
        Xf = bandpass_filter(X, fmin, fmax, fs)
        env = np.abs(hilbert(Xf, axis=1))
        post_env_trials.append(env)

    ds = 10
    pre_env_trials = [e[:, ::ds] for e in pre_env_trials]
    post_env_trials = [e[:, ::ds] for e in post_env_trials]

    # compute per-trial corr matrices (channels x channels)
    pre_Rs = np.array([np.corrcoef(e) for e in pre_env_trials])    # (n_pre, 64,64)
    post_Rs = np.array([np.corrcoef(e) for e in post_env_trials])  # (n_post,64,64)

    # average matrices
    R_pre_mean = pre_Rs.mean(axis=0)
    R_post_mean = post_Rs.mean(axis=0)
    R_diff = R_post_mean - R_pre_mean

    #stats : Fisher z then t-test
    iu = np.triu_indices(channels_keep, 1)
    A = fisher_z(pre_Rs[:, iu[0], iu[1]])   # (n_pre, E)
    B = fisher_z(post_Rs[:, iu[0], iu[1]])  # (n_post, E)
    tvals = np.zeros(A.shape[1])
    pvals = np.ones(A.shape[1])
    for e in range(A.shape[1]):
        _, p = ttest_ind(A[:, e], B[:, e], equal_var=False)
        pvals[e] = p
    sig_flat = benjamini_hochberg(pvals, alpha=0.05)
    sig_mask = np.zeros((channels_keep, channels_keep), dtype=bool)
    sig_mask[iu] = sig_flat
    sig_mask = sig_mask | sig_mask.T

    node_pre = pre_Rs.sum(axis=2)   # (n_pre, 64)
    node_post = post_Rs.sum(axis=2) # (n_post, 64)
    p_node = np.ones(channels_keep)
    for ch in range(channels_keep):
        _, p = ttest_ind(node_pre[:, ch], node_post[:, ch], equal_var=False)
        p_node[ch] = p
    sig_node_mask = benjamini_hochberg(p_node, alpha=0.05)

    #save results
    results[bname] = {
        "R_pre_mean": R_pre_mean,
        "R_post_mean": R_post_mean,
        "R_diff": R_diff,
        "sig_edges_mask": sig_mask,
        "pvals_edges_flat": pvals,
        "sig_nodes_mask": sig_node_mask,
        "node_pre": node_pre,
        "node_post": node_post
    }

    #save images
    plt.figure(figsize=(12,4))
    ax = plt.subplot(1,3,1); im = ax.imshow(R_pre_mean, cmap=cmap, vmin=-1, vmax=1); ax.set_title(f"{bname} PRE"); plt.colorbar(im, ax=ax)
    ax = plt.subplot(1,3,2); im = ax.imshow(R_post_mean, cmap=cmap, vmin=-1, vmax=1); ax.set_title(f"{bname} POST"); plt.colorbar(im, ax=ax)
    ax = plt.subplot(1,3,3); im = ax.imshow(R_diff, cmap=cmap, vmin=-0.5, vmax=0.5); ax.set_title(f"{bname} DIFF"); plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"connectivity_{bname}.png"), dpi=300)
    plt.show()

#Mapping
xs = np.linspace(-4.5, 4.5, 8)  
ys = np.linspace(-14, 7, 8)     
coords = []
for r in range(8):
    for c in range(8):
        coords.append((xs[c], ys[r]))
coords = np.array(coords)  # shape (64,2)

for bname, out in results.items():
    node_pre_mean = out["node_pre"].mean(axis=0)
    node_post_mean = out["node_post"].mean(axis=0)
    node_diff = node_post_mean - node_pre_mean
    sig_nodes = out["sig_nodes_mask"]

    plt.figure(figsize=(6,6))
    sc = plt.scatter(coords[:,0], coords[:,1], c=node_diff, cmap=cmap, s=220, edgecolors='k', vmin=-np.max(np.abs(node_diff)), vmax=np.max(np.abs(node_diff)))
    for idx in np.where(sig_nodes)[0]:
        plt.scatter(coords[idx,0], coords[idx,1], facecolors='none', edgecolors='yellow', s=600, linewidths=2)
    plt.title(f"{bname} node-strength (post - pre); yellow = significant")
    plt.colorbar(sc, label='node-strength difference')
    plt.axis('equal')
    plt.gca().invert_yaxis()
    plt.savefig(os.path.join(save_dir, f"topography_node_diff_{bname}.png"), dpi=300)
    plt.show()

# ---------- HEMISPHERE SUMMARY ----------
left_idx = np.arange(0, 32)
right_idx = np.arange(32, 64)
hem_results = {}
for bname, out in results.items():
    left_strength_pre = out["node_pre"][:, left_idx].mean(axis=1)
    left_strength_post = out["node_post"][:, left_idx].mean(axis=1)
    right_strength_pre = out["node_pre"][:, right_idx].mean(axis=1)
    right_strength_post = out["node_post"][:, right_idx].mean(axis=1)

    t_left_p = ttest_ind(left_strength_pre, left_strength_post, equal_var=False)[1]
    t_right_p = ttest_ind(right_strength_pre, right_strength_post, equal_var=False)[1]

    hem_results[bname] = {
        "left_pre_mean": left_strength_pre.mean(),
        "left_post_mean": left_strength_post.mean(),
        "right_pre_mean": right_strength_pre.mean(),
        "right_post_mean": right_strength_post.mean(),
        "p_left": t_left_p,
        "p_right": t_right_p
    }

# plot hemisphere bar summary across bands
bands_list = list(results.keys())
left_pre = [hem_results[b]["left_pre_mean"] for b in bands_list]
left_post = [hem_results[b]["left_post_mean"] for b in bands_list]
right_pre = [hem_results[b]["right_pre_mean"] for b in bands_list]
right_post = [hem_results[b]["right_post_mean"] for b in bands_list]

x = np.arange(len(bands_list))
width = 0.2
plt.figure(figsize=(12,6))
plt.bar(x - 1.5*width, left_pre, width, label='Left Pre', color='#888888')
plt.bar(x - 0.5*width, left_post, width, label='Left Post', color='#800000')
plt.bar(x + 0.5*width, right_pre, width, label='Right Pre', color='#bbbbbb')
plt.bar(x + 1.5*width, right_post, width, label='Right Post', color='#ff6666')
plt.xticks(x, bands_list)
plt.ylabel('Mean node-strength')
plt.title('Hemispheric comparison by band')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "hemisphere_summary_by_band.png"), dpi=300)
plt.show()
-
np.savez(os.path.join(save_dir, "connectivity_presentation_summary.npz"),
         bands=bands, results=results, hem_results=hem_results, example_ch=example_ch)

print("All outputs saved to:", save_dir)
