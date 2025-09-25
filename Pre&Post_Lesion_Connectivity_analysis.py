import os
import glob
import h5py
import numpy as np
from scipy.signal import butter, filtfilt, hilbert
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
from matplotlib import colors, animation

base_dir = r'\\bigdata\Science\Med\Physiology\PTN\Yasmine\IC-stroke_connectivity\data'
out_dir = r'\\bigdata\Science\Med\Physiology\PTN\Yasmine\IC-stroke_connectivity\ConnectivityResults'
os.makedirs(out_dir, exist_ok=True)

fs = 2000
date_cutoff = 20250811

bad_channel_indices = [64, 65, 66]

bands = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 12),
    "beta": (12, 30),
    "gamma": (30, 70),
    "high_gamma": (70, 150)
}

cmap = colors.LinearSegmentedColormap.from_list("gray_red", ["#2c2c2c", "#bfbfbf", "#ffb3b3", "#800000"])

def load_trial_mat(filepath):
    with h5py.File(filepath, "r") as f:
        signals = np.array(f["signals"])
    return signals

def bandpass(data, low, high, fs, order=4):
    b, a = butter(order, [low/(fs/2), high/(fs/2)], btype="band")
    return filtfilt(b, a, data, axis=-1)

sess_dirs = [d for d in glob.glob(os.path.join(base_dir, "*")) if os.path.isdir(d)]
pre_trials = []
post_trials = []

for sess in sess_dirs:
    sess_name = os.path.basename(sess)
    try:
        sess_date = int(sess_name)
    except:
        continue
    trial_files = glob.glob(os.path.join(sess, "trial*", "MatFiles", "*_allChannels.mat"))
    for tf in trial_files:
        try:
            sigs = load_trial_mat(tf)
        except Exception:
            continue
        if sigs.shape[0] < 64:
            continue
        X64 = sigs[:64, :]
        if sess_date < date_cutoff:
            pre_trials.append(X64)
        else:
            post_trials.append(X64)

if len(pre_trials) == 0 or len(post_trials) == 0:
    raise RuntimeError("Not enough pre or post trials found. Check base_dir and date_cutoff.")

min_len = min([t.shape[1] for t in pre_trials + post_trials])
pre_trials = [t[:, :min_len] for t in pre_trials]
post_trials = [t[:, :min_len] for t in post_trials]
pre_arr = np.stack(pre_trials, axis=0)
post_arr = np.stack(post_trials, axis=0)
n_pre, n_chan, n_time = pre_arr.shape
n_post = post_arr.shape[0]

ds_factor = 10
fs_ds = fs // ds_factor
win_sec = 0.2
step_sec = 0.1
win_ds = int(win_sec * fs_ds)
step_ds = int(step_sec * fs_ds)

def sliding_windows_env(env, win, step):
    n_chan, n_t = env.shape
    windows = []
    starts = list(range(0, n_t - win + 1, step))
    for s in starts:
        windows.append(env[:, s:s+win])
    return np.stack(windows, axis=0)

def corr_matrix_from_env(env):
    return np.corrcoef(env)

def benjamini_hochberg(pvals_flat, alpha=0.05):
    m = len(pvals_flat)
    sorted_idx = np.argsort(pvals_flat)
    sorted_p = pvals_flat[sorted_idx]
    thresh = np.arange(1, m+1) * alpha / m
    below = sorted_p <= thresh
    if not np.any(below):
        return np.zeros(m, dtype=bool)
    max_i = np.max(np.where(below)[0])
    crit = thresh[max_i]
    return pvals_flat <= crit

results_summary = {}

for band_name, (low, high) in bands.items():
    pre_env_trials = []
    post_env_trials = []
    for tr in range(n_pre):
        X = pre_arr[tr]
        Xf = bandpass(X, low, high, fs)
        env = np.abs(hilbert(Xf, axis=-1))[:, ::ds_factor]
        pre_env_trials.append(env)
    for tr in range(n_post):
        X = post_arr[tr]
        Xf = bandpass(X, low, high, fs)
        env = np.abs(hilbert(Xf, axis=-1))[:, ::ds_factor]
        post_env_trials.append(env)
    pre_env_trials = np.array(pre_env_trials)
    post_env_trials = np.array(post_env_trials)

    n_windows = (pre_env_trials.shape[2] - win_ds) // step_ds + 1
    pre_R_windows = np.zeros((n_pre, n_windows, n_chan, n_chan))
    post_R_windows = np.zeros((n_post, n_windows, n_chan, n_chan))

    for t in range(n_pre):
        win_stack = sliding_windows_env(pre_env_trials[t], win_ds, step_ds)
        for w in range(win_stack.shape[0]):
            pre_R_windows[t, w] = corr_matrix_from_env(win_stack[w])
    for t in range(n_post):
        win_stack = sliding_windows_env(post_env_trials[t], win_ds, step_ds)
        for w in range(win_stack.shape[0]):
            post_R_windows[t, w] = corr_matrix_from_env(win_stack[w])

    mean_Rpre_per_trial = pre_R_windows.mean(axis=1)
    mean_Rpost_per_trial = post_R_windows.mean(axis=1)
    avg_conn_pre = mean_Rpre_per_trial.mean(axis=0)
    avg_conn_post = mean_Rpost_per_trial.mean(axis=0)
    diff_conn = avg_conn_post - avg_conn_pre

    np.save(os.path.join(out_dir, f'R_mean_{band_name}_pre.npy'), avg_conn_pre)
    np.save(os.path.join(out_dir, f'R_mean_{band_name}_post.npy'), avg_conn_post)
    np.save(os.path.join(out_dir, f'R_diff_{band_name}.npy'), diff_conn)

    pvals = np.ones((n_chan, n_chan))
    for i in range(n_chan):
        for j in range(n_chan):
            a = mean_Rpre_per_trial[:, i, j]
            b = mean_Rpost_per_trial[:, i, j]
            _, p = ttest_ind(a, b, equal_var=False)
            pvals[i, j] = p

    iu = np.triu_indices(n_chan, 1)
    pvals_flat = pvals[iu]
    sig_flat = benjamini_hochberg(pvals_flat, alpha=0.05)
    sig_mask = np.zeros_like(pvals, dtype=bool)
    sig_mask[iu] = sig_flat
    sig_mask = sig_mask | sig_mask.T

    np.save(os.path.join(out_dir, f'pvals_{band_name}.npy'), pvals)
    np.save(os.path.join(out_dir, f'sigmask_{band_name}.npy'), sig_mask)

    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.imshow(avg_conn_pre, cmap=cmap, vmin=-1, vmax=1)
    plt.title(f'{band_name} PRE')
    plt.colorbar()
    plt.subplot(1,3,2)
    plt.imshow(avg_conn_post, cmap=cmap, vmin=-1, vmax=1)
    plt.title(f'{band_name} POST')
    plt.colorbar()
    plt.subplot(1,3,3)
    plt.imshow(diff_conn, cmap=cmap, vmin=-1, vmax=1)
    plt.title(f'{band_name} DIFF')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'connectivity_{band_name}.png'))
    plt.show()

    sig_diff_mat = np.zeros_like(diff_conn)
    sig_diff_mat[sig_mask] = diff_conn[sig_mask]
    plt.figure(figsize=(6,6))
    plt.imshow(sig_diff_mat, cmap=cmap, vmin=-1, vmax=1)
    plt.title(f'{band_name} SIGNIFICANT DIFF (FDR)')
    plt.colorbar()
    plt.savefig(os.path.join(out_dir, f'significant_diff_{band_name}.png'))
    plt.show()

    coords = {idx: (idx % 8, 7 - (idx // 8)) for idx in range(64)}
    plt.figure(figsize=(6,6))
    for i in range(64):
        x, y = coords[i]
        plt.scatter(x, y, c='k', s=25)
    pairs = np.argwhere(sig_mask)
    drawn = set()
    for (i, j) in pairs:
        if i < j and (i, j) not in drawn:
            x1, y1 = coords[i]; x2, y2 = coords[j]
            plt.plot([x1, x2], [y1, y2], color="#800000", alpha=0.6, linewidth=0.8)
            drawn.add((i, j))
    plt.title(f'{band_name} - significant edges (FDR)')
    plt.axis('off')
    plt.savefig(os.path.join(out_dir, f'electrode_mapping_{band_name}.png'))
    plt.show()

    mean_conn_pre_ts = pre_R_windows.mean(axis=(0,2,3))
    mean_conn_post_ts = post_R_windows.mean(axis=(0,2,3))
    plt.figure()
    plt.plot(mean_conn_pre_ts, label='Pre mean connectivity', color='#2c2c2c')
    plt.plot(mean_conn_post_ts, label='Post mean connectivity', color='#800000')
    plt.xlabel('Window index')
    plt.ylabel('Mean connectivity')
    plt.legend()
    plt.title(f'{band_name} - Temporal dynamics (mean over windows and channels)')
    plt.savefig(os.path.join(out_dir, f'temporal_mean_{band_name}.png'))
    plt.show()

    avg_R_pre_ts = pre_R_windows.mean(axis=0)
    avg_R_post_ts = post_R_windows.mean(axis=0)
    np.save(os.path.join(out_dir, f'R_windows_{band_name}_pre.npy'), avg_R_pre_ts)
    np.save(os.path.join(out_dir, f'R_windows_{band_name}_post.npy'), avg_R_post_ts)

    fig, ax = plt.subplots(figsize=(6,6))
    im = ax.imshow(avg_R_post_ts[0], cmap=cmap, vmin=-1, vmax=1)
    fig.colorbar(im)
    def update(frame):
        im.set_array(avg_R_post_ts[frame])
        ax.set_title(f'{band_name} Post - frame {frame}')
        return [im]
    ani = animation.FuncAnimation(fig, update, frames=avg_R_post_ts.shape[0], interval=200, blit=True)
    ani_path = os.path.join(out_dir, f'connectivity_anim_{band_name}_post.mp4')
    ani.save(ani_path, writer='ffmpeg')
    plt.close(fig)

    results_summary[band_name] = {
        "avg_conn_pre": avg_conn_pre,
        "avg_conn_post": avg_conn_post,
        "diff_conn": diff_conn,
        "pvals": pvals,
        "sig_mask": sig_mask,
        "R_windows_pre": avg_R_pre_ts,
        "R_windows_post": avg_R_post_ts,
        "mean_conn_pre_ts": mean_conn_pre_ts,
        "mean_conn_post_ts": mean_conn_post_ts
    }

np.savez(os.path.join(out_dir, "connectivity_summary_allbands.npz"), **results_summary)
print("All results saved in:", out_dir)


import os
import glob
import h5py
import numpy as np
from scipy.signal import butter, filtfilt, hilbert
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
from matplotlib import colors, animation

base_dir = r'\\bigdata\Science\Med\Physiology\PTN\Yasmine\IC-stroke_connectivity\data'
out_dir = r'\\bigdata\Science\Med\Physiology\PTN\Yasmine\IC-stroke_connectivity\ConnectivityResults'
os.makedirs(out_dir, exist_ok=True)

fs = 2000
date_cutoff = 20250811

bad_channel_indices = [64, 65, 66]

bands = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 12),
    "beta": (12, 30),
    "gamma": (30, 70),
    "high_gamma": (70, 150)
}

cmap = colors.LinearSegmentedColormap.from_list("gray_red", ["#2c2c2c", "#bfbfbf", "#ffb3b3", "#800000"])

def load_trial_mat(filepath):
    with h5py.File(filepath, "r") as f:
        signals = np.array(f["signals"])
    return signals

def bandpass(data, low, high, fs, order=4):
    b, a = butter(order, [low/(fs/2), high/(fs/2)], btype="band")
    return filtfilt(b, a, data, axis=-1)

sess_dirs = [d for d in glob.glob(os.path.join(base_dir, "*")) if os.path.isdir(d)]
pre_trials = []
post_trials = []

for sess in sess_dirs:
    sess_name = os.path.basename(sess)
    try:
        sess_date = int(sess_name)
    except:
        continue
    trial_files = glob.glob(os.path.join(sess, "trial*", "MatFiles", "*_allChannels.mat"))
    for tf in trial_files:
        try:
            sigs = load_trial_mat(tf)
        except Exception:
            continue
        if sigs.shape[0] < 64:
            continue
        X64 = sigs[:64, :]
        if sess_date < date_cutoff:
            pre_trials.append(X64)
        else:
            post_trials.append(X64)

if len(pre_trials) == 0 or len(post_trials) == 0:
    raise RuntimeError("Not enough pre or post trials found. Check base_dir and date_cutoff.")

min_len = min([t.shape[1] for t in pre_trials + post_trials])
pre_trials = [t[:, :min_len] for t in pre_trials]
post_trials = [t[:, :min_len] for t in post_trials]
pre_arr = np.stack(pre_trials, axis=0)
post_arr = np.stack(post_trials, axis=0)
n_pre, n_chan, n_time = pre_arr.shape
n_post = post_arr.shape[0]

ds_factor = 10
fs_ds = fs // ds_factor
win_sec = 0.2
step_sec = 0.1
win_ds = int(win_sec * fs_ds)
step_ds = int(step_sec * fs_ds)

def sliding_windows_env(env, win, step):
    n_chan, n_t = env.shape
    windows = []
    starts = list(range(0, n_t - win + 1, step))
    for s in starts:
        windows.append(env[:, s:s+win])
    return np.stack(windows, axis=0)

def corr_matrix_from_env(env):
    return np.corrcoef(env)

def benjamini_hochberg(pvals_flat, alpha=0.05):
    m = len(pvals_flat)
    sorted_idx = np.argsort(pvals_flat)
    sorted_p = pvals_flat[sorted_idx]
    thresh = np.arange(1, m+1) * alpha / m
    below = sorted_p <= thresh
    if not np.any(below):
        return np.zeros(m, dtype=bool)
    max_i = np.max(np.where(below)[0])
    crit = thresh[max_i]
    return pvals_flat <= crit

results_summary = {}

for band_name, (low, high) in bands.items():
    pre_env_trials = []
    post_env_trials = []
    for tr in range(n_pre):
        X = pre_arr[tr]
        Xf = bandpass(X, low, high, fs)
        env = np.abs(hilbert(Xf, axis=-1))[:, ::ds_factor]
        pre_env_trials.append(env)
    for tr in range(n_post):
        X = post_arr[tr]
        Xf = bandpass(X, low, high, fs)
        env = np.abs(hilbert(Xf, axis=-1))[:, ::ds_factor]
        post_env_trials.append(env)
    pre_env_trials = np.array(pre_env_trials)
    post_env_trials = np.array(post_env_trials)

    n_windows = (pre_env_trials.shape[2] - win_ds) // step_ds + 1
    pre_R_windows = np.zeros((n_pre, n_windows, n_chan, n_chan))
    post_R_windows = np.zeros((n_post, n_windows, n_chan, n_chan))

    for t in range(n_pre):
        win_stack = sliding_windows_env(pre_env_trials[t], win_ds, step_ds)
        for w in range(win_stack.shape[0]):
            pre_R_windows[t, w] = corr_matrix_from_env(win_stack[w])
    for t in range(n_post):
        win_stack = sliding_windows_env(post_env_trials[t], win_ds, step_ds)
        for w in range(win_stack.shape[0]):
            post_R_windows[t, w] = corr_matrix_from_env(win_stack[w])

    mean_Rpre_per_trial = pre_R_windows.mean(axis=1)
    mean_Rpost_per_trial = post_R_windows.mean(axis=1)
    avg_conn_pre = mean_Rpre_per_trial.mean(axis=0)
    avg_conn_post = mean_Rpost_per_trial.mean(axis=0)
    diff_conn = avg_conn_post - avg_conn_pre

    np.save(os.path.join(out_dir, f'R_mean_{band_name}_pre.npy'), avg_conn_pre)
    np.save(os.path.join(out_dir, f'R_mean_{band_name}_post.npy'), avg_conn_post)
    np.save(os.path.join(out_dir, f'R_diff_{band_name}.npy'), diff_conn)

    pvals = np.ones((n_chan, n_chan))
    for i in range(n_chan):
        for j in range(n_chan):
            a = mean_Rpre_per_trial[:, i, j]
            b = mean_Rpost_per_trial[:, i, j]
            _, p = ttest_ind(a, b, equal_var=False)
            pvals[i, j] = p

    iu = np.triu_indices(n_chan, 1)
    pvals_flat = pvals[iu]
    sig_flat = benjamini_hochberg(pvals_flat, alpha=0.05)
    sig_mask = np.zeros_like(pvals, dtype=bool)
    sig_mask[iu] = sig_flat
    sig_mask = sig_mask | sig_mask.T

    np.save(os.path.join(out_dir, f'pvals_{band_name}.npy'), pvals)
    np.save(os.path.join(out_dir, f'sigmask_{band_name}.npy'), sig_mask)

    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.imshow(avg_conn_pre, cmap=cmap, vmin=-1, vmax=1)
    plt.title(f'{band_name} PRE')
    plt.colorbar()
    plt.subplot(1,3,2)
    plt.imshow(avg_conn_post, cmap=cmap, vmin=-1, vmax=1)
    plt.title(f'{band_name} POST')
    plt.colorbar()
    plt.subplot(1,3,3)
    plt.imshow(diff_conn, cmap=cmap, vmin=-1, vmax=1)
    plt.title(f'{band_name} DIFF')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'connectivity_{band_name}.png'))
    plt.show()

    sig_diff_mat = np.zeros_like(diff_conn)
    sig_diff_mat[sig_mask] = diff_conn[sig_mask]
    plt.figure(figsize=(6,6))
    plt.imshow(sig_diff_mat, cmap=cmap, vmin=-1, vmax=1)
    plt.title(f'{band_name} SIGNIFICANT DIFF (FDR)')
    plt.colorbar()
    plt.savefig(os.path.join(out_dir, f'significant_diff_{band_name}.png'))
    plt.show()

    coords = {idx: (idx % 8, 7 - (idx // 8)) for idx in range(64)}
    plt.figure(figsize=(6,6))
    for i in range(64):
        x, y = coords[i]
        plt.scatter(x, y, c='k', s=25)
    pairs = np.argwhere(sig_mask)
    drawn = set()
    for (i, j) in pairs:
        if i < j and (i, j) not in drawn:
            x1, y1 = coords[i]; x2, y2 = coords[j]
            plt.plot([x1, x2], [y1, y2], color="#800000", alpha=0.6, linewidth=0.8)
            drawn.add((i, j))
    plt.title(f'{band_name} - significant edges (FDR)')
    plt.axis('off')
    plt.savefig(os.path.join(out_dir, f'electrode_mapping_{band_name}.png'))
    plt.show()

    mean_conn_pre_ts = pre_R_windows.mean(axis=(0,2,3))
    mean_conn_post_ts = post_R_windows.mean(axis=(0,2,3))
    plt.figure()
    plt.plot(mean_conn_pre_ts, label='Pre mean connectivity', color='#2c2c2c')
    plt.plot(mean_conn_post_ts, label='Post mean connectivity', color='#800000')
    plt.xlabel('Window index')
    plt.ylabel('Mean connectivity')
    plt.legend()
    plt.title(f'{band_name} - Temporal dynamics (mean over windows and channels)')
    plt.savefig(os.path.join(out_dir, f'temporal_mean_{band_name}.png'))
    plt.show()

    avg_R_pre_ts = pre_R_windows.mean(axis=0)
    avg_R_post_ts = post_R_windows.mean(axis=0)
    np.save(os.path.join(out_dir, f'R_windows_{band_name}_pre.npy'), avg_R_pre_ts)
    np.save(os.path.join(out_dir, f'R_windows_{band_name}_post.npy'), avg_R_post_ts)

    fig, ax = plt.subplots(figsize=(6,6))
    im = ax.imshow(avg_R_post_ts[0], cmap=cmap, vmin=-1, vmax=1)
    fig.colorbar(im)
    def update(frame):
        im.set_array(avg_R_post_ts[frame])
        ax.set_title(f'{band_name} Post - frame {frame}')
        return [im]
    ani = animation.FuncAnimation(fig, update, frames=avg_R_post_ts.shape[0], interval=200, blit=True)
    ani_path = os.path.join(out_dir, f'connectivity_anim_{band_name}_post.mp4')
    ani.save(ani_path, writer='ffmpeg')
    plt.close(fig)

    results_summary[band_name] = {
        "avg_conn_pre": avg_conn_pre,
        "avg_conn_post": avg_conn_post,
        "diff_conn": diff_conn,
        "pvals": pvals,
        "sig_mask": sig_mask,
        "R_windows_pre": avg_R_pre_ts,
        "R_windows_post": avg_R_post_ts,
        "mean_conn_pre_ts": mean_conn_pre_ts,
        "mean_conn_post_ts": mean_conn_post_ts
    }

np.savez(os.path.join(out_dir, "connectivity_summary_allbands.npz"), **results_summary)
print("All results saved in:", out_dir)

import os
import glob
import h5py
import numpy as np
from scipy.signal import butter, filtfilt, hilbert
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
from matplotlib import colors, animation

base_dir = r'\\bigdata\Science\Med\Physiology\PTN\Yasmine\IC-stroke_connectivity\data'
out_dir = r'\\bigdata\Science\Med\Physiology\PTN\Yasmine\IC-stroke_connectivity\ConnectivityResults'
os.makedirs(out_dir, exist_ok=True)

fs = 2000
date_cutoff = 20250811

bad_channel_indices = [64, 65, 66]

bands = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 12),
    "beta": (12, 30),
    "gamma": (30, 70),
    "high_gamma": (70, 150)
}

cmap = colors.LinearSegmentedColormap.from_list("gray_red", ["#2c2c2c", "#bfbfbf", "#ffb3b3", "#800000"])

def load_trial_mat(filepath):
    with h5py.File(filepath, "r") as f:
        signals = np.array(f["signals"])
    return signals

def bandpass(data, low, high, fs, order=4):
    b, a = butter(order, [low/(fs/2), high/(fs/2)], btype="band")
    return filtfilt(b, a, data, axis=-1)

sess_dirs = [d for d in glob.glob(os.path.join(base_dir, "*")) if os.path.isdir(d)]
pre_trials = []
post_trials = []

for sess in sess_dirs:
    sess_name = os.path.basename(sess)
    try:
        sess_date = int(sess_name)
    except:
        continue
    trial_files = glob.glob(os.path.join(sess, "trial*", "MatFiles", "*_allChannels.mat"))
    for tf in trial_files:
        try:
            sigs = load_trial_mat(tf)
        except Exception:
            continue
        if sigs.shape[0] < 64:
            continue
        X64 = sigs[:64, :]
        if sess_date < date_cutoff:
            pre_trials.append(X64)
        else:
            post_trials.append(X64)

if len(pre_trials) == 0 or len(post_trials) == 0:
    raise RuntimeError("Not enough pre or post trials found. Check base_dir and date_cutoff.")

min_len = min([t.shape[1] for t in pre_trials + post_trials])
pre_trials = [t[:, :min_len] for t in pre_trials]
post_trials = [t[:, :min_len] for t in post_trials]
pre_arr = np.stack(pre_trials, axis=0)
post_arr = np.stack(post_trials, axis=0)
n_pre, n_chan, n_time = pre_arr.shape
n_post = post_arr.shape[0]

ds_factor = 10
fs_ds = fs // ds_factor
win_sec = 0.2
step_sec = 0.1
win_ds = int(win_sec * fs_ds)
step_ds = int(step_sec * fs_ds)

def sliding_windows_env(env, win, step):
    n_chan, n_t = env.shape
    windows = []
    starts = list(range(0, n_t - win + 1, step))
    for s in starts:
        windows.append(env[:, s:s+win])
    return np.stack(windows, axis=0)

def corr_matrix_from_env(env):
    return np.corrcoef(env)

def benjamini_hochberg(pvals_flat, alpha=0.05):
    m = len(pvals_flat)
    sorted_idx = np.argsort(pvals_flat)
    sorted_p = pvals_flat[sorted_idx]
    thresh = np.arange(1, m+1) * alpha / m
    below = sorted_p <= thresh
    if not np.any(below):
        return np.zeros(m, dtype=bool)
    max_i = np.max(np.where(below)[0])
    crit = thresh[max_i]
    return pvals_flat <= crit

results_summary = {}

for band_name, (low, high) in bands.items():
    pre_env_trials = []
    post_env_trials = []
    for tr in range(n_pre):
        X = pre_arr[tr]
        Xf = bandpass(X, low, high, fs)
        env = np.abs(hilbert(Xf, axis=-1))[:, ::ds_factor]
        pre_env_trials.append(env)
    for tr in range(n_post):
        X = post_arr[tr]
        Xf = bandpass(X, low, high, fs)
        env = np.abs(hilbert(Xf, axis=-1))[:, ::ds_factor]
        post_env_trials.append(env)
    pre_env_trials = np.array(pre_env_trials)
    post_env_trials = np.array(post_env_trials)

    n_windows = (pre_env_trials.shape[2] - win_ds) // step_ds + 1
    pre_R_windows = np.zeros((n_pre, n_windows, n_chan, n_chan))
    post_R_windows = np.zeros((n_post, n_windows, n_chan, n_chan))

    for t in range(n_pre):
        win_stack = sliding_windows_env(pre_env_trials[t], win_ds, step_ds)
        for w in range(win_stack.shape[0]):
            pre_R_windows[t, w] = corr_matrix_from_env(win_stack[w])
    for t in range(n_post):
        win_stack = sliding_windows_env(post_env_trials[t], win_ds, step_ds)
        for w in range(win_stack.shape[0]):
            post_R_windows[t, w] = corr_matrix_from_env(win_stack[w])

    mean_Rpre_per_trial = pre_R_windows.mean(axis=1)
    mean_Rpost_per_trial = post_R_windows.mean(axis=1)
    avg_conn_pre = mean_Rpre_per_trial.mean(axis=0)
    avg_conn_post = mean_Rpost_per_trial.mean(axis=0)
    diff_conn = avg_conn_post - avg_conn_pre

    np.save(os.path.join(out_dir, f'R_mean_{band_name}_pre.npy'), avg_conn_pre)
    np.save(os.path.join(out_dir, f'R_mean_{band_name}_post.npy'), avg_conn_post)
    np.save(os.path.join(out_dir, f'R_diff_{band_name}.npy'), diff_conn)

    pvals = np.ones((n_chan, n_chan))
    for i in range(n_chan):
        for j in range(n_chan):
            a = mean_Rpre_per_trial[:, i, j]
            b = mean_Rpost_per_trial[:, i, j]
            _, p = ttest_ind(a, b, equal_var=False)
            pvals[i, j] = p

    iu = np.triu_indices(n_chan, 1)
    pvals_flat = pvals[iu]
    sig_flat = benjamini_hochberg(pvals_flat, alpha=0.05)
    sig_mask = np.zeros_like(pvals, dtype=bool)
    sig_mask[iu] = sig_flat
    sig_mask = sig_mask | sig_mask.T

    np.save(os.path.join(out_dir, f'pvals_{band_name}.npy'), pvals)
    np.save(os.path.join(out_dir, f'sigmask_{band_name}.npy'), sig_mask)

    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.imshow(avg_conn_pre, cmap=cmap, vmin=-1, vmax=1)
    plt.title(f'{band_name} PRE')
    plt.colorbar()
    plt.subplot(1,3,2)
    plt.imshow(avg_conn_post, cmap=cmap, vmin=-1, vmax=1)
    plt.title(f'{band_name} POST')
    plt.colorbar()
    plt.subplot(1,3,3)
    plt.imshow(diff_conn, cmap=cmap, vmin=-1, vmax=1)
    plt.title(f'{band_name} DIFF')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'connectivity_{band_name}.png'))
    plt.show()

    sig_diff_mat = np.zeros_like(diff_conn)
    sig_diff_mat[sig_mask] = diff_conn[sig_mask]
    plt.figure(figsize=(6,6))
    plt.imshow(sig_diff_mat, cmap=cmap, vmin=-1, vmax=1)
    plt.title(f'{band_name} SIGNIFICANT DIFF (FDR)')
    plt.colorbar()
    plt.savefig(os.path.join(out_dir, f'significant_diff_{band_name}.png'))
    plt.show()

    coords = {idx: (idx % 8, 7 - (idx // 8)) for idx in range(64)}
    plt.figure(figsize=(6,6))
    for i in range(64):
        x, y = coords[i]
        plt.scatter(x, y, c='k', s=25)
    pairs = np.argwhere(sig_mask)
    drawn = set()
    for (i, j) in pairs:
        if i < j and (i, j) not in drawn:
            x1, y1 = coords[i]; x2, y2 = coords[j]
            plt.plot([x1, x2], [y1, y2], color="#800000", alpha=0.6, linewidth=0.8)
            drawn.add((i, j))
    plt.title(f'{band_name} - significant edges (FDR)')
    plt.axis('off')
    plt.savefig(os.path.join(out_dir, f'electrode_mapping_{band_name}.png'))
    plt.show()

    mean_conn_pre_ts = pre_R_windows.mean(axis=(0,2,3))
    mean_conn_post_ts = post_R_windows.mean(axis=(0,2,3))
    plt.figure()
    plt.plot(mean_conn_pre_ts, label='Pre mean connectivity', color='#2c2c2c')
    plt.plot(mean_conn_post_ts, label='Post mean connectivity', color='#800000')
    plt.xlabel('Window index')
    plt.ylabel('Mean connectivity')
    plt.legend()
    plt.title(f'{band_name} - Temporal dynamics (mean over windows and channels)')
    plt.savefig(os.path.join(out_dir, f'temporal_mean_{band_name}.png'))
    plt.show()

    avg_R_pre_ts = pre_R_windows.mean(axis=0)
    avg_R_post_ts = post_R_windows.mean(axis=0)
    np.save(os.path.join(out_dir, f'R_windows_{band_name}_pre.npy'), avg_R_pre_ts)
    np.save(os.path.join(out_dir, f'R_windows_{band_name}_post.npy'), avg_R_post_ts)

    fig, ax = plt.subplots(figsize=(6,6))
    im = ax.imshow(avg_R_post_ts[0], cmap=cmap, vmin=-1, vmax=1)
    fig.colorbar(im)
    def update(frame):
        im.set_array(avg_R_post_ts[frame])
        ax.set_title(f'{band_name} Post - frame {frame}')
        return [im]
    ani = animation.FuncAnimation(fig, update, frames=avg_R_post_ts.shape[0], interval=200, blit=True)
    ani_path = os.path.join(out_dir, f'connectivity_anim_{band_name}_post.mp4')
    ani.save(ani_path, writer='ffmpeg')
    plt.close(fig)

    results_summary[band_name] = {
        "avg_conn_pre": avg_conn_pre,
        "avg_conn_post": avg_conn_post,
        "diff_conn": diff_conn,
        "pvals": pvals,
        "sig_mask": sig_mask,
        "R_windows_pre": avg_R_pre_ts,
        "R_windows_post": avg_R_post_ts,
        "mean_conn_pre_ts": mean_conn_pre_ts,
        "mean_conn_post_ts": mean_conn_post_ts
    }

np.savez(os.path.join(out_dir, "connectivity_summary_allbands.npz"), **results_summary)
print("All results saved in:", out_dir)
