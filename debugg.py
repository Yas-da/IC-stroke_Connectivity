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

cmap = LinearSegmentedColormap.from_list("gray_red", ["#2c2c2c", "#bfbfbf", "#ffb3b3", "#800000"])

ds = 10  # facteur de downsample (+ ça marche pas)

def load_signals_from_mat(path):
    with h5py.File(path, "r") as f:
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

per_trial_dir = os.path.join(save_dir, "per_trial")
per_session_dir = os.path.join(save_dir, "per_session")
agg_dir = os.path.join(save_dir, "aggregated")
diag_dir = os.path.join(save_dir, "diagnostics")
for d in (per_trial_dir, per_session_dir, agg_dir, diag_dir):
    os.makedirs(d, exist_ok=True)

sess_dirs = [d for d in glob.glob(os.path.join(base_dir, "*")) if os.path.isdir(d)]
session_map = {} 
for sess in sess_dirs:
    sess_name = os.path.basename(sess)
    try:
        sess_date = int(sess_name)
    except:
        continue
    trial_files = sorted(glob.glob(os.path.join(sess, "trial*", "MatFiles", "*_allChannels.mat")))
    if len(trial_files) > 0:
        session_map[sess_name] = trial_files

if len(session_map) == 0:
    raise RuntimeError("No sessions/trials found — check base_dir structure.")

print(f"Found {len(session_map)} sessions with trials.")

def save_heatmaps_and_mats(out_folder, tag, R_signed):
    np.save(os.path.join(out_folder, f"{tag}_R_signed.npy"), R_signed)
    np.save(os.path.join(out_folder, f"{tag}_R_01.npy"), (R_signed + 1) / 2.0)
    np.save(os.path.join(out_folder, f"{tag}_R_abs.npy"), np.abs(R_signed))
    np.save(os.path.join(out_folder, f"{tag}_R_sq.npy"), R_signed**2)

    # images
    plt.figure(figsize=(10,3))
    ax = plt.subplot(1,3,1)
    im = ax.imshow(R_signed, vmin=-1, vmax=1, cmap=cmap); ax.set_title(f"{tag} signed")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    ax = plt.subplot(1,3,2)
    im2 = ax.imshow((R_signed+1)/2, vmin=0, vmax=1, cmap=cmap); ax.set_title(f"{tag} 0..1")
    plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.02)
    ax = plt.subplot(1,3,3)
    im3 = ax.imshow(np.abs(R_signed), vmin=0, vmax=1, cmap=cmap); ax.set_title(f"{tag} abs")
    plt.colorbar(im3, ax=ax, fraction=0.046, pad=0.02)
    plt.suptitle(tag)
    plt.tight_layout(rect=[0,0,1,0.95])
    plt.savefig(os.path.join(out_folder, f"{tag}_heatmaps.png"), dpi=200)
    plt.close()

def compute_env_variant(sig, fmin, fmax, fs, ds, mode="raw"):
    if mode == "CAR":
        sig_used = sig - np.mean(sig, axis=0, keepdims=True)
    else:
        sig_used = sig.copy()

    Xf = bandpass_filter(sig_used, fmin, fmax, fs)
    env = np.abs(hilbert(Xf, axis=1))
    env_ds = env[:, ::ds]

    if mode == "demean":
        env_ds = env_ds - np.mean(env_ds, axis=0, keepdims=True)

    return env_ds

print("STEP 1 — per-trial connectivity matrices (envelope corr) ...")
for sess_name, tfiles in session_map.items():
    sess_out = os.path.join(per_trial_dir, sess_name)
    os.makedirs(sess_out, exist_ok=True)
    for tf in tfiles:
        try:
            S = load_signals_from_mat(tf)  # shape channels x T
        except Exception as e:
            print("  skip (load error):", tf, e)
            continue
        if S.shape[0] < channels_keep:
            print("  skip (too few channels):", tf)
            continue
        S = S[:channels_keep, :]
        #Per-band per-trial
        for bname, (fmin, fmax) in bands.items():
            env = np.abs(hilbert(bandpass_filter(S, fmin, fmax, fs), axis=1))[:, ::ds]  # (64, T_ds)
            if env.shape[1] < 10:
                print("    warning: very short after downsample:", tf, bname, "len", env.shape[1])
            R = np.corrcoef(env)
            tag = os.path.splitext(os.path.basename(tf))[0] + f"_{bname}"
            save_heatmaps_and_mats(sess_out, tag, R)
    print("  done session", sess_name)

#Mean across trials then connectivity
print("STEP 2 — per-session connectivity (mean across trials, variants) ...")
per_session_results = {}  
for sess_name, tfiles in session_map.items():
    trials = []
    for tf in tfiles:
        try:
            S = load_signals_from_mat(tf)
        except Exception:
            continue
        if S.shape[0] < channels_keep:
            continue
        trials.append(S[:channels_keep, :])
    if len(trials) == 0:
        continue
    min_len = min(t.shape[1] for t in trials)
    trials_trim = [t[:, :min_len] for t in trials]  #list of (64, min_len)
    trials_arr = np.stack(trials_trim, axis=0)      #(n_trials, 64, min_len)

    sess_out = os.path.join(per_session_dir, sess_name)
    os.makedirs(sess_out, exist_ok=True)

    per_session_results[sess_name] = {}
    #Mean per-channel across trials
    mean_sig = np.mean(trials_arr, axis=0)  #(64, T_session)
    np.save(os.path.join(sess_out, f"{sess_name}_mean_signal.npy"), mean_sig)
    
    sample_envs = {}

    for bname, (fmin, fmax) in bands.items():
        per_session_results[sess_name][bname] = {}
        for mode in ("raw", "CAR", "demean"):
            env = compute_env_variant(mean_sig, fmin, fmax, fs, ds, mode=mode)  #(64, T_ds)
            if env.shape[1] < 10:
                print(f"  Warning: session {sess_name} band {bname} mode {mode} -> very short after ds ({env.shape[1]})")
            R = np.corrcoef(env)
            per_session_results[sess_name][bname][mode] = R

            tag = f"{sess_name}_{bname}_{mode}"
            save_heatmaps_and_mats(sess_out, tag, R)
    
        sample_sig = trials_trim[0]  #(64, T)
        sample_envs[bname] = np.abs(hilbert(bandpass_filter(sample_sig, bname and bands[bname][0], bands[bname][1], fs), axis=1))[:, ::ds]

    for bname in bands.keys():
        try:
            example_env = sample_envs[bname]
            tvec = np.arange(example_env.shape[1]) * (ds / fs)
            plt.figure(figsize=(8,4))
            for ch in range(min(6, example_env.shape[0])):
                plt.plot(tvec, example_env[ch] / (np.max(np.abs(example_env[ch])) + 1e-12), alpha=0.8, label=f"ch{ch+1}")
            plt.title(f"{sess_name} example envelope first 6 channels - {bname}")
            plt.xlabel("Time (s)")
            plt.legend(ncol=3)
            plt.tight_layout()
            plt.savefig(os.path.join(sess_out, f"{sess_name}_env_example_{bname}.png"), dpi=200)
            plt.close()
        except Exception:
            pass

print("Per-session processing done.")

print("STEP 3 — diagnostics and aggregated pre/post matrices ...")
pre_sessions = [s for s in per_session_results.keys() if int(s) < lesion_date]
post_sessions = [s for s in per_session_results.keys() if int(s) >= lesion_date]

agg_results = {}
os.makedirs(agg_dir, exist_ok=True)

with open(os.path.join(diag_dir, "diagnostics_summary.txt"), "w") as df:
    for bname in bands.keys():
        agg_results[bname] = {}
        for mode in ("raw", "CAR", "demean"):
            # collect session matrices
            R_pre_list = [per_session_results[s][bname][mode] for s in pre_sessions if bname in per_session_results[s]]
            R_post_list = [per_session_results[s][bname][mode] for s in post_sessions if bname in per_session_results[s]]

            if len(R_pre_list) > 0:
                R_pre_mean = np.mean(np.stack(R_pre_list, axis=0), axis=0)
            else:
                R_pre_mean = None
            if len(R_post_list) > 0:
                R_post_mean = np.mean(np.stack(R_post_list, axis=0), axis=0)
            else:
                R_post_mean = None

            agg_results[bname][mode] = {"R_pre_mean": R_pre_mean, "R_post_mean": R_post_mean,
                                       "n_pre": len(R_pre_list), "n_post": len(R_post_list)}

            if R_pre_mean is not None:
                np.save(os.path.join(agg_dir, f"{bname}_{mode}_pre_mean.npy"), R_pre_mean)
            if R_post_mean is not None:
                np.save(os.path.join(agg_dir, f"{bname}_{mode}_post_mean.npy"), R_post_mean)

            if R_pre_mean is not None or R_post_mean is not None:
                plt.figure(figsize=(10,4))
                if R_pre_mean is not None:
                    ax = plt.subplot(1,2,1)
                    im = ax.imshow(R_pre_mean, vmin=-1, vmax=1, cmap=cmap); ax.set_title(f"{bname} {mode} PRE (n={len(R_pre_list)})")
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
                if R_post_mean is not None:
                    ax2 = plt.subplot(1,2,2)
                    im2 = ax2.imshow(R_post_mean, vmin=-1, vmax=1, cmap=cmap); ax2.set_title(f"{bname} {mode} POST (n={len(R_post_list)})")
                    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.02)
                plt.suptitle(f"Aggregated PRE vs POST - {bname} - {mode}")
                plt.tight_layout(rect=[0,0,1,0.95])
                plt.savefig(os.path.join(agg_dir, f"aggregated_{bname}_{mode}.png"), dpi=200)
                plt.close()

            #diagn: distributions and stats
            def write_stats(tag, Rmat):
                if Rmat is None:
                    df.write(f"{tag}: none\n")
                    return
                vals = Rmat[np.triu_indices_from(Rmat, 1)]
                df.write(f"{tag}: n_vals={vals.size}, mean={np.nanmean(vals):.4f}, std={np.nanstd(vals):.4f}, median={np.nanmedian(vals):.4f}, min={np.nanmin(vals):.4f}, max={np.nanmax(vals):.4f}\n")
                #histogram
                plt.figure(figsize=(5,3))
                plt.hist(vals, bins=60)
                plt.title(f"Hist {tag}")
                plt.tight_layout()
                plt.savefig(os.path.join(diag_dir, f"hist_{tag}.png"), dpi=150)
                plt.close()

            write_stats(f"{bname}_{mode}_pre", R_pre_mean)
            write_stats(f"{bname}_{mode}_post", R_post_mean)

print("Aggregated pre/post saved and diagnostics written.")

print("Step 4 — statistical tests across sessions for edges and nodes (Fisher z + t-test + BH)...")
stats_out = {}
for bname in bands.keys():
    stats_out[bname] = {}
    for mode in ("raw", "CAR", "demean"):
        pre_Rs = [per_session_results[s][bname][mode] for s in pre_sessions if bname in per_session_results[s]]
        post_Rs = [per_session_results[s][bname][mode] for s in post_sessions if bname in per_session_results[s]]
        if len(pre_Rs) == 0 or len(post_Rs) == 0:
            stats_out[bname][mode] = {"edge_pvals": None, "sig_mask": None}
            continue
        pre_Rs = np.stack(pre_Rs, axis=0)   #(n_pre, C, C)
        post_Rs = np.stack(post_Rs, axis=0) #(n_post, C, C)
        iu = np.triu_indices(channels_keep, 1)
        A = fisher_z(pre_Rs[:, iu[0], iu[1]])
        B = fisher_z(post_Rs[:, iu[0], iu[1]])
        pvals = np.ones(A.shape[1])
        for e in range(A.shape[1]):
            _, p = ttest_ind(A[:, e], B[:, e], equal_var=False)
            pvals[e] = p
        sig_flat = benjamini_hochberg(pvals, alpha=0.05)
        sig_mask = np.zeros((channels_keep, channels_keep), dtype=bool)
        sig_mask[iu] = sig_flat
        sig_mask = sig_mask | sig_mask.T

        #sum of R/session
        node_pre = pre_Rs.sum(axis=2)   #(n_pre, C)
        node_post = post_Rs.sum(axis=2) #(n_post, C)
        p_node = np.ones(channels_keep)
        for ch in range(channels_keep):
            _, p = ttest_ind(node_pre[:, ch], node_post[:, ch], equal_var=False)
            p_node[ch] = p
        sig_node_mask = benjamini_hochberg(p_node, alpha=0.05)

        stats_out[bname][mode] = {
            "edge_pvals": pvals,
            "sig_edges_mask": sig_mask,
            "sig_nodes_mask": sig_node_mask,
            "node_pre": node_pre,
            "node_post": node_post
        }

np.savez(os.path.join(save_dir, "connectivity_stats_per_band_mode.npz"), stats=stats_out)

print("Statistical comparisons saved.")

np.savez(os.path.join(save_dir, "connectivity_pipeline_summary.npz"),
         per_session_results=per_session_results,
         agg_results=agg_results,
         bands=list(bands.keys()),
         channels_keep=channels_keep,
         lesion_date=lesion_date)

print("All outputs saved to:", save_dir)
print("Folders created: per_trial, per_session, aggregated, diagnostics")
