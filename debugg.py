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

MIN_SAMPLES = 200   # minimum number of time points after downsampling to compute stable corr
MAX_DS = 50         # safety cap on ds
VERBOSE = True

def load_signals_from_mat(path):
    with h5py.File(path, "r") as f:
        if "signals" in f:
            signals = np.array(f["signals"])
        elif "data" in f:
            signals = np.array(f["data"])
        else:
            raise KeyError(f"'signals' not found in {path}")
    return signals[:channels_keep, :]

def bandpass_filter(data, low, high, fs, order=4):
    b, a = butter(order, [low / (fs / 2), high / (fs / 2)], btype="band")
    return filtfilt(b, a, data, axis=-1)

def save_heatmaps_and_mats(out_folder, tag, R_signed):
    np.save(os.path.join(out_folder, f"{tag}_R_signed.npy"), R_signed)
    np.save(os.path.join(out_folder, f"{tag}_R_01.npy"), (R_signed + 1) / 2.0)
    np.save(os.path.join(out_folder, f"{tag}_R_abs.npy"), np.abs(R_signed))
    plt.figure(figsize=(9,3))
    ax = plt.subplot(1,3,1); ax.imshow(R_signed, vmin=-1, vmax=1, cmap=cmap); ax.set_title("signed")
    ax = plt.subplot(1,3,2); ax.imshow((R_signed+1)/2, vmin=0, vmax=1, cmap=cmap); ax.set_title("0..1")
    ax = plt.subplot(1,3,3); ax.imshow(np.abs(R_signed), vmin=0, vmax=1, cmap=cmap); ax.set_title("abs")
    plt.suptitle(tag)
    plt.tight_layout()
    plt.savefig(os.path.join(out_folder, f"{tag}_heatmaps.png"), dpi=200)
    plt.close()

def choose_ds(T, min_samples=MIN_SAMPLES, max_ds=MAX_DS):
    """
    Choose integer ds >=1 such that ceil(T / ds) >= min_samples
    Equivalent: ds = floor(T / min_samples) but at least 1
    """
    if T <= 0:
        return 1
    ds = int(np.floor(T / float(min_samples)))
    if ds < 1:
        ds = 1
    if ds > max_ds:
        ds = max_ds
    return ds

def compute_env_and_ds(sig, fmin, fmax, fs, min_samples=MIN_SAMPLES):
    Xf = bandpass_filter(sig, fmin, fmax, fs)
    env = np.abs(hilbert(Xf, axis=1))  #(channels, T)
    T = env.shape[1]
    ds = choose_ds(T, min_samples)
    T_ds = int(np.ceil(T / float(ds)))
    env_ds = env[:, ::ds]
    return env_ds, ds, T_ds

sess_dirs = [d for d in glob.glob(os.path.join(base_dir, "*")) if os.path.isdir(d)]
session_map = {}
for sess in sess_dirs:
    sess_name = os.path.basename(sess)
    try:
        _ = int(sess_name)
    except:
        continue
    trial_files = sorted(glob.glob(os.path.join(sess, "trial*", "MatFiles", "*_allChannels.mat")))
    if len(trial_files) > 0:
        session_map[sess_name] = trial_files

if len(session_map) == 0:
    raise RuntimeError("No sessions found — check base_dir.")
    
per_trial_dir = os.path.join(save_dir, "per_trial")
per_session_dir = os.path.join(save_dir, "per_session")
agg_dir = os.path.join(save_dir, "aggregated")
diag_dir = os.path.join(save_dir, "diagnostics")
for d in (per_trial_dir, per_session_dir, agg_dir, diag_dir):
    os.makedirs(d, exist_ok=True)

if VERBOSE: print("STEP 1 — per-trial connectivity with adaptive ds")
trial_info_lines = []
for sess_name, tfiles in session_map.items():
    out_sess = os.path.join(per_trial_dir, sess_name)
    os.makedirs(out_sess, exist_ok=True)
    for tf in tfiles:
        try:
            S = load_signals_from_mat(tf)
        except Exception as e:
            print("skip (load error):", tf, e)
            continue
        if S.shape[0] < channels_keep:
            print("skip (too few channels):", tf); continue
        S = S[:channels_keep, :]
        for bname, (fmin, fmax) in bands.items():
            env_ds, ds_used, T_ds = compute_env_and_ds(S, fmin, fmax, fs)
            tag = os.path.splitext(os.path.basename(tf))[0] + f"_{bname}"
            if T_ds < MIN_SAMPLES:
                msg = f"TRIAL_SHORT: {sess_name}/{tag} T_ds={T_ds} ds={ds_used}"
                if VERBOSE: print("  warning:", msg)
                trial_info_lines.append(msg)
            R = np.corrcoef(env_ds)
            save_heatmaps_and_mats(out_sess, tag, R)

with open(os.path.join(diag_dir, "per_trial_ds_info.txt"), "w") as f:
    f.write("\n".join(trial_info_lines))

if VERBOSE: print("STEP 2 — per-session connectivity: mean_signal and concat_trials (both with adaptive ds per trial)")
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
    trials_trim = [t[:, :min_len] for t in trials]
    trials_arr = np.stack(trials_trim, axis=0)  # (n_trials, 64, min_len)
    mean_sig = np.mean(trials_arr, axis=0)      # (64, T_session)
    sess_out = os.path.join(per_session_dir, sess_name)
    os.makedirs(sess_out, exist_ok=True)
    per_session_results[sess_name] = {}

    for bname, (fmin, fmax) in bands.items():
        env_mean_ds, ds_mean, T_mean_ds = compute_env_and_ds(mean_sig, fmin, fmax, fs)
        R_from_mean = np.corrcoef(env_mean_ds)
        tag_mean = f"{sess_name}_{bname}_from_mean"
        save_heatmaps_and_mats(sess_out, tag_mean, R_from_mean)
        
        envs_ds_list = []
        ds_used_list = []
        short_trials = []
        for tr_idx, tr in enumerate(trials_trim):
            env_tr_ds, ds_tr, T_tr_ds = compute_env_and_ds(tr, fmin, fmax, fs)

            if T_tr_ds < MIN_SAMPLES:
                short_trials.append((tr_idx, T_tr_ds, ds_tr))
            envs_ds_list.append(env_tr_ds)
            ds_used_list.append(ds_tr)

        ds_concat = max(ds_used_list)
        
        envs_resampled = []
        for tr in range(len(trials_trim)):
        
            Xf_full = bandpass_filter(trials_trim[tr], fmin, fmax, fs)
            env_full = np.abs(hilbert(Xf_full, axis=1))
            env_ds_concat = env_full[:, ::ds_concat]
            envs_resampled.append(env_ds_concat)

        env_concat = np.concatenate(envs_resampled, axis=1)  #(64, sum_T_ds)
        T_concat = env_concat.shape[1]
        if T_concat < MIN_SAMPLES:
            if VERBOSE:
                print(f"  SESSION_SHORT: {sess_name} band {bname} concat T={T_concat} (ds_concat={ds_concat}) -> using mean-based R too")
            R_from_concat = R_from_mean.copy()
        else:
            R_from_concat = np.corrcoef(env_concat)
        tag_concat = f"{sess_name}_{bname}_from_concat_ds{ds_concat}"
        save_heatmaps_and_mats(sess_out, tag_concat, R_from_concat)

        per_session_results[sess_name][bname] = {
            "R_from_mean": R_from_mean,
            "R_from_concat": R_from_concat,
            "ds_mean": ds_mean,
            "ds_concat": ds_concat,
            "T_mean_ds": T_mean_ds,
            "T_concat": T_concat,
            "n_trials": len(trials_trim),
            "short_trials": short_trials
        }

np.save(os.path.join(per_session_dir, "per_session_results_summary.npy"), per_session_results)

if VERBOSE: print("STEP 3 — aggregated pre/post (use concat-derived R by default)")
pre_sessions = [s for s in per_session_results.keys() if int(s) < lesion_date]
post_sessions = [s for s in per_session_results.keys() if int(s) >= lesion_date]

agg_results = {}
for bname in bands.keys():
    agg_results[bname] = {}
    for mode_key in ("R_from_mean", "R_from_concat"):
        R_pre_list = [per_session_results[s][bname][mode_key] for s in pre_sessions if bname in per_session_results[s]]
        R_post_list = [per_session_results[s][bname][mode_key] for s in post_sessions if bname in per_session_results[s]]
        R_pre_mean = np.mean(np.stack(R_pre_list, axis=0), axis=0) if len(R_pre_list)>0 else None
        R_post_mean = np.mean(np.stack(R_post_list, axis=0), axis=0) if len(R_post_list)>0 else None
        agg_results[bname][mode_key] = {"R_pre_mean": R_pre_mean, "R_post_mean": R_post_mean,
                                       "n_pre": len(R_pre_list), "n_post": len(R_post_list)}
        
        if R_pre_mean is not None or R_post_mean is not None:
            plt.figure(figsize=(10,4))
            if R_pre_mean is not None:
                ax=plt.subplot(1,2,1); plt.imshow(R_pre_mean, vmin=-1, vmax=1, cmap=cmap); plt.title(f"{bname} PRE {mode_key}")
            if R_post_mean is not None:
                ax2=plt.subplot(1,2,2); plt.imshow(R_post_mean, vmin=-1, vmax=1, cmap=cmap); plt.title(f"{bname} POST {mode_key}")
            plt.suptitle(f"Aggregated {bname} {mode_key}")
            plt.tight_layout()
            plt.savefig(os.path.join(agg_dir, f"aggregated_{bname}_{mode_key}.png"), dpi=200)
            plt.close()

np.savez(os.path.join(agg_dir, "aggregated_results.npy"), agg_results=agg_results)

with open(os.path.join(diag_dir, "diagnostics_summary.txt"), "w") as df:
    df.write(f"MIN_SAMPLES = {MIN_SAMPLES}\n")
    df.write(f"Sessions found: {len(session_map)}\n")
    for sess_name, info in per_session_results.items():
        df.write(f"SESSION {sess_name}: n_bands={len(info.keys())}, n_trials={info[next(iter(info))]['n_trials']}\n")
        for bname, res in info.items():
            df.write(f"  {bname}: ds_mean={res['ds_mean']}, T_mean_ds={res['T_mean_ds']}, ds_concat={res['ds_concat']}, T_concat={res['T_concat']}, short_trials={len(res['short_trials'])}\n")

print("Done. Outputs saved to:", save_dir)
print("Check diagnostics in:", diag_dir)
