# ---------- DIAGNOSTIC QUICK CHECK ----------
import numpy as np
import matplotlib.pyplot as plt
import os

diag_dir = os.path.join(save_dir, "diagnostics")
os.makedirs(diag_dir, exist_ok=True)

def print_stats(name, M):
    print(f"--- {name} ---")
    print("shape:", M.shape)
    print("min, 1pct, 5pct, median, mean, 95pct, 99pct, max:",
          np.nanpercentile(M, [0,1,5,50,100*0.5,95,99,100]))
    print("mean +/- std:", np.nanmean(M), np.nanstd(M))
    print("nans:", np.isnan(M).sum())
    print()

# Choose one band to inspect (e.g. beta)
inspect_band = "beta"
fmin, fmax = bands[inspect_band]

# Build envelopes for first few pre and post trials (reuse code)
def compute_env_for_trials(arr):
    envs = []
    for tr in range(arr.shape[0]):
        X = arr[tr]  # (64, T)
        Xf = bandpass_filter(X, fmin, fmax, fs)
        env = np.abs(hilbert(Xf, axis=1))
        envs.append(env)
    return envs

pre_envs = compute_env_for_trials(pre_arr)
post_envs = compute_env_for_trials(post_arr)

print("n pre trials, n post trials:", len(pre_envs), len(post_envs))
# sample one trial
sample_pre = pre_envs[0]
sample_post = post_envs[0]
print("sample_env shapes (channels, time):", sample_pre.shape, sample_post.shape)

# downsample factor used in pipeline
ds = 10
sample_pre_ds = sample_pre[:, ::ds]
sample_post_ds = sample_post[:, ::ds]
print("after ds:", sample_pre_ds.shape)

# 1) Check variance across channels (is any channel zero / tiny variance?)
var_per_channel_pre = np.var(sample_pre_ds, axis=1)
var_per_channel_post = np.var(sample_post_ds, axis=1)
print_stats("var_per_channel_pre (example trial)", var_per_channel_pre)
print_stats("var_per_channel_post (example trial)", var_per_channel_post)

# 2) Are envelopes almost identical across channels? compute pairwise channel differences
mean_pairwise_corr_pre = np.mean([np.corrcoef(sample_pre_ds)[i,j] 
                                  for i in range(sample_pre_ds.shape[0]) for j in range(i+1, sample_pre_ds.shape[0])])
mean_pairwise_corr_post = np.mean([np.corrcoef(sample_post_ds)[i,j] 
                                   for i in range(sample_post_ds.shape[0]) for j in range(i+1, sample_post_ds.shape[0])])
print("Mean pairwise corr (pre example trial):", mean_pairwise_corr_pre)
print("Mean pairwise corr (post example trial):", mean_pairwise_corr_post)

# 3) Distribution of corr values for the trial
R_pre = np.corrcoef(sample_pre_ds)
R_post = np.corrcoef(sample_post_ds)
print_stats("R_pre (example trial)", R_pre.flatten())
print_stats("R_post (example trial)", R_post.flatten())

# 4) Plot a few envelopes to visual check (first 6 channels)
plt.figure(figsize=(10,6))
tvec = np.arange(sample_pre_ds.shape[1]) * (ds / fs)
for ch in range(6):
    plt.plot(tvec, sample_pre_ds[ch], label=f"ch{ch+1}", alpha=0.8)
plt.title(f"Envelopes pre trial (first 6 channels) - {inspect_band}")
plt.xlabel("Time (s)")
plt.savefig(os.path.join(diag_dir, f"env_pre_first6_{inspect_band}.png"), dpi=200)
plt.close()

plt.figure(figsize=(10,6))
for ch in range(6):
    plt.plot(tvec, sample_post_ds[ch], label=f"ch{ch+1}", alpha=0.8)
plt.title(f"Envelopes post trial (first 6 channels) - {inspect_band}")
plt.xlabel("Time (s)")
plt.savefig(os.path.join(diag_dir, f"env_post_first6_{inspect_band}.png"), dpi=200)
plt.close()

# 5) Check if env signals share large common mean across channels (global component)
global_pre = np.mean(sample_pre_ds, axis=0)
global_post = np.mean(sample_post_ds, axis=0)
print("Global signal (pre) mean/std:", np.mean(global_pre), np.std(global_pre))
print("Global signal (post) mean/std:", np.mean(global_post), np.std(global_post))

# correlation of each channel with global
corr_with_global_pre = [np.corrcoef(sample_pre_ds[ch], global_pre)[0,1] for ch in range(sample_pre_ds.shape[0])]
corr_with_global_post = [np.corrcoef(sample_post_ds[ch], global_post)[0,1] for ch in range(sample_post_ds.shape[0])]
print_stats("corr_with_global_pre", np.array(corr_with_global_pre))
print_stats("corr_with_global_post", np.array(corr_with_global_post))

# Save arrays for inspection
np.save(os.path.join(diag_dir, f"R_pre_example_{inspect_band}.npy"), R_pre)
np.save(os.path.join(diag_dir, f"R_post_example_{inspect_band}.npy"), R_post)

print("Diagnostics saved to:", diag_dir)
