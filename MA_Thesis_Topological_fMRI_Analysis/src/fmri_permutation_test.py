import numpy as np
import pandas as pd
import nibabel as nib
from scipy.stats import rankdata
import seaborn as sns
import matplotlib.pyplot as plt
import os

# ==============================================================================
# === Configuration & Meta-Data ===
# ==============================================================================

SESSIONS = list(range(1, 10))
# We need at least 2 presentations in DIFFERENT runs for this to work, 
# 3 is safer to ensure we have cross-run data.
REQUIRED_REPS = 3 
TSV_FILE_PATH = 'responses_sub_01.tsv'

# fMRI Config
ROI_ID = 1
R2_THRESH = 0
STANDARDIZE_LVL = 'run'

SAVE_DIR = '/Users/lukatepic/Documents/Topological_Data_Analysis/stimuli/fmri_reliability_test' 
os.makedirs(SAVE_DIR, exist_ok=True)

ROI_LABELS = {1: 'V1', 2: 'MST', 3: 'V6', 4: 'V2', 5: 'V3', 6: 'V4'}
def get_roi_name(roi_id): return ROI_LABELS.get(roi_id, f"Unknown_ROI_{roi_id}")

# ==============================================================================
# === 1. Data Loading ===
# ==============================================================================

def get_target_stimuli(tsv_path, sessions, required_reps=3):
    print(f"=== Filtering Stimuli (Sessions: {min(sessions)}-{max(sessions)}, Reps >= {required_reps}) ===")
    df = pd.read_csv(tsv_path, sep='\t')
    df_subset = df[df['SESSION'].isin(sessions)]
    stim_counts = df_subset['73KID'].value_counts()
    target_stim_ids = stim_counts[stim_counts >= required_reps].index.tolist()
    print(f" -> Found {len(target_stim_ids)} unique stimuli matching the criteria.")
    return sorted(target_stim_ids)

def standardize_voxels(beta_matrix):
    voxel_means = np.mean(beta_matrix, axis=1, keepdims=True)
    voxel_stds = np.std(beta_matrix, axis=1, keepdims=True)
    voxel_stds[voxel_stds == 0] = 1.0 
    return (beta_matrix - voxel_means) / voxel_stds

def load_roi_betas(sessions, roi_id, r2_threshold=50.0, standardize_level='run'):
    roi_name = get_roi_name(roi_id)
    print(f"\n=== Loading ROI {roi_id} ({roi_name}) | Sessions: {len(sessions)} total ===")
    
    roi_img = nib.load(f"rois/sub_01/HCP_MMP1.nii.gz")
    roi_mask = roi_img.get_fdata().astype(int) == roi_id
    
    r2_maps = [nib.load(f"betas/sub_01/R2_session{str(s).zfill(2)}.nii.gz").get_fdata() for s in sessions]
    avg_r2 = np.mean(np.stack(r2_maps, axis=-1), axis=-1)
    final_mask = roi_mask & (avg_r2 >= r2_threshold)
    print(f"  -> ✅ Voxel Filter: Extracted {np.sum(final_mask)} reliable voxels.")

    responses = pd.read_csv(TSV_FILE_PATH, sep='\t')
    sorted_df = responses.sort_values(['SESSION', 'RUN', 'TRIAL']).copy()
    sorted_df['presentation_index'] = np.arange(len(sorted_df))
    
    session_betas_list, loaded_global_indices = [], []

    for session in sessions:
        beta_path = f"betas/sub_01/betas_session{str(session).zfill(2)}.nii.gz"
        if not os.path.exists(beta_path): continue
            
        flat_betas = nib.load(beta_path).get_fdata()[final_mask, :] / 300.0
        session_df = sorted_df[sorted_df['SESSION'] == session]
        
        if standardize_level == 'run':
            for run_id in session_df['RUN'].unique():
                run_mask = (session_df['RUN'] == run_id).values
                flat_betas[:, run_mask] = standardize_voxels(flat_betas[:, run_mask])
        
        session_betas_list.append(flat_betas)
        loaded_global_indices.extend(session_df['presentation_index'].values)

    return np.concatenate(session_betas_list, axis=1), np.array(loaded_global_indices), sorted_df

def extract_trial_level_data(raw_betas, loaded_indices, full_df, sessions, target_stim_ids):
    print("\n=== Extracting Trial-Level fMRI Signatures ===")
    df_subset = full_df[full_df['SESSION'].isin(sessions)].copy()
    
    # Create a unique run identifier
    df_subset['UNIQUE_RUN'] = df_subset['SESSION'] * 100 + df_subset['RUN']
    
    target_mask = df_subset['73KID'].isin(target_stim_ids)
    target_trials = df_subset[target_mask].sort_values(by=['73KID', 'presentation_index']) 
    
    loaded_idx_map = {idx: i for i, idx in enumerate(loaded_indices)}
    
    valid_cols, valid_labels, valid_runs = [], [], []
    for glob_idx, label, run in zip(target_trials['presentation_index'].values, 
                                    target_trials['73KID'].values, 
                                    target_trials['UNIQUE_RUN'].values):
        if glob_idx in loaded_idx_map:
            valid_cols.append(loaded_idx_map[glob_idx])
            valid_labels.append(label)
            valid_runs.append(run)
            
    # Extract betas and transpose so shape is (n_trials, n_voxels)
    trial_betas = raw_betas[:, valid_cols].T
    
    print(f"  -> Extracted {trial_betas.shape[0]} total trials across {len(set(valid_labels))} unique stimuli.")
    return trial_betas, np.array(valid_labels), np.array(valid_runs)

# ==============================================================================
# === 2. Cross-Run Reliability Test ===
# ==============================================================================

def run_cross_run_reliability_test(trial_betas, stim_labels, run_labels, n_permutations=10000, save_dir=None):
    print(f"\n==========================================================")
    print(f"=== CROSS-RUN RELIABILITY PERMUTATION TEST ({n_permutations} shuffles) ===")
    print(f"==========================================================")
    
    n_trials = trial_betas.shape[0]

    # 1. Compute full trial-by-trial Pearson similarity matrix
    print("  -> Computing trial-by-trial correlation matrix...")
    sim_matrix = np.corrcoef(trial_betas)

    # 2. Precompute Structural Masks
    same_run_mask = run_labels[:, None] == run_labels[None, :]
    same_stim_mask = stim_labels[:, None] == stim_labels[None, :]

    # Our mathematical targets: Same Image, but DIFFERENT Runs
    target_mask = same_stim_mask & ~same_run_mask
    
    total_valid_pairs = np.sum(target_mask)
    print(f"  -> Identified {total_valid_pairs} valid cross-run repetition pairs.")
    if total_valid_pairs == 0:
        print("⚠️ No cross-run repetitions found. Check your stimulus/run labels.")
        return

    # 3. 🔥 MASSIVE OPTIMIZATION: Pre-compute the Rank Matrix ONCE 🔥
    print("  -> Pre-computing cross-run rank matrix...")
    rank_matrix = np.full((n_trials, n_trials), np.nan)
    
    for i in range(n_trials):
        # Identify trials not in the same run
        valid_idx = np.where(~same_run_mask[i])[0] 
        if len(valid_idx) > 0:
            valid_sims = sim_matrix[i, valid_idx]
            # Rank descending (highest correlation = rank 1)
            rank_matrix[i, valid_idx] = rankdata(-valid_sims)

    # 4. Calculate the Observed Statistic
    # We simply take the mean of the pre-computed ranks where our targets are
    observed_stat = np.nanmean(rank_matrix[target_mask])
    avg_pool_size = np.mean([np.sum(~same_run_mask[i]) for i in range(n_trials)])
    print(f"  -> Observed Mean Cross-Run Rank: {observed_stat:.2f} (out of an avg pool of {avg_pool_size:.0f} trials)")

    # 5. Build the Null Distribution (Permutation)
    print("  -> Building Null Distribution (Vectorized)...")
    null_distribution = np.zeros(n_permutations)
    base_stim_sequence = stim_labels.copy()
    
    for p in range(n_permutations):
        # Shuffle stimulus labels
        shuffled_stim_labels = np.random.permutation(base_stim_sequence)
        
        # Find where the "fake" identical stimuli are across different runs
        shuffled_same_stim_mask = shuffled_stim_labels[:, None] == shuffled_stim_labels[None, :]
        shuffled_target_mask = shuffled_same_stim_mask & ~same_run_mask
        
        # Vectorized lookup of the pre-computed ranks
        null_distribution[p] = np.nanmean(rank_matrix[shuffled_target_mask])

    # 6. Calculate P-Value
    beating_shuffles = np.sum(null_distribution <= observed_stat)
    p_value = beating_shuffles / n_permutations

    print(f"  -> Random shuffles scoring equal or better: {beating_shuffles}")
    print(f"  -> P-Value: {p_value:.5f}")
    
    if p_value < 0.05:
        print("  -> ✅ SIGNIFICANT: The ROI consistently encodes specific images across runs.")
    else:
        print("  -> ⚠️ NOT SIGNIFICANT: Cross-run patterns for the same image are indistinguishable from random.")

    # 7. Plotting
    plt.figure(figsize=(10, 6))
    sns.histplot(null_distribution, bins=30, kde=True, color='lightcoral', stat='density')
    plt.axvline(observed_stat, color='red', linestyle='dashed', linewidth=2.5, 
                label=f'Observed (Mean Rank = {observed_stat:.2f}, p={p_value:.4f})')
    
    roi_name = get_roi_name(ROI_ID)
    plt.title(f"Cross-Run Reliability Permutation Test: ROI {ROI_ID} ({roi_name})", fontsize=14, fontweight='bold')
    plt.xlabel("Mean Similarity Rank")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, f"Reliability_Test_ROI_{ROI_ID}.png"), dpi=300)
    plt.show()

    return p_value, null_distribution

# ==============================================================================
# === Main Execution ===
# ==============================================================================
if __name__ == "__main__":
    try:
        # 1. Prepare Target Stimuli
        target_stim_ids = get_target_stimuli(TSV_FILE_PATH, SESSIONS, REQUIRED_REPS)
        if not target_stim_ids: 
            exit("⚠️ No stimuli met criteria.")
            
        # 2. Load Corresponding fMRI Data
        raw_betas, loaded_indices, full_df = load_roi_betas(SESSIONS, ROI_ID, R2_THRESH, STANDARDIZE_LVL)
        
        if raw_betas is not None:
            # 3. Extract trial-level data and labels
            trial_betas, stim_labels, run_labels = extract_trial_level_data(
                raw_betas, loaded_indices, full_df, SESSIONS, target_stim_ids
            )
            
            # 4. Run the Reliability Test
            run_cross_run_reliability_test(
                trial_betas=trial_betas, 
                stim_labels=stim_labels, 
                run_labels=run_labels,
                n_permutations=10000,
                save_dir=SAVE_DIR
            )

    except Exception as e: 
        print(f"\n❌ An error occurred: {e}")
