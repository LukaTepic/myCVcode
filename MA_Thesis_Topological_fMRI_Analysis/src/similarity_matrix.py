import numpy as np
import pandas as pd
import nibabel as nib
import seaborn as sns
import matplotlib.pyplot as plt
import os
from collections import defaultdict
from scipy.stats import rankdata

# ==============================================================================
# === Configuration & Meta-Data ===
# ==============================================================================

# Map ROI IDs to their anatomical names
ROI_LABELS = {
    1: 'V1', 
    2: 'MST', 
    3: 'V6', 
    4: 'V2', 
    5: 'V3', 
    6: 'V4'
}

def get_roi_name(roi_id):
    """Helper to safely fetch the ROI name, defaulting to 'Unknown' if not in dict."""
    return ROI_LABELS.get(roi_id, f"Unknown_ROI_{roi_id}")

# ==============================================================================
# === 1. Data Loading & Index Management ===
# ==============================================================================

def create_presentation_mapping(responses_df):
    """Assigns a unique global 'presentation_index' to every row."""
    sorted_df = responses_df.sort_values(['SESSION', 'RUN', 'TRIAL']).copy()
    sorted_df['presentation_index'] = np.arange(len(sorted_df))
    return sorted_df

def standardize_voxels(beta_matrix):
    """
    Z-scores each voxel's responses across the selected trials.
    Robust to zero variance to prevent NaNs.
    """
    voxel_means = np.mean(beta_matrix, axis=1, keepdims=True)
    voxel_stds = np.std(beta_matrix, axis=1, keepdims=True)
    voxel_stds[voxel_stds == 0] = 1.0 
    return (beta_matrix - voxel_means) / voxel_stds

def load_roi_betas(sessions, roi_id, r2_threshold=50.0, standardize_level='run'):
    roi_name = get_roi_name(roi_id)
    print(f"\n=== 🧠 Loading ROI {roi_id} ({roi_name}) ===")
    
    try:
        # --- 1. Load ROI Mask & Voxel Diagnostics ---
        roi_img = nib.load(f"rois/sub_01/HCP_MMP1.nii.gz")
        roi_data = roi_img.get_fdata()
        
        roi_mask = roi_data.astype(int) == roi_id
        voxels_in_roi_orig = np.sum(roi_mask)
        
        # --- 2. Load R2 and Filter ---
        r2_maps = [nib.load(f"betas/sub_01/R2_session{str(s).zfill(2)}.nii.gz").get_fdata() for s in sessions]
        avg_r2 = np.mean(np.stack(r2_maps, axis=-1), axis=-1)
        final_mask = roi_mask & (avg_r2 >= r2_threshold)
        n_voxels_final = np.sum(final_mask)

        print(f"  -> 📏 Voxel Count: {voxels_in_roi_orig} (Anatomical) -> {n_voxels_final} (R2-Filtered)")

        # --- 3. Load Metadata & Sessions ---
        responses = pd.read_csv(f"responses_sub_01.tsv", sep='\t')
        sorted_df = create_presentation_mapping(responses)
        
        session_betas_list = []
        loaded_global_indices = []

        for session in sessions:
            beta_path = f"betas/sub_01/betas_session{str(session).zfill(2)}.nii.gz"
            beta_data = nib.load(beta_path).get_fdata()
            
            # Metadata for this session
            session_df = sorted_df[sorted_df['SESSION'] == session]
            run_counts = session_df['RUN'].value_counts().sort_index()
            num_runs = len(run_counts)
            
            # --- Diagnostic Print for Session/Run Structure ---
            print(f"  -> 📅 Session {session}: Total Trials = {beta_data.shape[3]} | Runs = {num_runs}")
            run_lengths_str = ", ".join([f"R{run}: {count}t" for run, count in run_counts.items()])
            print(f"     └─ Run Lengths: {run_lengths_str}")
            
            # Extract voxels and process
            flat_betas = beta_data[final_mask, :] / 300.0
            
            if standardize_level == 'run':
                for run_id in run_counts.index:
                    run_mask = (session_df['RUN'] == run_id).values
                    flat_betas[:, run_mask] = standardize_voxels(flat_betas[:, run_mask])
            elif standardize_level == 'session':
                flat_betas = standardize_voxels(flat_betas)
            
            session_betas_list.append(flat_betas)
            loaded_global_indices.extend(session_df['presentation_index'].values)

        # --- 4. Final Combination ---
        full_betas = np.concatenate(session_betas_list, axis=1)
        
        print(f"\n✅ COMBINED DATASET COMPLETE")
        print(f"   Shape: {full_betas.shape[0]} voxels x {full_betas.shape[1]} total trials")
        print("-" * 50)
        
        return full_betas, np.array(loaded_global_indices), sorted_df

    except Exception as e:
        print(f"❌ Error: {e}")
        return None, None, None

# ==============================================================================
# === 2. Logic for "Repeated Stimuli" Selection ===
# ==============================================================================

def filter_for_repeated_stimuli(betas, loaded_indices, full_df, sessions, max_stimuli=100):
    print(f"\n=== Filtering for Repeated Stimuli ===")
    
    df_subset = full_df[full_df['SESSION'].isin(sessions)].copy()
    stim_counts = df_subset['73KID'].value_counts()
    repeated_ids = stim_counts[stim_counts > 1].index.tolist()
    
    if not repeated_ids:
        print("⚠️ No repeated stimuli found in these sessions.")
        return None, None
        
    target_stim_ids = repeated_ids[:max_stimuli]
    print(f"  -> Found {len(repeated_ids)} repeated stimuli. Analyzing top {len(target_stim_ids)}.")

    target_mask = df_subset['73KID'].isin(target_stim_ids)
    target_trials = df_subset[target_mask].sort_values('73KID') 
    
    target_global_indices = target_trials['presentation_index'].values
    target_labels = target_trials['73KID'].values
    
    matrix_col_indices = []
    valid_labels = []
    loaded_idx_map = {idx: i for i, idx in enumerate(loaded_indices)}
    
    for glob_idx, label in zip(target_global_indices, target_labels):
        if glob_idx in loaded_idx_map:
            matrix_col_indices.append(loaded_idx_map[glob_idx])
            valid_labels.append(label)
            
    filtered_betas = betas[:, matrix_col_indices]
    
    print(f"  -> 📊 Final Cleaned Matrix Shape: {filtered_betas.shape} (Voxels x Filtered Trials)")
    return filtered_betas, np.array(valid_labels)

# ==============================================================================
# === 3. Similarity Matrix Manipulations ===
# ==============================================================================

def compute_similarity_matrices(beta_matrix, labels, metric='pearson'):
    print(f"\n=== Computing Similarity Matrices (Metric: {metric}) ===")
    
    if metric == 'pearson':
        full_sim = np.corrcoef(beta_matrix.T)
    elif metric == 'spearman':
        ranks = np.apply_along_axis(rankdata, 0, beta_matrix)
        full_sim = np.corrcoef(ranks.T)
    else:
        raise ValueError("Unsupported metric")
        
    unique_labels = sorted(list(set(labels)))
    n_unique = len(unique_labels)
    avg_sim = np.zeros((n_unique, n_unique))
    
    label_to_idx = defaultdict(list)
    for i, label in enumerate(labels):
        label_to_idx[label].append(i)
    
    for i, label_a in enumerate(unique_labels):
        for j, label_b in enumerate(unique_labels):
            idx_a = label_to_idx[label_a]
            idx_b = label_to_idx[label_b]
            sub_grid = full_sim[np.ix_(idx_a, idx_b)]
            
            if i == j:
                n = sub_grid.shape[0]
                if n > 1:
                    avg_sim[i, j] = (np.sum(sub_grid) - n) / (n * (n - 1))
                else:
                    avg_sim[i, j] = 1.0 
            else:
                avg_sim[i, j] = np.mean(sub_grid)

    z_sim = np.zeros_like(avg_sim)
    for r in range(n_unique):
        row_vals = avg_sim[r, :]
        mask = np.ones(n_unique, dtype=bool)
        mask[r] = False
        other_vals = row_vals[mask]
        
        if np.std(other_vals) > 0:
            z_sim[r, :] = (row_vals - np.mean(other_vals)) / np.std(other_vals)
        else:
            z_sim[r, :] = row_vals
            
    print(f"  -> 🧮 Full Trial Matrix Shape: {full_sim.shape}")
    print(f"  -> 🧮 Averaged Stimulus Matrix Shape: {avg_sim.shape}")
    
    return full_sim, avg_sim, z_sim, unique_labels

# ==============================================================================
# === Main Execution ===
# ==============================================================================

if __name__ == "__main__":
    # --- Configuration ---
    SESSIONS = [1,2] 
    ROI_ID = 1            
    R2_THRESH = 50.0 
    MAX_STIM = 50
    METRIC = 'pearson'
    STANDARDIZE_LVL = 'run'  # Options: 'run', 'session', None
    
    roi_name = get_roi_name(ROI_ID)
    
    # Step 1: Load raw data (now with drift correction!)
    raw_betas, loaded_indices, full_df = load_roi_betas(
        SESSIONS, ROI_ID, R2_THRESH, standardize_level=STANDARDIZE_LVL
    )
    
    if raw_betas is not None:
        # Step 2: Filter specifically for repeated stimuli
        clean_betas, stim_labels = filter_for_repeated_stimuli(
            raw_betas, loaded_indices, full_df, SESSIONS, max_stimuli=MAX_STIM
        )
        
        if clean_betas is not None:
            # Step 3: Compute Similarity & Manipulations
            mat_full, mat_avg, mat_z, unique_names = compute_similarity_matrices(
                clean_betas, stim_labels, metric=METRIC
            )
            
            # Step 4: Visualize
            print("\n=== Generating Plots ===")
            
            # Determine correct color bounds: if standardized, Pearson ranges from -1 to 1.
            vmin_val = -1 if STANDARDIZE_LVL is not None else 0

            plt.figure(figsize=(11, 9), dpi=180)
            ax = sns.heatmap(
                mat_full,
                cmap='viridis',
                vmin=vmin_val,
                vmax=1,
                square=True,
                cbar_kws={'shrink': 0.9, 'pad': 0.02}
            )
            ax.set_title(f"Trial-wise Similarity Matrix | {roi_name}", fontsize=15, fontweight='bold', pad=12)
            ax.set_xlabel("Trials")
            ax.set_ylabel("Trials")
            ax.tick_params(axis='both', labelsize=8)
            plt.tight_layout(pad=1.2)

            plt.figure(figsize=(11, 9), dpi=180)
            ax = sns.heatmap(
                mat_avg,
                cmap='viridis',
                vmin=vmin_val,
                vmax=1,
                square=True,
                cbar_kws={'shrink': 0.9, 'pad': 0.02}
            )
            ax.set_title(f"Averaged Stimulus Matrix | {roi_name}", fontsize=15, fontweight='bold', pad=12)
            ax.set_xlabel("Unique Stimuli ID")
            ax.set_ylabel("Unique Stimuli ID")
            ax.tick_params(axis='both', labelsize=8)
            plt.tight_layout(pad=1.2)
            
            plt.figure(figsize=(11, 9), dpi=180)
            ax = sns.heatmap(
                mat_z,
                cmap='RdBu_r',
                center=0,
                square=True,
                cbar_kws={'shrink': 0.9, 'pad': 0.02}
            )
            ax.set_title(f"Row-wise Z-Scored Similarity | {roi_name}", fontsize=15, fontweight='bold', pad=12)
            ax.set_xlabel("Unique Stimuli ID")
            ax.set_ylabel("Unique Stimuli ID")
            ax.tick_params(axis='both', labelsize=8)
            plt.tight_layout(pad=1.2)

            print("✅ Displaying plots...")
            plt.show()
            
            print("\n✅ Analysis pipeline complete.")
