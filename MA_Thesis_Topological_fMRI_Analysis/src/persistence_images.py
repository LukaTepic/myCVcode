import numpy as np
import pandas as pd
import nibabel as nib
from sklearn.metrics.pairwise import cosine_distances
from scipy.stats import pearsonr
import seaborn as sns
import matplotlib.pyplot as plt
from ripser import ripser
from persim import PersistenceImager
import os
import h5py
import cv2
import sys
from collections import defaultdict

# ==============================================================================
# === 0. IKN Saliency Setup ===
# ==============================================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
saliency_path = os.path.join(script_dir, 'pySaliencyMap')
sys.path.append(saliency_path)

try:
    import pySaliencyMap
    IKN_AVAILABLE = True
except ImportError:
    print(f"⚠️ WARNING: Could not import 'pySaliencyMap'. Looked in: {saliency_path}")
    IKN_AVAILABLE = False

# ==============================================================================
# === Configuration & Meta-Data ===
# ==============================================================================

SESSIONS = list(range(1, 10))
REQUIRED_REPS = 3
TSV_FILE_PATH = 'responses_sub_01.tsv'

# Gabor Config (4x4 Grid for Spatial Biological Accuracy)
GABOR_FREQS = [4, 8, 16, 32, 64]
GABOR_ANGLES_DEG = [0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5]
GABOR_ANGLES_RAD = np.deg2rad(GABOR_ANGLES_DEG)
GRID_SIZE = (4, 4)

# Saliency Config
SALIENCY_GRID_SIZE = (32, 32) 

# fMRI Config
ROI_ID = 1
R2_THRESH = 50.0
STANDARDIZE_LVL = 'run'

# PI Resolution Config
PI_PIXELS = 50 
PI_SPREAD = 0.001  

SAVE_DIR = '/Users/lukatepic/Documents/Topological_Data_Analysis/stimuli/fmri_tda_persistence_images_final' 
os.makedirs(SAVE_DIR, exist_ok=True)

HDF5_FILE_PATH = '/Users/lukatepic/Documents/Topological_Data_Analysis/stimuli/nsd_stimuli.hdf5'
DATASET_NAME = 'imgBrick'

ROI_LABELS = {1: 'V1', 2: 'MST', 3: 'V6', 4: 'V2', 5: 'V3', 6: 'V4'}
def get_roi_name(roi_id): return ROI_LABELS.get(roi_id, f"Unknown_ROI_{roi_id}")

# ==============================================================================
# === 1. fMRI Loading & Standardization ===
# ==============================================================================

def get_target_stimuli(tsv_path, sessions, required_reps=3):
    df = pd.read_csv(tsv_path, sep='\t')
    df_subset = df[df['SESSION'].isin(sessions)]
    stim_counts = df_subset['73KID'].value_counts()
    return sorted(stim_counts[stim_counts >= required_reps].index.tolist())

def fetch_image_from_hdf5(stim_id, hdf5_path, dset_name, force_rgb=True):
    with h5py.File(hdf5_path, 'r') as f:
        img_array = f[dset_name][stim_id, ...]
        if img_array.dtype != np.uint8:
            img_array = np.clip(img_array * 255.0, 0, 255).astype(np.uint8) if np.max(img_array) <= 1.1 else np.clip(img_array, 0, 255).astype(np.uint8)
        if force_rgb:
            if len(img_array.shape) == 2: img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            elif img_array.shape[0] == 3: img_array = np.transpose(img_array, (1, 2, 0))
    return img_array

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

def extract_averaged_fmri_signatures(raw_betas, loaded_indices, full_df, sessions, target_stim_ids):
    df_subset = full_df[full_df['SESSION'].isin(sessions)]
    target_mask = df_subset['73KID'].isin(target_stim_ids)
    target_trials = df_subset[target_mask].sort_values(by=['73KID', 'presentation_index']) 
    
    loaded_idx_map = {idx: i for i, idx in enumerate(loaded_indices)}
    
    valid_cols, valid_labels = [], []
    for glob_idx, label in zip(target_trials['presentation_index'].values, target_trials['73KID'].values):
        if glob_idx in loaded_idx_map:
            valid_cols.append(loaded_idx_map[glob_idx])
            valid_labels.append(label)
            
    filtered_betas = raw_betas[:, valid_cols]
    avg_betas_dict = {}
    label_to_idx = defaultdict(list)
    for i, label in enumerate(valid_labels): label_to_idx[label].append(i)
        
    for label in sorted(list(set(valid_labels))):
        avg_betas_dict[label] = np.mean(filtered_betas[:, label_to_idx[label]], axis=1)
        
    return avg_betas_dict

# ==============================================================================
# === 2. Gabor & Saliency Feature Extraction ===
# ==============================================================================

def build_gabor_bank(frequencies, orientations):
    filters = []
    for theta in orientations:
        for lambd in frequencies:
            cv2_theta = theta + (np.pi / 2.0) 
            k_cos = cv2.getGaborKernel((0, 0), 0.5 * lambd, cv2_theta, lambd, 0.5, 0, ktype=cv2.CV_32F)
            k_sin = cv2.getGaborKernel((0, 0), 0.5 * lambd, cv2_theta, lambd, 0.5, np.pi/2, ktype=cv2.CV_32F)
            k_cos -= np.mean(k_cos); k_sin -= np.mean(k_sin)
            k_cos /= (np.sum(np.abs(k_cos)) + 1e-8); k_sin /= (np.sum(np.abs(k_sin)) + 1e-8) 
            filters.append((k_cos, k_sin, theta, lambd))
    return filters

def build_gabor_feature_matrix(stim_ids, hdf5_path, dset_name, gabor_bank, grid_size):
    print(f"\n-> Extracting Gabor Pyramids (Grid: {grid_size})...")
    feature_list = []
    for stim_id in stim_ids:
        img = fetch_image_from_hdf5(stim_id, hdf5_path, dset_name, force_rgb=False)
        if len(img.shape) == 3: img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = (np.float32(img) - np.mean(img)) / (np.std(img) + 1e-8)
        
        h, w = img.shape
        ch, cw = h // grid_size[0], w // grid_size[1]
        vec = []
        for k_cos, k_sin, _, _ in gabor_bank:
            mag = cv2.magnitude(cv2.filter2D(img, cv2.CV_32F, k_cos), cv2.filter2D(img, cv2.CV_32F, k_sin))
            for i in range(grid_size[0]):
                for j in range(grid_size[1]):
                    vec.append(np.sqrt(np.mean(mag[i*ch:(i+1)*ch, j*cw:(j+1)*cw])))
        feature_list.append(vec)
    return np.vstack(feature_list) 

def build_saliency_feature_matrix(stim_ids, hdf5_path, dset_name, grid_size):
    print(f"-> Extracting IKN Saliency Maps (Grid: {grid_size})...")
    feature_list = []
    for stim_id in stim_ids:
        img_rgb = fetch_image_from_hdf5(stim_id, hdf5_path, dset_name, force_rgb=True)
        sm = pySaliencyMap.pySaliencyMap(img_rgb.shape[1], img_rgb.shape[0])
        smap_2d = sm.SMGetSM(img_rgb)
        
        smap_resized = cv2.resize(smap_2d, grid_size, interpolation=cv2.INTER_AREA)
        if np.std(smap_resized) > 0:
            smap_resized = (smap_resized - np.mean(smap_resized)) / np.std(smap_resized)
            
        feature_list.append(smap_resized.flatten())
    return np.vstack(feature_list) 

# ==============================================================================
# === 3. TDA Math & Outlier Pruning ===
# ==============================================================================

def remove_extreme_outliers(dgm, multiplier=3.0):
    """ 
    Uses Tukey's IQR method to prune rogue loops that artificially 
    stretch the Min-Max scaling boundaries.
    """
    if len(dgm) < 4: return dgm 
    
    # Check for extreme anomalies in Birth coordinates
    q1_b, q3_b = np.percentile(dgm[:, 0], [25, 75])
    iqr_b = q3_b - q1_b
    upper_b = q3_b + multiplier * iqr_b
    
    # Check for extreme anomalies in Death coordinates
    q1_d, q3_d = np.percentile(dgm[:, 1], [25, 75])
    iqr_d = q3_d - q1_d
    upper_d = q3_d + multiplier * iqr_d
    
    # Keep only points within the extreme upper bounds
    mask = (dgm[:, 0] <= upper_b) & (dgm[:, 1] <= upper_d)
    
    removed = len(dgm) - np.sum(mask)
    if removed > 0:
        print(f"      -> Pruned {removed} extreme outlier point(s).")
        
    return dgm[mask]

def scale_diagram(dgm):
    """ Scales diagram so births start at 0 and max death is 1. """
    if len(dgm) == 0: return dgm
    min_birth = np.min(dgm[:, 0])
    max_death = np.max(dgm[:, 1])
    if max_death == min_birth: return dgm
    return (dgm - min_birth) / (max_death - min_birth)

# ==============================================================================
# === 4. Main Pipeline ===
# ==============================================================================

def run_pipeline():
    target_stim_ids = get_target_stimuli(TSV_FILE_PATH, SESSIONS, REQUIRED_REPS)
    if not target_stim_ids: return print("No stimuli found.")

    # 1. Image Extraction
    gabor_bank = build_gabor_bank(GABOR_FREQS, GABOR_ANGLES_RAD)
    feat_gab = build_gabor_feature_matrix(target_stim_ids, HDF5_FILE_PATH, DATASET_NAME, gabor_bank, GRID_SIZE)
    feat_sal = build_saliency_feature_matrix(target_stim_ids, HDF5_FILE_PATH, DATASET_NAME, SALIENCY_GRID_SIZE)

    # 2. fMRI Extraction
    raw_betas, loaded_indices, full_df = load_roi_betas(SESSIONS, ROI_ID, R2_THRESH, STANDARDIZE_LVL)
    fmri_dict = extract_averaged_fmri_signatures(raw_betas, loaded_indices, full_df, SESSIONS, target_stim_ids)
    feat_fmri = np.array([fmri_dict[stim] for stim in target_stim_ids])

    # 3. Distance Matrices
    print("\n-> Computing Distance Matrices & TDA...")
    dist_gab = np.clip(cosine_distances(feat_gab), 0, None)
    dist_sal = np.clip(cosine_distances(feat_sal), 0, None)
    dist_fmri = np.clip(1.0 - np.corrcoef(feat_fmri), 0, None) 

    # 4. Ripser (Extract H1 Diagrams)
    dgm_gab = ripser(dist_gab, maxdim=1, distance_matrix=True)['dgms'][1]
    dgm_sal = ripser(dist_sal, maxdim=1, distance_matrix=True)['dgms'][1]
    dgm_fmri = ripser(dist_fmri, maxdim=1, distance_matrix=True)['dgms'][1]

    dgm_gab = dgm_gab[dgm_gab[:, 1] != np.inf]
    dgm_sal = dgm_sal[dgm_sal[:, 1] != np.inf]
    dgm_fmri = dgm_fmri[dgm_fmri[:, 1] != np.inf]

    # 5. Clean Outliers
    print("-> Pruning Extreme Outliers before Scaling...")
    print("   [Gabor]")
    dgm_gab = remove_extreme_outliers(dgm_gab)
    print("   [Saliency]")
    dgm_sal = remove_extreme_outliers(dgm_sal)
    print("   [fMRI]")
    dgm_fmri = remove_extreme_outliers(dgm_fmri)

    # 6. Scale the Diagrams (Removes absolute noise-floor differences)
    print("-> Applying Min-Max Scaling...")
    dgm_gab_scaled = scale_diagram(dgm_gab)
    dgm_sal_scaled = scale_diagram(dgm_sal)
    dgm_fmri_scaled = scale_diagram(dgm_fmri)

    # 7. Transform to (Birth, Persistence) space for visual matching
    vis_gab = np.column_stack([dgm_gab_scaled[:, 0], dgm_gab_scaled[:, 1] - dgm_gab_scaled[:, 0]]) if len(dgm_gab_scaled) > 0 else []
    vis_sal = np.column_stack([dgm_sal_scaled[:, 0], dgm_sal_scaled[:, 1] - dgm_sal_scaled[:, 0]]) if len(dgm_sal_scaled) > 0 else []
    vis_fmri = np.column_stack([dgm_fmri_scaled[:, 0], dgm_fmri_scaled[:, 1] - dgm_fmri_scaled[:, 0]]) if len(dgm_fmri_scaled) > 0 else []

    # 8. Configure Persistence Imager
    print("-> Generating Persistence Images...")
    imager = PersistenceImager(pixel_size=1.0/PI_PIXELS, birth_range=(0, 1), pers_range=(0, 1))
    
    # CRITICAL: Square the spread to correctly configure the Gaussian Variance (Sigma^2)
    imager.kernel_params = {'sigma': [[PI_SPREAD**2, 0.0], [0.0, PI_SPREAD**2]]}

    # CRITICAL: Do NOT use imager.fit() here to prevent matplotlib rubber-banding
    pi_gab = imager.transform(dgm_gab_scaled)
    pi_sal = imager.transform(dgm_sal_scaled)
    pi_fmri = imager.transform(dgm_fmri_scaled)

    # 9. Cross-Modality Correlation
    r_gab_fmri, _ = pearsonr(pi_gab.flatten(), pi_fmri.flatten())
    r_sal_fmri, _ = pearsonr(pi_sal.flatten(), pi_fmri.flatten())
    
    print("\n===========================================")
    print("=== TOPOLOGICAL ISOMORPHISM RESULTS ===")
    print("===========================================")
    print(f"fMRI <-> Gabor Correlation:    r = {r_gab_fmri:.4f}")
    print(f"fMRI <-> Saliency Correlation: r = {r_sal_fmri:.4f}")

    # ==========================================================================
    # === 10. Plotting ===
    # ==========================================================================
    # Fetch the dynamic ROI name using the ID selected at the top of the script
    roi_name = get_roi_name(ROI_ID)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("Topological Persistence Isomorphism (Birth-Persistence Format)\n" + 
                 f"fMRI vs Gabor (r={r_gab_fmri:.3f})  |  fMRI vs Saliency (r={r_sal_fmri:.3f})", 
                 fontsize=18, fontweight='bold', y=0.98)

    # Make the third title dynamic based on the selected ROI
    titles = ["Gabor Features", "IKN Saliency Maps", f"{roi_name} fMRI Betas"]
    
    vis_dgms = [vis_gab, vis_sal, vis_fmri]
    pis = [pi_gab, pi_sal, pi_fmri]

    # Row 1: Persistence Diagrams (Scatter Format)
    for i in range(3):
        ax = axes[0, i]
        if len(vis_dgms[i]) > 0:
            ax.scatter(vis_dgms[i][:, 0], vis_dgms[i][:, 1], alpha=0.6, color='dodgerblue', edgecolor='k', s=30)
        
        ax.set_title(f"{titles[i]} (H1 Diagram)", fontsize=14, pad=10, wrap=True)
        ax.set_xlabel("Scaled Birth")
        ax.set_ylabel("Scaled Persistence")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, linestyle='--', alpha=0.5)

    # Row 2: Persistence Images (Heatmap Format)
    for i in range(3):
        ax = axes[1, i]
        im = ax.imshow(pis[i].T, cmap='inferno', origin='lower', 
                       extent=[0, 1, 0, 1])
        ax.set_title(f"{titles[i]} (Persistence Image)", fontsize=14, pad=10, wrap=True)
        ax.set_xlabel("Scaled Birth")
        ax.set_ylabel("Scaled Persistence")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout(rect=[0, 0, 1, 0.95], pad=2.0)
    plt.savefig(
        os.path.join(SAVE_DIR, f"Persistence_Image_Correlations_Cleaned_{roi_name}.png"),
        dpi=200,
        bbox_inches='tight',
        pad_inches=0.3
    )
    plt.show()

if __name__ == "__main__":
    run_pipeline()
