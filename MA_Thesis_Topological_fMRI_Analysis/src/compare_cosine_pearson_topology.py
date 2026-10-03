import numpy as np
import pandas as pd
import nibabel as nib
from scipy.stats import rankdata
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA  # <-- NEW IMPORT
import seaborn as sns
import matplotlib.pyplot as plt
from ripser import ripser
import networkx as nx
import os
from collections import defaultdict
import h5py
import cv2

# ==============================================================================
# === Configuration & Meta-Data ===
# ==============================================================================

ROI_LABELS = {1: 'V1', 2: 'MST', 3: 'V6', 4: 'V2', 5: 'V3', 6: 'V4'}

def get_roi_name(roi_id):
    return ROI_LABELS.get(roi_id, f"Unknown_ROI_{roi_id}")

# --- Gabor Configuration ---
GABOR_FREQS = [4, 8, 16, 32, 64]
GABOR_ANGLES_DEG = [0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5]
GABOR_ANGLES_RAD = np.deg2rad(GABOR_ANGLES_DEG)

LAMBDA_COLORS = {
    4: '#1f77b4',  8: '#ff7f0e',  16: '#2ca02c', 32: '#9467bd', 64: '#d62728'
}

# --- Save Configuration ---
SAVE_PLOTS = False 
SAVE_DIR = '/Users/lukatepic/Documents/Topological_Data_Analysis/stimuli/extracted_loops_gabor_ROI1'
os.makedirs(SAVE_DIR, exist_ok=True)

# ==============================================================================
# === 1. Gabor Filter Bank & Image Fetching ===
# ==============================================================================

def build_gabor_bank(frequencies, orientations):
    filters = []
    gamma = 0.5 
    for theta in orientations:
        for lambd in frequencies:
            sigma = 0.5 * lambd 
            ksize = (0, 0)
            kernel_cos = cv2.getGaborKernel(ksize, sigma, theta, lambd, gamma, 0, ktype=cv2.CV_32F)
            kernel_sin = cv2.getGaborKernel(ksize, sigma, theta, lambd, gamma, np.pi/2, ktype=cv2.CV_32F)
            kernel_cos -= np.mean(kernel_cos)
            kernel_sin -= np.mean(kernel_sin)
            kernel_cos /= (np.sum(np.abs(kernel_cos)) + 1e-8)
            kernel_sin /= (np.sum(np.abs(kernel_sin)) + 1e-8)
            filters.append((kernel_cos, kernel_sin, theta, lambd))
    return filters

def apply_filters(img, filters, pool_method='mean'):
    if len(img.shape) == 3 and img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
    img = np.float32(img)
    img = (img - np.mean(img)) / (np.std(img) + 1e-8)
    
    results = {}
    for k_cos, k_sin, theta, lambd in filters:
        fimg_cos = cv2.filter2D(img, cv2.CV_32F, k_cos)
        fimg_sin = cv2.filter2D(img, cv2.CV_32F, k_sin)
        magnitude = cv2.magnitude(fimg_cos, fimg_sin)
        energy = np.max(magnitude) if pool_method == 'max' else np.mean(magnitude)
        results[f"deg{np.degrees(theta):g}_lam{lambd}"] = energy
    return results

def fetch_image_from_hdf5(stim_id, hdf5_path, dset_name):
    with h5py.File(hdf5_path, 'r') as f:
        img_array = f[dset_name][stim_id, ...]
        if img_array.dtype != np.uint8:
            if np.issubdtype(img_array.dtype, np.floating):
                img_array = np.clip(img_array * 255.0, 0, 255).astype(np.uint8) if np.max(img_array) <= 1.1 else np.clip(img_array, 0, 255).astype(np.uint8)
            else:
                img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    return img_array

# ==============================================================================
# === 2. Data Loading, PCA & Similarity ===
# ==============================================================================

def create_presentation_mapping(responses_df):
    sorted_df = responses_df.sort_values(['SESSION', 'RUN', 'TRIAL']).copy()
    sorted_df['presentation_index'] = np.arange(len(sorted_df))
    return sorted_df

def standardize_voxels(beta_matrix):
    voxel_means = np.mean(beta_matrix, axis=1, keepdims=True)
    voxel_stds = np.std(beta_matrix, axis=1, keepdims=True)
    voxel_stds[voxel_stds == 0] = 1.0 
    return (beta_matrix - voxel_means) / voxel_stds

def load_roi_betas(sessions, roi_id, r2_threshold=50.0, standardize_level='run'):
    roi_name = get_roi_name(roi_id)
    print(f"\n=== Loading ROI {roi_id} ({roi_name}) | Sessions: {len(sessions)} total ===")
    try:
        roi_img = nib.load(f"rois/sub_01/HCP_MMP1.nii.gz")
        roi_mask = roi_img.get_fdata().astype(int) == roi_id
        
        r2_maps = [nib.load(f"betas/sub_01/R2_session{str(s).zfill(2)}.nii.gz").get_fdata() for s in sessions]
        avg_r2 = np.mean(np.stack(r2_maps, axis=-1), axis=-1)
        final_mask = roi_mask & (avg_r2 >= r2_threshold)
        n_voxels = np.sum(final_mask)
        if n_voxels == 0: raise ValueError(f"No voxels survived.")

        responses = pd.read_csv(f"responses_sub_01.tsv", sep='\t')
        sorted_df = create_presentation_mapping(responses)
        
        session_betas_list, loaded_global_indices = [], []

        for session in sessions:
            beta_path = f"betas/sub_01/betas_session{str(session).zfill(2)}.nii.gz"
            if not os.path.exists(beta_path): continue
                
            beta_data = nib.load(beta_path).get_fdata()
            flat_betas = beta_data[final_mask, :] / 300.0
            session_df = sorted_df[sorted_df['SESSION'] == session]
            
            if standardize_level == 'session':
                flat_betas = standardize_voxels(flat_betas)
            elif standardize_level == 'run':
                for run_id in session_df['RUN'].unique():
                    run_mask = (session_df['RUN'] == run_id).values
                    flat_betas[:, run_mask] = standardize_voxels(flat_betas[:, run_mask])
            
            session_betas_list.append(flat_betas)
            loaded_global_indices.extend(session_df['presentation_index'].values)

        return np.concatenate(session_betas_list, axis=1), np.array(loaded_global_indices), sorted_df
    except Exception as e:
        print(f"❌ Error: {e}")
        return None, None, None

def filter_for_repeated_stimuli(betas, loaded_indices, full_df, sessions, required_reps=3):
    df_subset = full_df[full_df['SESSION'].isin(sessions)].copy()
    stim_counts = df_subset['73KID'].value_counts()
    target_stim_ids = stim_counts[stim_counts == required_reps].index.tolist()
    if not target_stim_ids: return None, None, None
        
    target_mask = df_subset['73KID'].isin(target_stim_ids)
    target_trials = df_subset[target_mask].sort_values(by=['73KID', 'presentation_index']) 
    
    matrix_col_indices, valid_labels = [], []
    loaded_idx_map = {idx: i for i, idx in enumerate(loaded_indices)}
    
    for glob_idx, label in zip(target_trials['presentation_index'].values, target_trials['73KID'].values):
        if glob_idx in loaded_idx_map:
            matrix_col_indices.append(loaded_idx_map[glob_idx])
            valid_labels.append(label)
            
    return betas[:, matrix_col_indices], np.array(valid_labels)

# --- NEW PCA FUNCTION ---
def apply_pca_to_betas(beta_matrix, variance_threshold=0.90):
    """Applies PCA to the voxel dimensions to reduce noise and dimensionality."""
    print(f"\n=== Applying PCA (Target Variance: {variance_threshold*100}%) ===")
    pca = PCA(n_components=variance_threshold)
    reduced_betas = pca.fit_transform(beta_matrix.T)
    print(f"  -> Reduced voxels from {beta_matrix.shape[0]} to {pca.n_components_} principal components.")
    return reduced_betas.T

def compute_similarity_matrices(beta_matrix, labels, metric='pearson'):
    unique_labels = sorted(list(set(labels)))
    avg_betas = np.zeros((beta_matrix.shape[0], len(unique_labels)))
    label_to_idx = defaultdict(list)
    for i, label in enumerate(labels): label_to_idx[label].append(i)
    for i, label in enumerate(unique_labels):
        avg_betas[:, i] = np.mean(beta_matrix[:, label_to_idx[label]], axis=1)

    if metric == 'pearson': avg_sim = np.corrcoef(avg_betas.T)
    else: avg_sim = cosine_similarity(avg_betas.T)
    return avg_sim, unique_labels

def compute_tda(similarity_matrix):
    distance_matrix = np.maximum(1 - similarity_matrix, 0)
    np.fill_diagonal(distance_matrix, 0)
    try: return ripser(distance_matrix, maxdim=1, distance_matrix=True, do_cocycles=True)
    except Exception as e: return None

# ==============================================================================
# === 3. Topology Extraction & Comparison ===
# ==============================================================================

def extract_top_loops(ripser_result, similarity_matrix, stim_labels, percentile=95.0):
    """Extracts loop nodes logically without plotting them."""
    dgm1 = ripser_result['dgms'][1]
    cocycles = ripser_result['cocycles'][1]
    
    if len(dgm1) == 0: return []

    persistence = dgm1[:, 1] - dgm1[:, 0]
    thresh_val = np.percentile(persistence, percentile)
    top_loop_indices = [i for i, p in enumerate(persistence) if p >= thresh_val]
    top_loop_indices = sorted(top_loop_indices, key=lambda i: persistence[i], reverse=True)
    
    dist_matrix = np.maximum(1 - similarity_matrix, 0)
    np.fill_diagonal(dist_matrix, 0)
    num_nodes = dist_matrix.shape[0]
    
    extracted_loops = []
    
    for rank, loop_idx in enumerate(top_loop_indices):
        cocycle = cocycles[loop_idx]
        birth, death = dgm1[loop_idx]
        threshold = birth + 0.0001 
        
        G_birth = nx.Graph()
        for u in range(num_nodes):
            for v in range(u + 1, num_nodes):
                if dist_matrix[u, v] <= threshold: G_birth.add_edge(u, v)

        cocycle_edges = set((min(u, v), max(u, v)) for u, v, _ in cocycle)
        cycles = nx.cycle_basis(G_birth)
        valid_cycles = [c for c in cycles if len(set((min(c[k], c[(k+1)%len(c)]), max(c[k], c[(k+1)%len(c)])) for k in range(len(c))).intersection(cocycle_edges)) % 2 == 1]

        ordered_nodes = max(valid_cycles, key=len) if valid_cycles else list(G_birth.nodes())
        
        kids = [stim_labels[n] for n in ordered_nodes]
        extracted_loops.append({
            'rank': rank + 1,
            'persistence': death - birth,
            'ordered_nodes': ordered_nodes,
            'kids': kids,
            'kids_set': set(kids)
        })
        
    return extracted_loops

def compare_topologies(loops_a, name_a, loops_b, name_b):
    print(f"\n=== Comparing Topologies: {name_a} vs {name_b} ===")
    exact_matches = []
    partial_matches = []
    
    for loop_a in loops_a:
        set_a = loop_a['kids_set']
        for loop_b in loops_b:
            set_b = loop_b['kids_set']
            
            intersection = set_a.intersection(set_b)
            union = set_a.union(set_b)
            jaccard = len(intersection) / len(union) if len(union) > 0 else 0
            
            if set_a == set_b:
                exact_matches.append((loop_a['rank'], loop_b['rank'], set_a))
            elif jaccard >= 0.3: # Threshold to show meaningful overlaps
                partial_matches.append((loop_a['rank'], loop_b['rank'], jaccard, intersection, set_a, set_b))
                
    if exact_matches:
        print("\n✅ EXACT MATCHES FOUND:")
        for rank_a, rank_b, nodes in exact_matches:
            print(f"  -> {name_a} Rank {rank_a} perfectly matches {name_b} Rank {rank_b}")
            print(f"     Loop Size: {len(nodes)} stimuli")
            print(f"     Overlapping Stimuli IDs: {sorted(list(nodes))}\n")
    else:
        print("\n❌ NO EXACT MATCHES FOUND.")
        
    if partial_matches:
        print("\n⚠️ SIGNIFICANT PARTIAL OVERLAPS (Jaccard >= 0.30):")
        # Sort by highest jaccard similarity
        partial_matches.sort(key=lambda x: x[2], reverse=True)
        for rank_a, rank_b, jac, intersect, set_a, set_b in partial_matches:
            print(f"  -> {name_a} Rank {rank_a} partially matches {name_b} Rank {rank_b}")
            print(f"     Jaccard Similarity: {jac:.2f}")
            print(f"     {name_a} Loop Size: {len(set_a)} | {name_b} Loop Size: {len(set_b)}")
            print(f"     Number of Overlapping Stimuli: {len(intersect)}")
            print(f"     Overlapping Stimuli IDs: {sorted(list(intersect))}\n")
    else:
        print("\n❌ NO SIGNIFICANT PARTIAL OVERLAPS FOUND.")

# ==============================================================================
# === 4. Visualization ===
# ==============================================================================

def visualize_extracted_loops(extracted_loops, stim_labels, hdf5_path, dset_name, 
                              gabor_bank, roi_name, target_lambda, save_dir, metric_name):
    """Visualizes pre-extracted loops."""
    if not extracted_loops: return
    
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(extracted_loops))))
    NODE_RADIUS = 0.25 
    
    full_angles_rad = np.deg2rad(GABOR_ANGLES_DEG + [ang + 180 for ang in GABOR_ANGLES_DEG])
    angles_rad_closed = np.concatenate((full_angles_rad, [full_angles_rad[0]]))

    for loop_dict in extracted_loops:
        rank = loop_dict['rank']
        ordered_nodes = loop_dict['ordered_nodes']
        lifetime = loop_dict['persistence']
        loop_color = colors[rank - 1]
        
        loop_activations = {}
        loop_max_e = 1e-8
        
        for node in ordered_nodes:
            stim_id = stim_labels[node]
            img_array = fetch_image_from_hdf5(stim_id, hdf5_path, dset_name)
            activations = apply_filters(img_array, gabor_bank, pool_method='mean')
            loop_activations[node] = activations
            node_max = max([activations[f"deg{ang:g}_lam{target_lambda}"] for ang in GABOR_ANGLES_DEG])
            if node_max > loop_max_e: loop_max_e = node_max

        fig, ax = plt.subplots(figsize=(12, 12))
        pos = {node: (np.cos(2.0 * np.pi * k / len(ordered_nodes)), np.sin(2.0 * np.pi * k / len(ordered_nodes))) 
               for k, node in enumerate(ordered_nodes)}
            
        for k in range(len(ordered_nodes)):
            u, v = ordered_nodes[k], ordered_nodes[(k + 1) % len(ordered_nodes)]
            ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], color=loop_color, linewidth=3.0, alpha=0.5, zorder=1)

        for node in ordered_nodes:
            x, y = pos[node]
            base_energies = [loop_activations[node][f"deg{ang:g}_lam{target_lambda}"] for ang in GABOR_ANGLES_DEG]
            energies_closed = np.concatenate((base_energies + base_energies, [base_energies[0]]))
            
            axins = ax.inset_axes([x - NODE_RADIUS/2, y - NODE_RADIUS/2, NODE_RADIUS, NODE_RADIUS], transform=ax.transData, polar=True, zorder=5)
            plot_color = LAMBDA_COLORS.get(target_lambda, '#333333') 
            
            axins.plot(angles_rad_closed, energies_closed, color=plot_color, linewidth=2.0)
            axins.fill(angles_rad_closed, energies_closed, color=plot_color, alpha=0.3)
            axins.set_ylim(0, loop_max_e)
            axins.set_theta_zero_location('N') 
            axins.set_theta_direction(-1)
            axins.set_xticks([]); axins.set_yticks([]) 
            axins.spines['polar'].set_color(loop_color)
            axins.spines['polar'].set_linewidth(2.0)
            ax.text(x * 1.25, y * 1.25, str(stim_labels[node]), fontsize=11, fontweight='bold', ha='center', va='center')

        plt.title(f"{roi_name} ({metric_name}) - Rank {rank} Loop (Persistence: {lifetime:.3f})\n"
                  f"Spatial Freq: λ = {target_lambda}", fontsize=16, pad=20)
        ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5); ax.axis('equal'); ax.axis('off')
        
        if save_dir:
            lambda_folder = os.path.join(save_dir, metric_name, f"lam_{target_lambda}")
            os.makedirs(lambda_folder, exist_ok=True)
            file_name = f"{roi_name}_{metric_name}_Loop{rank}_lam{target_lambda}.png"
            plt.savefig(os.path.join(lambda_folder, file_name), dpi=150, bbox_inches='tight')
        plt.close(fig) # Prevent 100s of plots opening at once

# ==============================================================================
# === Main Execution ===
# ==============================================================================
if __name__ == "__main__":
    SESSIONS = list(range(1, 10)) 
    ROI_ID = 1       
    R2_THRESH = 50.0 
    REQUIRED_REPS = 3  
    STANDARDIZE_LVL = 'run'
    LOOP_PERCENTILE = 99.5  
    PLOT_SPECIFIC_LAMBDAS = [64, 32, 16, 8]  
    
    # --- PCA CONFIGURATION ---
    APPLY_PCA = True
    PCA_VARIANCE_THRESH = 0.90
    
    HDF5_FILE_PATH = '/Users/lukatepic/Documents/Topological_Data_Analysis/stimuli/nsd_stimuli.hdf5' 
    DATASET_NAME = 'imgBrick'

    roi_name = get_roi_name(ROI_ID)
    print("=== Initializing V1 Gabor Filter Bank ===")
    gabor_bank = build_gabor_bank(GABOR_FREQS, GABOR_ANGLES_RAD)
    
    raw_betas, loaded_indices, full_df = load_roi_betas(SESSIONS, ROI_ID, R2_THRESH, STANDARDIZE_LVL)
    
    if raw_betas is not None:
        clean_betas, stim_labels = filter_for_repeated_stimuli(raw_betas, loaded_indices, full_df, SESSIONS, REQUIRED_REPS)
        
        if clean_betas is not None:
            # === APPLY PCA ===
            if APPLY_PCA:
                clean_betas = apply_pca_to_betas(clean_betas, variance_threshold=PCA_VARIANCE_THRESH)

            # 1. Compute for BOTH metrics
            mat_avg_pearson, unique_names = compute_similarity_matrices(clean_betas, stim_labels, 'pearson')
            mat_avg_cosine, _ = compute_similarity_matrices(clean_betas, stim_labels, 'cosine')
            
            res_pearson = compute_tda(mat_avg_pearson)
            res_cosine = compute_tda(mat_avg_cosine)
            
            # 2. Extract logic (No plotting yet)
            print("\n🧠 Extracting Top Loops...")
            loops_pearson = extract_top_loops(res_pearson, mat_avg_pearson, unique_names, LOOP_PERCENTILE)
            loops_cosine = extract_top_loops(res_cosine, mat_avg_cosine, unique_names, LOOP_PERCENTILE)
            
            # 3. Compare them logically
            compare_topologies(loops_pearson, "Pearson", loops_cosine, "Cosine")
            
            # 4. Generate visual plots (Currently set to Cosine, but you can plot Pearson by passing loops_pearson)
            print(f"\n🎨 Generating plots for Cosine loops across lambdas: {PLOT_SPECIFIC_LAMBDAS}")
            for current_lambda in PLOT_SPECIFIC_LAMBDAS:
                visualize_extracted_loops(
                    loops_cosine, unique_names, HDF5_FILE_PATH, DATASET_NAME, 
                    gabor_bank, roi_name, current_lambda, SAVE_DIR, "Cosine"
                )
                
            print(f"\n🎉 Analysis Complete. Images saved in: {SAVE_DIR}")