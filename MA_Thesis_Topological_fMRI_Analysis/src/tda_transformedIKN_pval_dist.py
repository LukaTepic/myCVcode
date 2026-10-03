import numpy as np
import pandas as pd
import nibabel as nib
from scipy.stats import rankdata
from scipy.spatial.distance import pdist, squareform
from scipy.ndimage import shift
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
from sklearn.feature_selection import mutual_info_regression
from sklearn.manifold import MDS
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ripser import ripser
import networkx as nx
import plotly.graph_objects as go
import os
import h5py
import cv2
import sys
from collections import defaultdict

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("⚠️ WARNING: 'umap-learn' is not installed. Please run: pip install umap-learn for 3D visualizations.")

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

# Saliency Config
SALIENCY_GRID_SIZE = (32, 32) 
INVARIANCE_MODE = 'com'  # Toggle: 'spatial' (raw retinotopy), 'fourier' (math invariance), 'com' (physical centering)
#132, 133, 134, 136, 177
# fMRI Config
ROI_IDS = [132, 133, 134, 136, 177]  # IT composite
R2_THRESH = 50
STANDARDIZE_LVL = 'run'
FMRI_METRIC = 'pearson'  # Toggle: 'pearson', 'mutual_info', or 'distance_correlation'

SAVE_DIR = '/Users/lukatepic/Documents/Topological_Data_Analysis/stimuli/fmri_tda_saliency_final' 
os.makedirs(SAVE_DIR, exist_ok=True)

HDF5_FILE_PATH = '/Users/lukatepic/Documents/Topological_Data_Analysis/stimuli/nsd_stimuli.hdf5'
DATASET_NAME = 'imgBrick'

ROI_LABELS = {1: 'V1', 2: 'MST', 3: 'V6', 4: 'V2', 5: 'V3', 6: 'V4', 132: 'PIT', 133: 'CIT', 134: 'AIT'}
def get_roi_names(roi_ids): 
    return ", ".join([ROI_LABELS.get(r, f"ROI_{r}") for r in roi_ids])

# ==============================================================================
# === 1. Data Loading (Images & fMRI) ===
# ==============================================================================
def get_target_stimuli(tsv_path, sessions, required_reps=3):
    print(f"=== Filtering Stimuli (Sessions: {min(sessions)}-{max(sessions)}, Reps >= {required_reps}) ===")
    df = pd.read_csv(tsv_path, sep='\t')
    df_subset = df[df['SESSION'].isin(sessions)]
    stim_counts = df_subset['73KID'].value_counts()
    target_stim_ids = stim_counts[stim_counts >= required_reps].index.tolist()
    print(f" -> Found {len(target_stim_ids)} unique stimuli matching the criteria.")
    return sorted(target_stim_ids)

def fetch_image_from_hdf5(stim_id, hdf5_path, dset_name, force_rgb=True):
    with h5py.File(hdf5_path, 'r') as f:
        img_array = f[dset_name][stim_id, ...]
        if img_array.dtype != np.uint8:
            img_array = np.clip(img_array * 255.0, 0, 255).astype(np.uint8) if np.max(img_array) <= 1.1 else np.clip(img_array, 0, 255).astype(np.uint8)
        
        if force_rgb:
            if len(img_array.shape) == 2:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            elif img_array.shape[0] == 3:
                img_array = np.transpose(img_array, (1, 2, 0))
    return img_array

def standardize_voxels(beta_matrix):
    voxel_means = np.mean(beta_matrix, axis=1, keepdims=True)
    voxel_stds = np.std(beta_matrix, axis=1, keepdims=True)
    voxel_stds[voxel_stds == 0] = 1.0 
    return (beta_matrix - voxel_means) / voxel_stds

def load_roi_betas(sessions, roi_ids, r2_threshold=50.0, standardize_level='run'):
    roi_names = get_roi_names(roi_ids)
    print(f"\n=== Loading ROIs {roi_ids} ({roi_names}) | Sessions: {len(sessions)} total ===")
    
    roi_img = nib.load(f"rois/sub_01/HCP_MMP1.nii.gz")
    roi_data = roi_img.get_fdata().astype(int)
    roi_mask = np.isin(roi_data, roi_ids)
    
    r2_maps = [nib.load(f"betas/sub_01/R2_session{str(s).zfill(2)}.nii.gz").get_fdata() for s in sessions]
    avg_r2 = np.mean(np.stack(r2_maps, axis=-1), axis=-1)
    
    final_mask = roi_mask & (avg_r2 >= r2_threshold)
    print(f"  -> ✅ Voxel Filter: Extracted {np.sum(final_mask)} reliable voxels across {roi_names}.")

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
    print("\n=== Extracting & Averaging fMRI Signatures ===")
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
    unique_labels = sorted(list(set(valid_labels)))
    label_to_idx = defaultdict(list)
    for i, label in enumerate(valid_labels): label_to_idx[label].append(i)
        
    for label in unique_labels:
        avg_betas_dict[label] = np.mean(filtered_betas[:, label_to_idx[label]], axis=1)
        
    return avg_betas_dict

# ==============================================================================
# === 2. Saliency Extraction (Location Invariant Modes) ===
# ==============================================================================
def generate_ikn_saliency_map(img_rgb):
    if not IKN_AVAILABLE:
        raise ImportError("pySaliencyMap module not found.")
    
    height, width = img_rgb.shape[:2]
    sm = pySaliencyMap.pySaliencyMap(width, height)
    structural_saliency_map = sm.SMGetIntensityAndOrientation(img_rgb)
    return structural_saliency_map

def build_saliency_feature_matrix(stim_ids, hdf5_path, dset_name, grid_size, invariance_mode='fourier'):
    print(f"\n=== Extracting IKN Saliency Maps (Grid: {grid_size} | Mode: {invariance_mode.upper()}) ===")
    feature_list = []
    
    for count, stim_id in enumerate(stim_ids):
        if count % 100 == 0 and count > 0: print(f" -> Processed {count}/{len(stim_ids)} images...")
        img_rgb = fetch_image_from_hdf5(stim_id, hdf5_path, dset_name, force_rgb=True)
        
        smap_2d = generate_ikn_saliency_map(img_rgb)
        smap_resized = cv2.resize(smap_2d, grid_size, interpolation=cv2.INTER_AREA)
        
        if invariance_mode == 'fourier':
            # 2D FFT -> Shift zero-freq to center -> Get Magnitude -> Log Scale
            f_transform = np.fft.fft2(smap_resized)
            f_shift = np.fft.fftshift(f_transform)
            feature_map = np.log1p(np.abs(f_shift))
            
        elif invariance_mode == 'com':
            # Calculate Center of Mass and physically shift the matrix to the center
            total_mass = np.sum(smap_resized)
            if total_mass > 0:
                y_idx, x_idx = np.indices(grid_size)
                com_y = np.sum(y_idx * smap_resized) / total_mass
                com_x = np.sum(x_idx * smap_resized) / total_mass
                
                # Calculate required shift to hit the center
                dy = (grid_size[0] / 2.0) - com_y
                dx = (grid_size[1] / 2.0) - com_x
                
                # Shift and pad with zeros
                feature_map = shift(smap_resized, shift=(dy, dx), mode='constant', cval=0.0)
            else:
                feature_map = smap_resized
                
        else:
            # Default 'spatial' mode (retinotopic)
            feature_map = smap_resized
        
        if np.std(feature_map) > 0:
            feature_map = (feature_map - np.mean(feature_map)) / np.std(feature_map)
            
        feature_list.append(feature_map.flatten())
        
    return np.vstack(feature_list)

# ==============================================================================
# === 3. TDA Engine (H1 Loops & H2 Voids) ===
# ==============================================================================
def compute_image_topology(feature_matrix, maxdim=2):
    print(f"\n=== Computing TDA on Saliency Feature Space (maxdim={maxdim}) ===")
    dist_matrix = np.clip(cosine_distances(feature_matrix), 0, None)
    return ripser(dist_matrix, maxdim=maxdim, distance_matrix=True, do_cocycles=True), dist_matrix

def get_cycle_edges(cycle):
    return {
        (min(cycle[k], cycle[(k + 1) % len(cycle)]), max(cycle[k], cycle[(k + 1) % len(cycle)]))
        for k in range(len(cycle))
    }

def get_neighbor_rank_within_cycle(node, neighbor, cycle, dist_matrix):
    ranked_neighbors = sorted((candidate for candidate in cycle if candidate != node), key=lambda candidate: dist_matrix[node, candidate])
    return ranked_neighbors.index(neighbor) + 1

def report_h1_loop_sequence_diagnostics(cycle, dist_matrix, birth_threshold, label="H1"):
    if len(cycle) < 3:
        return

    neighbor_ranks = []
    edge_distances = []
    for idx, node in enumerate(cycle):
        prev_node = cycle[(idx - 1) % len(cycle)]
        next_node = cycle[(idx + 1) % len(cycle)]
        neighbor_ranks.append(get_neighbor_rank_within_cycle(node, prev_node, cycle, dist_matrix))
        neighbor_ranks.append(get_neighbor_rank_within_cycle(node, next_node, cycle, dist_matrix))
        edge_distances.append(dist_matrix[node, next_node])

    neighbor_ranks = np.asarray(neighbor_ranks)
    edge_distances = np.asarray(edge_distances)
    edge_count_at_birth = int(np.sum(edge_distances <= birth_threshold + 1e-9))

    print(f"\n--- {label} sequence diagnostic ---")
    print(f"Loop nodes: {len(cycle)}")
    print(f"Edges at birth threshold: {edge_count_at_birth}/{len(edge_distances)}")
    print(f"Plotted-neighbor ranks within this cycle: median={np.median(neighbor_ranks):.1f}, max={np.max(neighbor_ranks)}")
    print(f"Share of plotted neighbors in top 2/top 3/top 5: {np.mean(neighbor_ranks <= 2):.2%} / {np.mean(neighbor_ranks <= 3):.2%} / {np.mean(neighbor_ranks <= 5):.2%}")

def extract_h1_loop_nodes(ripser_result, dist_matrix, target_rank=1):
    dgm1, cocycles = ripser_result['dgms'][1], ripser_result['cocycles'][1]
    if len(dgm1) == 0: return []

    persistence = dgm1[:, 1] - dgm1[:, 0]
    if target_rank > len(persistence): return []

    loop_idx = np.argsort(persistence)[::-1][target_rank - 1]
    num_nodes = dist_matrix.shape[0]
    threshold = dgm1[loop_idx][0] + 0.0001 
    
    G_birth = nx.Graph()
    for u in range(num_nodes):
        for v in range(u + 1, num_nodes):
            if dist_matrix[u, v] <= threshold: G_birth.add_edge(u, v)

    cocycle_edges = set((min(u, v), max(u, v)) for u, v, _ in cocycles[loop_idx])
    valid_cycles = [c for c in nx.cycle_basis(G_birth) if len(get_cycle_edges(c).intersection(cocycle_edges)) % 2 == 1]
    if valid_cycles:
        selected_cycle = max(valid_cycles, key=len)
        report_h1_loop_sequence_diagnostics(selected_cycle, dist_matrix, threshold, label=f"H1 Rank {target_rank}")
        return selected_cycle

    print(f"\nWARNING: no cocycle-paired graph cycle found for H1 Rank {target_rank}; falling back to birth-threshold graph nodes.")
    print("WARNING: fallback node order is not guaranteed to be a valid sequential loop.")
    return list(G_birth.nodes())

def extract_h2_void_graph(ripser_result, target_rank=1):
    if len(ripser_result['dgms']) < 3: return None, None
    dgm2, cocycles2 = ripser_result['dgms'][2], ripser_result['cocycles'][2]
    if len(dgm2) == 0: return None, None

    persistence = dgm2[:, 1] - dgm2[:, 0]
    if target_rank > len(persistence): return None, None

    void_idx = np.argsort(persistence)[::-1][target_rank - 1]
    cocycle = cocycles2[void_idx]
    H2_graph = nx.Graph()
    
    for element in cocycle:
        u, v, w = int(element[0]), int(element[1]), int(element[2])
        H2_graph.add_edge(u, v)
        H2_graph.add_edge(v, w)
        H2_graph.add_edge(u, w)
        
    return list(H2_graph.nodes()), H2_graph

# ==============================================================================
# === 4. Shared fMRI Evaluation Functions ===
# ==============================================================================
def compute_distance_correlation(x, y):
    x = np.atleast_1d(x)[:, np.newaxis]
    y = np.atleast_1d(y)[:, np.newaxis]
    
    a = squareform(pdist(x))
    b = squareform(pdist(y))
    
    A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
    B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()
    
    n = x.shape[0]
    dcov2_xy = (A * B).sum() / float(n * n)
    dcov2_xx = (A * A).sum() / float(n * n)
    dcov2_yy = (B * B).sum() / float(n * n)
    
    if dcov2_xx > 0.0 and dcov2_yy > 0.0:
        return np.sqrt(dcov2_xy) / np.sqrt(np.sqrt(dcov2_xx) * np.sqrt(dcov2_yy))
    return 0.0

def build_fmri_similarity_matrix(nodes, stim_labels, fmri_betas_dict, metric='distance_correlation'):
    num_nodes = len(nodes)
    fmri_matrix = np.zeros((num_nodes, len(fmri_betas_dict[stim_labels[nodes[0]]])))
    for i, node in enumerate(nodes):
        fmri_matrix[i] = fmri_betas_dict[stim_labels[node]]

    fmri_sim_matrix = np.zeros((num_nodes, num_nodes))
    
    if metric == 'pearson':
        fmri_sim_matrix = np.corrcoef(fmri_matrix)
    elif metric == 'mutual_info':
        for i in range(num_nodes):
            X = fmri_matrix[i].reshape(-1, 1)
            for j in range(i, num_nodes):
                y = fmri_matrix[j]
                score = 0.0 if i == j else mutual_info_regression(X, y, n_neighbors=3, random_state=42)[0]
                fmri_sim_matrix[i, j] = fmri_sim_matrix[j, i] = score
        max_mi = np.max(fmri_sim_matrix)
        if max_mi > 0: fmri_sim_matrix = fmri_sim_matrix / max_mi
        np.fill_diagonal(fmri_sim_matrix, 1.0)
    elif metric == 'distance_correlation':
        for i in range(num_nodes):
            for j in range(i, num_nodes):
                score = 1.0 if i == j else compute_distance_correlation(fmri_matrix[i], fmri_matrix[j])
                fmri_sim_matrix[i, j] = fmri_sim_matrix[j, i] = score
    return fmri_sim_matrix

# ==============================================================================
# === 5. Landscape Mass Evaluation (Global Null Evaluation) ===
# ==============================================================================
def evaluate_persistence_landscape_h1(ripser_result, dist_matrix, target_stim_ids, fmri_signatures, n_top=100, n_permutations=2000, save_dir=None):
    print(f"\n=== Evaluating Top {n_top} H1 Saliency Loops (Global Random Loop Null) ===")
    
    dgm1, cocycles = ripser_result['dgms'][1], ripser_result['cocycles'][1]
    if len(dgm1) == 0:
        print("⚠️ No H1 loops found in the data.")
        return [], [], []
        
    print(" -> Precomputing global fMRI similarity matrix...")
    total_global_nodes = len(target_stim_ids)
    global_nodes = list(range(total_global_nodes))
    global_sim_matrix = build_fmri_similarity_matrix(global_nodes, target_stim_ids, fmri_signatures, metric=FMRI_METRIC)
        
    max_available = len(dgm1)
    n_test = min(n_top, max_available)
    persistence = dgm1[:, 1] - dgm1[:, 0]
    sorted_loop_indices = np.argsort(persistence)[::-1]
    
    print(f" -> Found {max_available} total loops. Running global permutations on the top {n_test}...")

    p_values, valid_ranks = [], []
    num_nodes_total = dist_matrix.shape[0]

    for rank in range(1, n_test + 1):
        loop_idx = sorted_loop_indices[rank - 1]
        threshold = dgm1[loop_idx][0] + 0.0001 
        G_birth = nx.Graph()
        for u in range(num_nodes_total):
            for v in range(u + 1, num_nodes_total):
                if dist_matrix[u, v] <= threshold: G_birth.add_edge(u, v)

        cocycle_edges = set((min(u, v), max(u, v)) for u, v, _ in cocycles[loop_idx])
        valid_cycles = [c for c in nx.cycle_basis(G_birth) if len(set((min(c[k], c[(k+1)%len(c)]), max(c[k], c[(k+1)%len(c)])) for k in range(len(c))).intersection(cocycle_edges)) % 2 == 1]
        loop_nodes = max(valid_cycles, key=len) if valid_cycles else list(G_birth.nodes())
        
        loop_size = len(loop_nodes)
        if loop_size < 3: 
            continue
            
        obs_corrs = [global_sim_matrix[loop_nodes[i], loop_nodes[(i + 1) % loop_size]] for i in range(loop_size)]
        observed_stat = np.mean(obs_corrs)
        
        null_distribution = np.zeros(n_permutations)
        for p in range(n_permutations):
            random_loop = np.random.choice(total_global_nodes, size=loop_size, replace=False)
            rand_corrs = [global_sim_matrix[random_loop[i], random_loop[(i + 1) % loop_size]] for i in range(loop_size)]
            null_distribution[p] = np.mean(rand_corrs)
            
        p_val = np.sum(null_distribution >= observed_stat) / n_permutations
        p_values.append(p_val)
        valid_ranks.append(rank)
        
        if rank % 25 == 0:
            print(f"   ... Processed {rank}/{n_test} loops.")

    if not p_values:
        print("No valid loops to evaluate.")
        return [], [], []
        
    significant_ranks = [r for r, p in zip(valid_ranks, p_values) if p <= 0.05]
    
    print("\n--- Final H1 Landscape Results (Uncorrected) ---")
    print(f"Total Loops Tested: {len(p_values)}")
    print(f"Significant Loops (Raw p <= 0.05): {len(significant_ranks)}")
    print(f"Significant Ranks: {significant_ranks}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(p_values, bins=np.linspace(0, 1, 51), color='royalblue', kde=False, ax=axes[0])
    axes[0].set_xlim(0, 1.0)
    axes[0].axvline(0.05, color='red', linestyle='dashed', linewidth=2, label='p = 0.05 threshold')
    axes[0].set_title(f'Distribution of Raw p-values (H1)\nMode: {INVARIANCE_MODE}', fontweight='bold')
    axes[0].set_xlabel('Raw p-value')
    axes[0].set_ylabel('Number of H1 Loops')
    axes[0].legend()
    
    colors = ['#FF4C4C' if p <= 0.05 else '#B0B0B0' for p in p_values]
    axes[1].scatter(valid_ranks, p_values, c=colors, alpha=0.8, edgecolor='black', s=50)
    axes[1].axhline(0.05, color='red', linestyle='dashed', linewidth=2, label='Alpha threshold (0.05)')
    axes[1].set_title(f'Raw p-values by Rank (H1)\nMode: {INVARIANCE_MODE}', fontweight='bold')
    axes[1].set_xlabel('Persistence Rank')
    axes[1].set_ylabel('Raw p-value')
    axes[1].set_yscale('log') 
    axes[1].invert_yaxis()
    axes[1].legend()
    
    plt.tight_layout()
    if save_dir: plt.savefig(os.path.join(save_dir, f"H1_Landscape_pvalues_{INVARIANCE_MODE}.png"), dpi=150)
    plt.show()
    
    return significant_ranks, valid_ranks, p_values

def evaluate_persistence_landscape_h2(ripser_result, dist_matrix, target_stim_ids, fmri_signatures, n_top=100, n_permutations=2000, save_dir=None):
    print(f"\n=== Evaluating Top {n_top} H2 Saliency Voids (Global Random Subgraph Null) ===")
    
    if len(ripser_result['dgms']) < 3:
        print("⚠️ No H2 components generated by Ripser!")
        return [], [], []
        
    dgm2, cocycles2 = ripser_result['dgms'][2], ripser_result['cocycles'][2]
    if len(dgm2) == 0:
        print("⚠️ No H2 voids found in the data.")
        return [], [], []
        
    print(" -> Precomputing global fMRI similarity matrix...")
    total_global_nodes = len(target_stim_ids)
    global_nodes = list(range(total_global_nodes))
    global_sim_matrix = build_fmri_similarity_matrix(global_nodes, target_stim_ids, fmri_signatures, metric=FMRI_METRIC)
        
    max_available = len(dgm2)
    n_test = min(n_top, max_available)
    persistence = dgm2[:, 1] - dgm2[:, 0]
    sorted_void_indices = np.argsort(persistence)[::-1]
    
    print(f" -> Found {max_available} total voids. Running global permutations on the top {n_test}...")

    p_values, valid_ranks = [], []

    for rank in range(1, n_test + 1):
        void_idx = sorted_void_indices[rank - 1]
        cocycle = cocycles2[void_idx]
        H2_graph = nx.Graph()
        
        for element in cocycle:
            u, v, w = int(element[0]), int(element[1]), int(element[2])
            H2_graph.add_edge(u, v)
            H2_graph.add_edge(v, w)
            H2_graph.add_edge(u, w)
            
        void_nodes = list(H2_graph.nodes())
        num_nodes = len(void_nodes)
        
        if num_nodes < 4: 
            continue
            
        edges = list(H2_graph.edges())
        node_to_idx = {node: i for i, node in enumerate(void_nodes)}
        edge_indices = [(node_to_idx[u], node_to_idx[v]) for u, v in edges]
        
        obs_corrs = [global_sim_matrix[u, v] for u, v in edges]
        observed_stat = np.mean(obs_corrs)
        
        null_distribution = np.zeros(n_permutations)
        
        for p in range(n_permutations):
            random_nodes = np.random.choice(total_global_nodes, size=num_nodes, replace=False)
            rand_corrs = [global_sim_matrix[random_nodes[i], random_nodes[j]] for i, j in edge_indices]
            null_distribution[p] = np.mean(rand_corrs)
            
        p_val = np.sum(null_distribution >= observed_stat) / n_permutations
        p_values.append(p_val)
        valid_ranks.append(rank)
        
        if rank % 25 == 0:
            print(f"   ... Processed {rank}/{n_test} voids.")

    if not p_values:
        print("No valid voids to evaluate.")
        return [], [], []
        
    significant_ranks = [r for r, p in zip(valid_ranks, p_values) if p <= 0.05]
    
    print("\n--- Final H2 Landscape Results (Uncorrected) ---")
    print(f"Total Voids Tested: {len(p_values)}")
    print(f"Significant Voids (Raw p <= 0.05): {len(significant_ranks)}")
    print(f"Significant Ranks: {significant_ranks}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(p_values, bins=np.linspace(0, 1, 51), color='mediumseagreen', kde=False, ax=axes[0])
    axes[0].set_xlim(0, 1.0)
    axes[0].axvline(0.05, color='red', linestyle='dashed', linewidth=2, label='p = 0.05 threshold')
    axes[0].set_title(f'Distribution of Raw p-values (H2)\nMode: {INVARIANCE_MODE}', fontweight='bold')
    axes[0].set_xlabel('Raw p-value')
    axes[0].set_ylabel('Number of H2 Voids')
    axes[0].legend()
    
    colors = ['#FF4C4C' if p <= 0.05 else '#B0B0B0' for p in p_values]
    axes[1].scatter(valid_ranks, p_values, c=colors, alpha=0.8, edgecolor='black', s=50)
    axes[1].axhline(0.05, color='red', linestyle='dashed', linewidth=2, label='Alpha threshold (0.05)')
    axes[1].set_title(f'Raw p-values by Rank (H2)\nMode: {INVARIANCE_MODE}', fontweight='bold')
    axes[1].set_xlabel('Persistence Rank')
    axes[1].set_ylabel('Raw p-value')
    axes[1].set_yscale('log') 
    axes[1].invert_yaxis()
    axes[1].legend()
    
    plt.tight_layout()
    if save_dir: plt.savefig(os.path.join(save_dir, f"H2_Landscape_pvalues_{INVARIANCE_MODE}.png"), dpi=150)
    plt.show()
    
    return significant_ranks, valid_ranks, p_values

# ==============================================================================
# === 6. Visualization Tooling ===
# ==============================================================================
def visualize_saliency_loop(ordered_nodes, stim_labels, hdf5_path, dset_name, save_dir, target_rank=1):
    print(f" -> Generating Saliency Map Overlay Plot for Rank {target_rank} Loop...")
    num_nodes_loop = len(ordered_nodes)
    NODE_BOX_SIZE = 1.6  
    
    if num_nodes_loop > 1:
        min_dist = NODE_BOX_SIZE * 2.0  
        circle_radius = max(3.5, min_dist / (2 * np.sin(np.pi / num_nodes_loop)))
    else: circle_radius = 3.5
        
    fig_size = min(45, max(15, int(circle_radius * 3.5)))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    pos, node_angles = {}, {}
    
    for k, node in enumerate(ordered_nodes):
        angle = 2.0 * np.pi * k / num_nodes_loop
        node_angles[node] = angle
        pos[node] = (circle_radius * np.cos(angle), circle_radius * np.sin(angle))
        
    for k in range(num_nodes_loop):
        u, v = ordered_nodes[k], ordered_nodes[(k + 1) % num_nodes_loop]
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], color='#888888', linewidth=3.0, alpha=0.4, zorder=1)

    for i, node in enumerate(ordered_nodes):
        img_rgb = fetch_image_from_hdf5(stim_labels[node], hdf5_path, dset_name, force_rgb=True)
        
        smap_raw = generate_ikn_saliency_map(img_rgb)
        smap_norm = cv2.normalize(smap_raw, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        heatmap = cv2.applyColorMap(smap_norm, cv2.COLORMAP_JET)
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        img_gray_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
        
        overlay = cv2.addWeighted(heatmap, 0.7, img_gray_bgr, 0.3, 0)
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        
        x, y = pos[node]
        axins = ax.inset_axes([x - NODE_BOX_SIZE/2, y - NODE_BOX_SIZE/2, NODE_BOX_SIZE, NODE_BOX_SIZE], transform=ax.transData, zorder=5)
        axins.imshow(overlay_rgb)
        axins.axis('off')
        
        rect = patches.Rectangle((0, 0), overlay_rgb.shape[1], overlay_rgb.shape[0], linewidth=3, edgecolor='white', facecolor='none', zorder=10)
        axins.add_patch(rect)

        offset = NODE_BOX_SIZE/2 + 0.15
        text_x = x + offset * np.cos(node_angles[node])
        text_y = y + offset * np.sin(node_angles[node])
        
        ha_align = 'left' if np.cos(node_angles[node]) > 0.1 else ('right' if np.cos(node_angles[node]) < -0.1 else 'center')
        va_align = 'bottom' if np.sin(node_angles[node]) > 0.1 else ('top' if np.sin(node_angles[node]) < -0.1 else 'center')

        ax.text(text_x, text_y, f"ID: {stim_labels[node]}", ha=ha_align, va=va_align, 
                fontsize=max(14, int(fig_size*0.5)), fontweight='bold', 
                bbox=dict(facecolor='white', alpha=0.85, edgecolor='black', boxstyle='round,pad=0.3'), zorder=15)

    ax.set_xlim(-circle_radius - NODE_BOX_SIZE*2, circle_radius + NODE_BOX_SIZE*2)
    ax.set_ylim(-circle_radius - NODE_BOX_SIZE*2, circle_radius + NODE_BOX_SIZE*2)
    ax.axis('equal'); ax.axis('off')
    plt.savefig(os.path.join(save_dir, f"H1_Physical_Loop_{INVARIANCE_MODE}_Rank{target_rank}.png"), dpi=150, bbox_inches='tight')
    plt.show()

def visualize_h2_fmri_network(H2_graph, void_nodes, stim_labels, fmri_sim_matrix, node_to_idx, save_dir, target_rank=1):
    print(f" -> Generating H2 Network Isomorphism Plot for Rank {target_rank} Void...")
    
    fig, ax = plt.subplots(figsize=(12, 10))
    pos = nx.spring_layout(H2_graph, seed=42)
    edges = H2_graph.edges()
    edge_colors = [fmri_sim_matrix[node_to_idx[u], node_to_idx[v]] for u, v in edges]
        
    nx.draw_networkx_nodes(H2_graph, pos, node_size=700, node_color='lightgray', edgecolors='black', ax=ax)
    
    nx.draw_networkx_edges(
        H2_graph, pos, edgelist=edges, 
        edge_color=edge_colors, 
        edge_cmap=plt.cm.viridis, 
        width=3.0,
        edge_vmin=min(edge_colors),
        edge_vmax=max(edge_colors),
        ax=ax
    )
    
    labels = {node: str(stim_labels[node]) for node in void_nodes}
    nx.draw_networkx_labels(H2_graph, pos, labels, font_size=10, font_weight="bold", ax=ax)
    
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=min(edge_colors), vmax=max(edge_colors)))
    sm.set_array([])
    
    cbar = fig.colorbar(sm, ax=ax, shrink=0.7, aspect=20, pad=0.05)
    cbar.set_label('fMRI Response Similarity (Edges)', fontweight='bold', fontsize=12)
    
    ax.set_title(f"H2 Void Topology Colored by fMRI Correlation\nRank {target_rank} ({INVARIANCE_MODE})", fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')
    plt.tight_layout()
    
    if save_dir: 
        plt.savefig(os.path.join(save_dir, f"H2_Network_Isomorphism_{INVARIANCE_MODE}_Rank{target_rank}.png"), dpi=150)
    plt.show()

# ==============================================================================
# === Main Execution ===
# ==============================================================================
if __name__ == "__main__":
    try:
        # 1. Data Prep
        target_stim_ids = get_target_stimuli(TSV_FILE_PATH, SESSIONS, REQUIRED_REPS)
        if not target_stim_ids: exit("⚠️ No stimuli met criteria.")
            
        # Build features using the specified invariance mode at the top of the script
        saliency_features = build_saliency_feature_matrix(
            target_stim_ids, HDF5_FILE_PATH, DATASET_NAME, SALIENCY_GRID_SIZE, invariance_mode=INVARIANCE_MODE
        )
        
        raw_betas, loaded_indices, full_df = load_roi_betas(SESSIONS, ROI_IDS, R2_THRESH, STANDARDIZE_LVL)
        fmri_signatures = extract_averaged_fmri_signatures(raw_betas, loaded_indices, full_df, SESSIONS, target_stim_ids)
        
        # 2. Compute TDA (maxdim=2 captures H0, H1, H2)
        ripser_res, dist_matrix = compute_image_topology(saliency_features, maxdim=2)
        
        # ==========================================
        # 3a. Process H1 Loops
        # ==========================================
        sig_ranks_h1, valid_ranks_h1, p_values_h1 = evaluate_persistence_landscape_h1(
            ripser_result=ripser_res, 
            dist_matrix=dist_matrix, 
            target_stim_ids=target_stim_ids, 
            fmri_signatures=fmri_signatures, 
            n_top=500,           
            n_permutations=10000, 
            save_dir=SAVE_DIR
        )

        target_h1_rank = sig_ranks_h1[0] if sig_ranks_h1 else 1
        print(f"\n>>> Visualizing Physics for Target H1 Rank: {target_h1_rank}")
        loop_nodes = extract_h1_loop_nodes(ripser_res, dist_matrix, target_h1_rank)
        if loop_nodes:
            visualize_saliency_loop(loop_nodes, target_stim_ids, HDF5_FILE_PATH, DATASET_NAME, SAVE_DIR, target_h1_rank)

        # ==========================================
        # 3b. Process H2 Voids
        # ==========================================
        sig_ranks_h2, valid_ranks_h2, p_values_h2 = evaluate_persistence_landscape_h2(
            ripser_result=ripser_res, 
            dist_matrix=dist_matrix, 
            target_stim_ids=target_stim_ids, 
            fmri_signatures=fmri_signatures, 
            n_top=500,           
            n_permutations=10000,        
            save_dir=SAVE_DIR
        )

        target_h2_rank = sig_ranks_h2[0] if sig_ranks_h2 else 1
        print(f"\n>>> Visualizing Physics for Target H2 Rank: {target_h2_rank}")
        void_nodes, H2_graph = extract_h2_void_graph(ripser_res, target_rank=target_h2_rank)
        if void_nodes and H2_graph:
            h2_sim_matrix = build_fmri_similarity_matrix(void_nodes, target_stim_ids, fmri_signatures, metric=FMRI_METRIC)
            node_to_idx = {node: i for i, node in enumerate(void_nodes)}
            visualize_h2_fmri_network(H2_graph, void_nodes, target_stim_ids, h2_sim_matrix, node_to_idx, SAVE_DIR, target_rank=target_h2_rank)

    except Exception as e: 
        import traceback
        traceback.print_exc()
        print(f"\n❌ An error occurred: {e}")
