import numpy as np
import pandas as pd
import nibabel as nib
from scipy.stats import rankdata
from scipy.spatial.distance import pdist, squareform
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
import itertools
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

# fMRI Config
ROI_NAME = 'V1'
ROI_IDS = [1]  # Using V1 as requested for H0 evaluation
R2_THRESH = 50.0
STANDARDIZE_LVL = 'run'
FMRI_METRIC = 'pearson'  # Toggle: 'pearson', 'mutual_info', or 'distance_correlation'

SAVE_DIR = '/Users/lukatepic/Documents/Topological_Data_Analysis/stimuli/fmri_tda_saliency_final' 
os.makedirs(SAVE_DIR, exist_ok=True)

HDF5_FILE_PATH = '/Users/lukatepic/Documents/Topological_Data_Analysis/stimuli/nsd_stimuli.hdf5'
DATASET_NAME = 'imgBrick'

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

def load_roi_betas(sessions, roi_name, roi_ids, r2_threshold=50.0, standardize_level='run'):
    print(f"\n=== Loading ROI: {roi_name} (IDs: {roi_ids}) | Sessions: {len(sessions)} total ===")
    
    roi_img = nib.load(f"rois/sub_01/HCP_MMP1.nii.gz")
    atlas_data = roi_img.get_fdata().astype(int)
    
    if isinstance(roi_ids, (list, tuple, np.ndarray)):
        roi_mask = np.isin(atlas_data, roi_ids)
    else:
        roi_mask = (atlas_data == roi_ids)
        
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
# === 2. Saliency Extraction ===
# ==============================================================================
def generate_ikn_saliency_map(img_rgb):
    if not IKN_AVAILABLE:
        raise ImportError("pySaliencyMap module not found.")
    
    height, width = img_rgb.shape[:2]
    sm = pySaliencyMap.pySaliencyMap(width, height)
    structural_saliency_map = sm.SMGetIntensityAndOrientation(img_rgb)
    return structural_saliency_map

def build_saliency_feature_matrix(stim_ids, hdf5_path, dset_name, grid_size):
    print(f"\n=== Extracting IKN Saliency Maps (Grid: {grid_size}) ===")
    feature_list = []
    for count, stim_id in enumerate(stim_ids):
        if count % 100 == 0 and count > 0: print(f" -> Processed {count}/{len(stim_ids)} images...")
        img_rgb = fetch_image_from_hdf5(stim_id, hdf5_path, dset_name, force_rgb=True)
        smap_2d = generate_ikn_saliency_map(img_rgb)
        smap_resized = cv2.resize(smap_2d, grid_size, interpolation=cv2.INTER_AREA)
        if np.std(smap_resized) > 0:
            smap_resized = (smap_resized - np.mean(smap_resized)) / np.std(smap_resized)
        feature_list.append(smap_resized.flatten())
    return np.vstack(feature_list) 

# ==============================================================================
# === 3. TDA Engine Extraction Functions ===
# ==============================================================================
def compute_image_topology(feature_matrix, maxdim=2):
    print(f"\n=== Computing TDA on Saliency Map Space (maxdim={maxdim}) ===")
    dist_matrix = np.clip(cosine_distances(feature_matrix), 0, None)
    return ripser(dist_matrix, maxdim=maxdim, distance_matrix=True, do_cocycles=True), dist_matrix

def report_lowest_p_value(feature_label, p_values, valid_ranks, feature_sizes, size_label):
    if not p_values:
        return

    min_idx = int(np.argmin(p_values))
    print(
        f"Lowest p-value for {feature_label}: {p_values[min_idx]:.5f} "
        f"(Rank {valid_ranks[min_idx]}, {size_label}: {feature_sizes[min_idx]})"
    )

def extract_h0_cluster_nodes(ripser_result, dist_matrix, target_rank=1):
    dgm0 = ripser_result['dgms'][0]
    
    # Filter out the infinite baseline component
    dgm0_finite = dgm0[dgm0[:, 1] != np.inf]
    if len(dgm0_finite) == 0: return []

    persistence = dgm0_finite[:, 1]
    if target_rank > len(persistence): return []

    # Get the exact distance where the cluster dies
    idx = np.argsort(persistence)[::-1][target_rank - 1]
    death_dist = persistence[idx]

    # Reconstruct the graph precisely BEFORE the merge happens
    G_before = nx.Graph()
    num_nodes = dist_matrix.shape[0]
    G_before.add_nodes_from(range(num_nodes))
    
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if dist_matrix[i, j] < death_dist - 1e-7:
                G_before.add_edge(i, j)

    # Find ALL edges that share this exact distance
    match_edges = np.argwhere(np.isclose(dist_matrix, death_dist, atol=1e-5))
    if len(match_edges) == 0:
        match_edges = np.argwhere(dist_matrix == death_dist)
    if len(match_edges) == 0: return []

    # Iterate through to find the true bridging edge
    for u, v in match_edges:
        # Prove the edge bridges two previously disconnected islands
        if not nx.has_path(G_before, u, v):
            comp_u = list(nx.node_connected_component(G_before, u))
            comp_v = list(nx.node_connected_component(G_before, v))
            
            # The Elder Rule: The younger/smaller component is the one that dies
            if len(comp_u) == 1 and len(comp_v) == 1:
                return [u, v]
                
            cluster_nodes = comp_u if len(comp_u) < len(comp_v) else comp_v
            return cluster_nodes

    return []


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
    valid_cycles = [c for c in nx.cycle_basis(G_birth) if len(set((min(c[k], c[(k+1)%len(c)]), max(c[k], c[(k+1)%len(c)])) for k in range(len(c))).intersection(cocycle_edges)) % 2 == 1]
    return max(valid_cycles, key=len) if valid_cycles else list(G_birth.nodes())

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
def evaluate_persistence_landscape_h0(ripser_result, dist_matrix, target_stim_ids, fmri_signatures, n_top=100, n_permutations=2000, save_dir=None):
    print(f"\n=== Evaluating Top {n_top} H0 Saliency Clusters (Global Random Cluster Null) ===")
    
    dgm0 = ripser_result['dgms'][0]
    dgm0_finite = dgm0[dgm0[:, 1] != np.inf]
    
    if len(dgm0_finite) == 0:
        print("⚠️ No finite H0 components found.")
        return [], [], []
        
    print(" -> Precomputing global fMRI similarity matrix...")
    total_global_nodes = len(target_stim_ids)
    global_nodes = list(range(total_global_nodes))
    global_sim_matrix = build_fmri_similarity_matrix(global_nodes, target_stim_ids, fmri_signatures, metric=FMRI_METRIC)
        
    max_available = len(dgm0_finite)
    n_test = min(n_top, max_available)
    persistence = dgm0_finite[:, 1]
    
    print(f" -> Found {max_available} total clusters. Searching for the top {n_test} valid multi-image clusters...")

    p_values, valid_ranks, cluster_sizes = [], [], []
    valid_tested = 0

    # Iterate through all available ranks until we hit our target number of valid clusters
    for rank in range(1, max_available + 1):
        if valid_tested >= n_test:
            break  # Stop once we have successfully tested the requested amount
            
        cluster_nodes = extract_h0_cluster_nodes(ripser_result, dist_matrix, rank)
        cluster_size = len(cluster_nodes)
        
        # Skip size-1 single-image outliers
        if cluster_size < 2: 
            continue
            
        # 1. Observed Statistic: Mean similarity of all unique pairs in the cluster
        import itertools
        pairs = list(itertools.combinations(cluster_nodes, 2))
        obs_corrs = [global_sim_matrix[u, v] for u, v in pairs]
        observed_stat = np.mean(obs_corrs)
        
        # 2. Null Distribution: Random clusters of the same size from the entire dataset
        null_distribution = np.zeros(n_permutations)
        for p in range(n_permutations):
            random_nodes = np.random.choice(total_global_nodes, size=cluster_size, replace=False)
            rand_pairs = list(itertools.combinations(random_nodes, 2))
            rand_corrs = [global_sim_matrix[u, v] for u, v in rand_pairs]
            null_distribution[p] = np.mean(rand_corrs)
            
        p_val = np.sum(null_distribution >= observed_stat) / n_permutations
        p_values.append(p_val)
        valid_ranks.append(rank)
        cluster_sizes.append(cluster_size)
        
        valid_tested += 1
        
        if valid_tested % 25 == 0:
            print(f"   ... Successfully tested {valid_tested}/{n_test} clusters (Current Rank depth: {rank}).")

    if not p_values:
        print("No valid clusters to evaluate.")
        return [], [], []
        
    significant_ranks = [r for r, p in zip(valid_ranks, p_values) if p <= 0.05]
    
    print("\n--- Final H0 Landscape Results (Uncorrected) ---")
    print(f"Total Clusters Tested: {len(p_values)}")
    print(f"Significant Clusters (Raw p <= 0.05): {len(significant_ranks)}")
    print(f"Significant Ranks: {significant_ranks}")
    report_lowest_p_value("saliency H0 clusters", p_values, valid_ranks, cluster_sizes, "Cluster Size")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(p_values, bins=np.linspace(0, 1, 51), color='darkorange', kde=False, ax=axes[0])
    axes[0].set_xlim(0, 1.0)
    axes[0].axvline(0.05, color='red', linestyle='dashed', linewidth=2, label='p = 0.05 threshold')
    axes[0].set_title('p-value distribution for saliency H0 clusters', fontweight='bold')
    axes[0].set_xlabel('Raw p-value')
    axes[0].set_ylabel('Number of H0 Clusters')
    axes[0].legend()
    
    colors = ['#FF4C4C' if p <= 0.05 else '#B0B0B0' for p in p_values]
    axes[1].scatter(cluster_sizes, p_values, c=colors, alpha=0.8, edgecolor='black', s=50)
    axes[1].axhline(0.05, color='red', linestyle='dashed', linewidth=2, label='Alpha threshold (0.05)')
    axes[1].set_title('p-values by cluster size\nSaliency-based evaluation', fontweight='bold')
    axes[1].set_xlabel('Cluster Size (Number of Nodes)')
    axes[1].set_ylabel('Raw p-value')
    axes[1].set_yscale('log') 
    axes[1].invert_yaxis()
    axes[1].invert_xaxis()
    axes[1].legend()
    
    plt.tight_layout()
    if save_dir: plt.savefig(os.path.join(save_dir, "H0_Persistence_Landscape_pvalues_global_null.png"), dpi=150)
    plt.show()
    
    return significant_ranks, valid_ranks, p_values

def evaluate_persistence_landscape_h1(ripser_result, dist_matrix, target_stim_ids, fmri_signatures, n_top=100, n_permutations=2000, save_dir=None):
    print(f"\n=== Evaluating Top {n_top} H1 Saliency Loops (Global Random Loop Null) ===")
    
    dgm1, cocycles = ripser_result['dgms'][1], ripser_result['cocycles'][1]
    if len(dgm1) == 0:
        print("⚠️ No H1 loops found in the data.")
        return [], [], []
        
    total_global_nodes = len(target_stim_ids)
    global_nodes = list(range(total_global_nodes))
    global_sim_matrix = build_fmri_similarity_matrix(global_nodes, target_stim_ids, fmri_signatures, metric=FMRI_METRIC)
        
    max_available = len(dgm1)
    n_test = min(n_top, max_available)
    persistence = dgm1[:, 1] - dgm1[:, 0]
    sorted_loop_indices = np.argsort(persistence)[::-1]
    
    p_values, valid_ranks, loop_sizes = [], [], []
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
        loop_sizes.append(loop_size)
        
    if not p_values: return [], [], []
    significant_ranks = [r for r, p in zip(valid_ranks, p_values) if p <= 0.05]
    report_lowest_p_value("saliency H1 loops", p_values, valid_ranks, loop_sizes, "Loop Size")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(p_values, bins=np.linspace(0, 1, 51), color='royalblue', kde=False, ax=axes[0])
    axes[0].set_xlim(0, 1.0)
    axes[0].axvline(0.05, color='red', linestyle='dashed', linewidth=2)
    axes[0].set_title('p-value distribution for saliency H1 loops', fontweight='bold')
    axes[0].set_xlabel('p-value')
    axes[0].set_ylabel('Number of H1 Loops')
    
    colors = ['#FF4C4C' if p <= 0.05 else '#B0B0B0' for p in p_values]
    axes[1].scatter(loop_sizes, p_values, c=colors, alpha=0.8, edgecolor='black', s=50)
    axes[1].axhline(0.05, color='red', linestyle='dashed', linewidth=2)
    axes[1].set_title('p-values by loop size\nSaliency-based evaluation', fontweight='bold')
    axes[1].set_xlabel('Loop Size (Number of Nodes)')
    axes[1].set_ylabel('p-value')
    axes[1].set_yscale('log'); axes[1].invert_yaxis(); axes[1].invert_xaxis()
    plt.tight_layout()
    if save_dir: plt.savefig(os.path.join(save_dir, "H1_Persistence_Landscape_pvalues_global_null.png"), dpi=150)
    plt.show()
    
    return significant_ranks, valid_ranks, p_values

def evaluate_persistence_landscape_h2(ripser_result, dist_matrix, target_stim_ids, fmri_signatures, n_top=100, n_permutations=2000, save_dir=None):
    print(f"\n=== Evaluating Top {n_top} H2 Saliency Voids (Global Random Subgraph Null) ===")
    
    if len(ripser_result['dgms']) < 3 or len(ripser_result['dgms'][2]) == 0:
        return [], [], []
        
    dgm2, cocycles2 = ripser_result['dgms'][2], ripser_result['cocycles'][2]
    total_global_nodes = len(target_stim_ids)
    global_nodes = list(range(total_global_nodes))
    global_sim_matrix = build_fmri_similarity_matrix(global_nodes, target_stim_ids, fmri_signatures, metric=FMRI_METRIC)
        
    max_available = len(dgm2)
    n_test = min(n_top, max_available)
    persistence = dgm2[:, 1] - dgm2[:, 0]
    sorted_void_indices = np.argsort(persistence)[::-1]

    p_values, valid_ranks, void_sizes = [], [], []

    for rank in range(1, n_test + 1):
        void_idx = sorted_void_indices[rank - 1]
        cocycle = cocycles2[void_idx]
        H2_graph = nx.Graph()
        
        for element in cocycle:
            u, v, w = int(element[0]), int(element[1]), int(element[2])
            H2_graph.add_edge(u, v); H2_graph.add_edge(v, w); H2_graph.add_edge(u, w)
            
        void_nodes = list(H2_graph.nodes())
        num_nodes = len(void_nodes)
        if num_nodes < 4: continue
            
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
        p_values.append(p_val); valid_ranks.append(rank); void_sizes.append(num_nodes)

    if not p_values: return [], [], []
    significant_ranks = [r for r, p in zip(valid_ranks, p_values) if p <= 0.05]
    report_lowest_p_value("saliency H2 voids", p_values, valid_ranks, void_sizes, "Void Size")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(p_values, bins=np.linspace(0, 1, 51), color='mediumseagreen', kde=False, ax=axes[0])
    axes[0].set_xlim(0, 1.0); axes[0].axvline(0.05, color='red', linestyle='dashed', linewidth=2)
    axes[0].set_title('p-value distribution for saliency H2 voids', fontweight='bold')
    axes[0].set_xlabel('p-value')
    axes[0].set_ylabel('Number of H2 Voids')
    
    colors = ['#FF4C4C' if p <= 0.05 else '#B0B0B0' for p in p_values]
    axes[1].scatter(void_sizes, p_values, c=colors, alpha=0.8, edgecolor='black', s=50)
    axes[1].axhline(0.05, color='red', linestyle='dashed', linewidth=2)
    axes[1].set_title('p-values by void size\nSaliency-based evaluation', fontweight='bold')
    axes[1].set_xlabel('Void Size (Number of Nodes)')
    axes[1].set_ylabel('p-value')
    axes[1].set_yscale('log'); axes[1].invert_yaxis(); axes[1].invert_xaxis()
    plt.tight_layout()
    if save_dir: plt.savefig(os.path.join(save_dir, "H2_Persistence_Landscape_pvalues_global_null.png"), dpi=150)
    plt.show()
    
    return significant_ranks, valid_ranks, p_values

# ==============================================================================
# === 6. Visualizations ===
# ==============================================================================
def visualize_h0_fmri_network(cluster_nodes, stim_labels, fmri_signatures, save_dir, target_rank=1):
    print(f" -> Generating H0 Cluster Network Plot for Rank {target_rank}...")
    
    G = nx.complete_graph(len(cluster_nodes))
    fig, ax = plt.subplots(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)
    
    # Compute local similarity matrix strictly for the visual edge coloring
    local_sim_matrix = build_fmri_similarity_matrix(cluster_nodes, stim_labels, fmri_signatures, metric=FMRI_METRIC)
    
    edges = G.edges()
    edge_colors = [local_sim_matrix[u, v] for u, v in edges]
        
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color='darkorange', edgecolors='black', ax=ax)
    
    nx.draw_networkx_edges(
        G, pos, edgelist=edges, 
        edge_color=edge_colors, 
        edge_cmap=plt.cm.plasma, 
        width=3.0,
        edge_vmin=min(edge_colors) if edge_colors else 0,
        edge_vmax=max(edge_colors) if edge_colors else 1,
        ax=ax
    )
    
    labels = {i: str(stim_labels[node]) for i, node in enumerate(cluster_nodes)}
    nx.draw_networkx_labels(G, pos, labels, font_size=10, font_weight="bold", ax=ax)
    
    if edge_colors:
        sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma, norm=plt.Normalize(vmin=min(edge_colors), vmax=max(edge_colors)))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.7, aspect=20, pad=0.05)
        cbar.set_label('fMRI Response Similarity (Edges)', fontweight='bold', fontsize=12)
    
    ax.set_title(f"H0 Cluster Topology Colored by fMRI Correlation (Rank {target_rank})", fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')
    plt.tight_layout()
    
    if save_dir: 
        plt.savefig(os.path.join(save_dir, f"H0_fMRI_Network_Isomorphism_Rank{target_rank}.png"), dpi=150)
    plt.show(fig)

# ==============================================================================
# === Main Execution ===
# ==============================================================================
if __name__ == "__main__":
    
    try:
        # 1. Data Prep
        target_stim_ids = get_target_stimuli(TSV_FILE_PATH, SESSIONS, REQUIRED_REPS)
        if not target_stim_ids: exit("⚠️ No stimuli met criteria.")
            
        saliency_features = build_saliency_feature_matrix(
            target_stim_ids, HDF5_FILE_PATH, DATASET_NAME, SALIENCY_GRID_SIZE
        )
        
        raw_betas, loaded_indices, full_df = load_roi_betas(SESSIONS, ROI_NAME, ROI_IDS, R2_THRESH, STANDARDIZE_LVL)
        fmri_signatures = extract_averaged_fmri_signatures(raw_betas, loaded_indices, full_df, SESSIONS, target_stim_ids)
        
        # 2. Compute TDA (maxdim=2 captures H0, H1, H2)
        ripser_res, dist_matrix = compute_image_topology(saliency_features, maxdim=2)

        # ==========================================
        # 3a. Process H0 Clusters (V1 Focused)
        # ==========================================
        # sig_ranks_h0, valid_ranks_h0, p_values_h0 = evaluate_persistence_landscape_h0(
        #     ripser_result=ripser_res, 
        #     dist_matrix=dist_matrix, 
        #     target_stim_ids=target_stim_ids, 
        #     fmri_signatures=fmri_signatures, 
        #     n_top=250,           
        #     n_permutations=10000, 
        #     save_dir=SAVE_DIR
        # )

        # target_h0_rank = sig_ranks_h0[0] if sig_ranks_h0 else 1
        # print(f"\n>>> Visualizing Physics for Target H0 Rank: {target_h0_rank}")
        
        # cluster_nodes = extract_h0_cluster_nodes(ripser_res, dist_matrix, target_h0_rank)
        # if cluster_nodes:
        #     visualize_h0_fmri_network(cluster_nodes, target_stim_ids, fmri_signatures, SAVE_DIR, target_rank=target_h0_rank)

        # ==========================================
        # 3b. Process H1 Loops
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

        # ==========================================
        # 3c. Process H2 Voids
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

    except Exception as e: 
        import traceback
        traceback.print_exc()
        print(f"\n❌ An error occurred: {e}")
