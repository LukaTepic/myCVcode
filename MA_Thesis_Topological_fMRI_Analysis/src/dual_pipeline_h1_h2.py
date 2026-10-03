import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
from ripser import ripser
import networkx as nx
import os
import h5py
import cv2
import plotly.graph_objects as go

# ==============================================================================
# === Configuration & Meta-Data ===
# ==============================================================================

SESSIONS = list(range(1, 10))  
REQUIRED_REPS = 3              
TSV_FILE_PATH = 'responses_sub_01.tsv' 

GABOR_FREQS = [4, 8, 16, 32, 64]
GABOR_ANGLES_DEG = [0.0, 22.5, 45.0, 67.5, 90.0, 112.5, 135.0, 157.5]
GABOR_ANGLES_RAD = np.deg2rad(GABOR_ANGLES_DEG)
GRID_SIZE = (4, 4) 

ANGLE_COLORS = {
    0.0:   '#1f77b4', 22.5:  '#2ca02c', 45.0:  '#bcbd22', 67.5:  '#ff7f0e',
    90.0:  '#d62728', 112.5: '#e377c2', 135.0: '#9467bd', 157.5: '#17becf'
}

LAMBDA_COLORS = {4: '#1f77b4', 8: '#ff7f0e', 16: '#2ca02c', 32: '#9467bd', 64: '#d62728'}

SAVE_PLOTS = True  
SAVE_DIR = '/Users/lukatepic/Documents/Topological_Data_Analysis/stimuli/extracted_loops_coexistence_tda'
os.makedirs(SAVE_DIR, exist_ok=True)

HDF5_FILE_PATH = '/Users/lukatepic/Documents/Topological_Data_Analysis/stimuli/nsd_stimuli.hdf5' 
DATASET_NAME = 'imgBrick'

NORMALIZATION_MODE = 'within_loop' # Options: 'per_node' or 'within_loop'

# ==============================================================================
# === 1. Stimulus Selection & Gabor Setup ===
# ==============================================================================

def get_target_stimuli(tsv_path, sessions, required_reps=3):
    print(f"=== Filtering Stimuli (Sessions: {min(sessions)}-{max(sessions)}, Reps >= {required_reps}) ===")
    if not os.path.exists(tsv_path): raise FileNotFoundError("TSV not found.")
    df = pd.read_csv(tsv_path, sep='\t')
    stim_counts = df[df['SESSION'].isin(sessions)]['73KID'].value_counts()
    target_ids = stim_counts[stim_counts >= required_reps].index.tolist()
    print(f"  -> Found {len(target_ids)} unique stimuli.")
    return sorted(target_ids)

def build_gabor_bank(frequencies, orientations):
    filters = []
    gamma = 0.5 
    for theta in orientations:
        for lambd in frequencies:
            sigma = 0.5 * lambd 
            ksize = (0, 0)
            k_cos = cv2.getGaborKernel(ksize, sigma, theta, lambd, gamma, 0, ktype=cv2.CV_32F)
            k_sin = cv2.getGaborKernel(ksize, sigma, theta, lambd, gamma, np.pi/2, ktype=cv2.CV_32F)
            k_cos -= np.mean(k_cos); k_sin -= np.mean(k_sin)
            k_cos /= (np.sum(np.abs(k_cos)) + 1e-8); k_sin /= (np.sum(np.abs(k_sin)) + 1e-8)
            filters.append((k_cos, k_sin, theta, lambd))
    return filters

def fetch_image(stim_id, hdf5_path, dset_name):
    with h5py.File(hdf5_path, 'r') as f:
        img = f[dset_name][stim_id, ...]
        if img.dtype != np.uint8:
            img = np.clip(img * 255.0, 0, 255).astype(np.uint8) if np.max(img) <= 1.1 else np.clip(img, 0, 255).astype(np.uint8)
    return img

# ==============================================================================
# === 2. UNIFIED Feature Extraction ===
# ==============================================================================

def extract_unified_features(img, filters, grid_size=(4, 4)):
    if len(img.shape) == 3 and img.shape[0] == 3: img = np.transpose(img, (1, 2, 0))
    if len(img.shape) == 3: img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
    img = np.float32(img)
    img = (img - np.mean(img)) / (np.std(img) + 1e-8)
    h, w = img.shape
    cell_h, cell_w = h // grid_size[0], w // grid_size[1]
    
    feature_vector = []
    for k_cos, k_sin, theta, lambd in filters:
        fimg_cos = cv2.filter2D(img, cv2.CV_32F, k_cos)
        fimg_sin = cv2.filter2D(img, cv2.CV_32F, k_sin)
        magnitude = cv2.magnitude(fimg_cos, fimg_sin)
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                cell_mean = np.mean(magnitude[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w])
                feature_vector.append(np.sqrt(cell_mean))
    return np.array(feature_vector, dtype=np.float32)

def build_feature_matrix(stim_ids, hdf5_path, dset_name, gabor_bank, grid_size):
    print(f"\n=== Extracting Unified Euclidean Feature Matrix ===")
    feature_list = []
    for count, stim_id in enumerate(stim_ids):
        if count % 100 == 0 and count > 0: print(f"  -> Processed {count}/{len(stim_ids)} images...")
        img = fetch_image(stim_id, hdf5_path, dset_name)
        feature_list.append(extract_unified_features(img, gabor_bank, grid_size))
        
    feature_matrix = np.vstack(feature_list)
    stds = np.std(feature_matrix, axis=0, keepdims=True)
    stds[stds == 0] = 1.0 
    return feature_matrix / stds

def apply_filters_global(img, filters):
    img = np.float32(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)) if len(img.shape)==3 else np.float32(img)
    img = (img - np.mean(img)) / (np.std(img) + 1e-8)
    results = {}
    for k_cos, k_sin, theta, lambd in filters:
        mag = cv2.magnitude(cv2.filter2D(img, cv2.CV_32F, k_cos), cv2.filter2D(img, cv2.CV_32F, k_sin))
        results[f"deg{np.degrees(theta):g}_lam{lambd}"] = np.mean(mag)
    return results

# ==============================================================================
# === 3. 2D Plotting Engine (Dynamic Dense Packing) ===
# ==============================================================================

def visualize_2d_h1_gabor_loop(h1_ordered_nodes, h2_nodes_set, stim_labels, hdf5_path, 
                               dset_name, gabor_bank, target_lambda, h2_rank, h1_count, 
                               save_dir, norm_mode, global_max_energy=1e-8):
    
    num_nodes = len(h1_ordered_nodes)
    
    NODE_BOX_SIZE = 1.2  
    ARC_SPACE = 1.0      
    
    circle_radius = max(1.5, (num_nodes * ARC_SPACE) / (2 * np.pi))
    fig_size = min(35, max(12, int(circle_radius * 4.5)))
    
    full_angles_rad = np.deg2rad(GABOR_ANGLES_DEG + [a + 180 for a in GABOR_ANGLES_DEG])
    angles_rad_closed = np.concatenate((full_angles_rad, [full_angles_rad[0]]))

    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    
    pos = {}
    node_angles = {}
    for k, node in enumerate(h1_ordered_nodes):
        angle = 2.0 * np.pi * k / num_nodes
        node_angles[node] = angle
        pos[node] = (circle_radius * np.cos(angle), circle_radius * np.sin(angle))
        
    # Draw underlying gray structure lines
    for k in range(num_nodes):
        u, v = h1_ordered_nodes[k], h1_ordered_nodes[(k + 1) % num_nodes]
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], color='#555555', linewidth=4.0, alpha=0.5, zorder=1)

    # Draw the flowers
    for node in h1_ordered_nodes:
        x, y = pos[node]
        stim_id = stim_labels[node]
        
        is_shared_with_h2 = node in h2_nodes_set
        node_border_color = '#FFD700' if is_shared_with_h2 else '#555555'
        node_border_width = 4.0 if is_shared_with_h2 else 2.0
        
        img_array = fetch_image(stim_id, hdf5_path, dset_name)
        activations = apply_filters_global(img_array, gabor_bank)
        
        raw_energies = np.array([activations[f"deg{ang:g}_lam{target_lambda}"] for ang in GABOR_ANGLES_DEG])
        
        if norm_mode == 'per_node':
            normalized_energies = raw_energies / (np.max(raw_energies) + 1e-8)
        else:
            # Uses the newly passed GLOBAL void-context maximum!
            normalized_energies = raw_energies / global_max_energy
            
        energies_closed = np.concatenate((normalized_energies, normalized_energies, [normalized_energies[0]]))
        
        axins = ax.inset_axes([x - NODE_BOX_SIZE/2, y - NODE_BOX_SIZE/2, NODE_BOX_SIZE, NODE_BOX_SIZE], transform=ax.transData, polar=True, zorder=5)
        plot_color = LAMBDA_COLORS.get(target_lambda, '#333333') 
        
        axins.plot(angles_rad_closed, energies_closed, color=plot_color, linewidth=2.0)
        axins.fill(angles_rad_closed, energies_closed, color=plot_color, alpha=0.3)
        axins.set_ylim(0, 1.05) 
        
        axins.set_theta_zero_location('N') 
        axins.set_theta_direction(-1)
        axins.set_xticks([]); axins.set_yticks([]) 
        axins.spines['polar'].set_color(node_border_color)
        axins.spines['polar'].set_linewidth(node_border_width)
        
        font_weight = 'bold' if is_shared_with_h2 else 'normal'
        text_offset = (NODE_BOX_SIZE / 2) + 0.15
        text_x = x + (text_offset * np.cos(node_angles[node]))
        text_y = y + (text_offset * np.sin(node_angles[node]))
        
        ax.text(text_x, text_y, str(stim_id), fontsize=max(12, int(fig_size*0.7)), fontweight=font_weight, ha='center', va='center', zorder=10)

    norm_title = "Shape Focus (Per-Node Max)" if norm_mode == 'per_node' else "Magnitude Focus (Max across ALL Void's coexisting loops)"
    plt.title(f"2D H1 Orientation Loop #{h1_count} (Coexisting with Rank {h2_rank} H2 Sphere)\n"
              f"Nodes: {num_nodes} | Normalization: {norm_title}\n"
              f"Gold Nodes = Structurally shared with 3D Sphere", 
              fontsize=max(16, int(fig_size*1.2)), pad=30)
    
    ax.set_xlim(-circle_radius - NODE_BOX_SIZE, circle_radius + NODE_BOX_SIZE)
    ax.set_ylim(-circle_radius - NODE_BOX_SIZE, circle_radius + NODE_BOX_SIZE)
    ax.axis('equal'); ax.axis('off')
    
    if save_dir:
        file_name = f"Rank{h2_rank}_Sphere_Coexisting_H1_Loop{h1_count}_{norm_mode}.png"
        plt.savefig(os.path.join(save_dir, file_name), dpi=150, bbox_inches='tight')
        plt.close(fig) 
    else:
        plt.show(block=False)

# ==============================================================================
# === 4. Topology Pipeline: Temporal Coexistence Detective ===
# ==============================================================================

def unified_topology_analysis(feature_matrix, stim_labels, hdf5_path, dset_name, 
                              gabor_bank, h2_percentile=95.0, target_lambda=64, save_dir=None, norm_mode='per_node'):
    
    print("\n=== Computing Unified Euclidean Topology (maxdim=2) ===")
    dist_matrix = euclidean_distances(feature_matrix)
    res = ripser(dist_matrix, maxdim=2, distance_matrix=True, do_cocycles=True)
    
    dgm1 = res['dgms'][1]
    dgm2 = res['dgms'][2]
    
    if len(dgm2) == 0: return print("⚠️ No H2 voids found in this space.")
    
    print(f"\n  -> Extracting Top H2 Spheres (Top {100-h2_percentile}%)...")
    p2 = dgm2[:, 1] - dgm2[:, 0]
    thresh2 = np.percentile(p2, h2_percentile)
    top_voids = sorted([i for i, p in enumerate(p2) if p >= thresh2], key=lambda i: p2[i], reverse=True)
    
    for rank, void_idx in enumerate(top_voids):
        birth_h2, death_h2 = dgm2[void_idx]
        cocycle = res['cocycles'][2][void_idx]
        
        h2_nodes_set, h2_edges_set = set(), set()
        for triangle in cocycle:
            u, v, w = int(triangle[0]), int(triangle[1]), int(triangle[2])
            h2_nodes_set.update([u, v, w])
            h2_edges_set.update([(min(u, v), max(u, v)), (min(v, w), max(v, w)), (min(u, w), max(u, w))])

        nodes = list(h2_nodes_set)
        G_h2 = nx.Graph()
        G_h2.add_nodes_from(nodes); G_h2.add_edges_from(h2_edges_set)
        
        pos_3d = nx.spring_layout(G_h2, dim=3, iterations=200, seed=42)
        pos_sphere = {n: coords / (np.linalg.norm(coords) or 1) for n, coords in pos_3d.items()}

        print(f"\n  ------------------------------------------------")
        print(f"  🟢 Rank {rank+1} H2 Sphere Verification (Birth: {birth_h2:.4f} | Death: {death_h2:.4f})")
        print(f"     -> Sphere contains {len(nodes)} total images.")
        
        coexisting_h1_nodes = set()
        coexisting_h1_edges = set()
        
        # Data structures for the 2-Pass Mapping
        loops_to_plot = []
        unique_coexisting_nodes_for_max = set()
        overlapping_loop_count = 0
        
        if len(dgm1) > 0:
            p1_all = dgm1[:, 1] - dgm1[:, 0]
            strict_thresh_h1 = np.percentile(p1_all, 95.0) 
            
            # --- PASS 1: Find all valid coexisting loops ---
            for h1_idx, (birth_h1, death_h1) in enumerate(dgm1):
                if (death_h1 - birth_h1) >= strict_thresh_h1:
                    if birth_h1 < death_h2 and death_h1 > birth_h2:
                        h1_cocycle = res['cocycles'][1][h1_idx]
                        
                        G_h1 = nx.Graph()
                        for u in range(len(stim_labels)):
                            for v in range(u + 1, len(stim_labels)):
                                if dist_matrix[u, v] <= birth_h1 + 0.0001: 
                                    G_h1.add_edge(u, v)

                        cocycle_edges = set((min(u, v), max(u, v)) for u, v, _ in h1_cocycle)
                        cycles = nx.cycle_basis(G_h1)
                        valid_cycles = [c for c in cycles if len(set((min(c[k], c[(k+1)%len(c)]), max(c[k], c[(k+1)%len(c)])) for k in range(len(c))).intersection(cocycle_edges)) % 2 == 1]
                        h1_cycle_nodes = max(valid_cycles, key=len) if valid_cycles else list(G_h1.nodes())
                        
                        touches_sphere = False
                        for n in h1_cycle_nodes:
                            if n in h2_nodes_set:
                                touches_sphere = True
                                break
                                
                        if touches_sphere:
                            overlapping_loop_count += 1
                            loops_to_plot.append(h1_cycle_nodes)
                            unique_coexisting_nodes_for_max.update(h1_cycle_nodes)
                        
                            for i in range(len(h1_cycle_nodes)):
                                u, v = h1_cycle_nodes[i], h1_cycle_nodes[(i + 1) % len(h1_cycle_nodes)]
                                coexisting_h1_nodes.update([stim_labels[u], stim_labels[v]])
                                coexisting_h1_edges.add(frozenset([stim_labels[u], stim_labels[v]]))
        
        # --- NEW: Calculate Global Max across ALL loops that touched this void ---
        void_coexisting_max = 1e-8
        if norm_mode == 'within_loop' and unique_coexisting_nodes_for_max:
            for node in unique_coexisting_nodes_for_max:
                acts = apply_filters_global(fetch_image(stim_labels[node], hdf5_path, dset_name), gabor_bank)
                node_max = max([acts[f"deg{ang:g}_lam{target_lambda}"] for ang in GABOR_ANGLES_DEG])
                if node_max > void_coexisting_max:
                    void_coexisting_max = node_max

        # --- PASS 2: Plot them with the shared Global Context ---
        for plotted_h1_count, h1_cycle_nodes in enumerate(loops_to_plot, start=1):
            visualize_2d_h1_gabor_loop(
                h1_ordered_nodes=h1_cycle_nodes, 
                h2_nodes_set=h2_nodes_set, 
                stim_labels=stim_labels, 
                hdf5_path=hdf5_path, dset_name=dset_name, 
                gabor_bank=gabor_bank, target_lambda=target_lambda, 
                h2_rank=rank+1, h1_count=plotted_h1_count, save_dir=save_dir,
                norm_mode=norm_mode, global_max_energy=void_coexisting_max
            )

        h2_nodes_kids = set(stim_labels[n] for n in nodes)
        h2_edges_kids = set(frozenset([stim_labels[u], stim_labels[v]]) for u, v in G_h2.edges())
        
        intersecting_nodes = h2_nodes_kids.intersection(coexisting_h1_nodes)
        intersecting_edges = h2_edges_kids.intersection(coexisting_h1_edges)
        
        print(f"     -> Strict Detective: Found {overlapping_loop_count} HIGHLY PERSISTENT H1 loops physically attached to this sphere.")
        print(f"     -> Spatial Intersection: {len(intersecting_nodes)} of those nodes physically exist on this H2 Sphere.")
        print(f"     -> EDGE PRESERVATION: {len(intersecting_edges)} exact connections are mapped onto this surface!")
        if overlapping_loop_count > 0 and save_dir:
            print(f"     -> 🖼️ Saved {overlapping_loop_count} dynamically scaled 2D Gabor Loop Plots.")
        
        # --- BUILD THE 3D PLOTLY SPHERE WITH DYNAMIC NORMALIZATION ---
        node_activations = {}
        max_e = 0
        sizes, is_gold = [], []
        
        for node in nodes:
            stim_id = stim_labels[node]
            acts = apply_filters_global(fetch_image(stim_id, hdf5_path, dset_name), gabor_bank)
            node_activations[node] = acts
            sizes.append(sum([acts[f"deg{a:g}_lam{target_lambda}"] for a in GABOR_ANGLES_DEG]))
            is_gold.append(stim_id in intersecting_nodes)
            if max(acts.values()) > max_e: max_e = max(acts.values())
            
        scale_within_loop = 0.35 / (max_e or 1e-8)
        min_s, max_s = min(sizes), max(sizes)
        
        hub_colors = ['#FFD700' if gold else 'rgba(150, 150, 150, 0.4)' for gold in is_gold]
        hub_lines = ['#FF8C00' if gold else 'white' for gold in is_gold]
        hub_sizes = [ (12 if gold else 6) + 12 * ((s - min_s) / (max_s - min_s + 1e-8)) for gold, s in zip(is_gold, sizes)]
        
        hover_texts = [f"<b>⭐ COEXISTING H1 NODE ⭐</b><br>73KID: {stim_labels[n]}" if g else f"73KID: {stim_labels[n]}" for n, g in zip(nodes, is_gold)]

        traces = []
        standard_x, standard_y, standard_z = [], [], []
        gold_x, gold_y, gold_z = [], [], []
        
        for u, v in G_h2.edges():
            edge_id = frozenset([stim_labels[u], stim_labels[v]])
            p_u, p_v = pos_sphere[u], pos_sphere[v]
            
            if edge_id in intersecting_edges:
                gold_x.extend([p_u[0], p_v[0], None])
                gold_y.extend([p_u[1], p_v[1], None])
                gold_z.extend([p_u[2], p_v[2], None])
            else:
                standard_x.extend([p_u[0], p_v[0], None])
                standard_y.extend([p_u[1], p_v[1], None])
                standard_z.extend([p_u[2], p_v[2], None])

        traces.append(go.Scatter3d(x=standard_x, y=standard_y, z=standard_z, mode='lines',
                                   line=dict(color='rgba(100, 100, 100, 0.3)', width=2), hoverinfo='none', showlegend=False))
                                   
        if len(gold_x) > 0:
            traces.append(go.Scatter3d(x=gold_x, y=gold_y, z=gold_z, mode='lines',
                                       line=dict(color='#FFD700', width=6), name="Coexisting H1 Belt", hoverinfo='none'))

        traces.append(go.Scatter3d(
            x=[pos_sphere[n][0] for n in nodes], y=[pos_sphere[n][1] for n in nodes], z=[pos_sphere[n][2] for n in nodes],
            mode='markers', marker=dict(size=hub_sizes, color=hub_colors, line=dict(color=hub_lines, width=2)),
            hoverinfo='text', text=hover_texts, showlegend=False
        ))

        for ang_deg in GABOR_ANGLES_DEG:
            s_x, s_y, s_z, t_x, t_y, t_z, t_v = [], [], [], [], [], [], []
            for node in nodes:
                c = np.array(pos_sphere[node])
                E = node_activations[node][f"deg{ang_deg:g}_lam{target_lambda}"]
                
                base_energies = [node_activations[node][f"deg{a:g}_lam{target_lambda}"] for a in GABOR_ANGLES_DEG]
                local_max = max(base_energies) + 1e-8
                
                if norm_mode == 'per_node':
                    length = (E / local_max) * 0.35
                else:
                    length = E * scale_within_loop
                
                n = c / np.linalg.norm(c)
                k = np.array([0, 0, 1]) if np.abs(np.dot(n, [0,0,1])) <= 0.99 else np.array([0, 1, 0])
                u = np.cross(k, n); u = u / np.linalg.norm(u)
                v = np.cross(n, u)
                
                D = np.cos(np.deg2rad(ang_deg)) * u + np.sin(np.deg2rad(ang_deg)) * v
                p1 = c + length * D; p2 = c - length * D
                s_x.extend([p1[0], p2[0], None]); s_y.extend([p1[1], p2[1], None]); s_z.extend([p1[2], p2[2], None])
                
                tp = c + (length + 0.05) * D
                t_x.append(tp[0]); t_y.append(tp[1]); t_z.append(tp[2])
                t_v.append(f"{E * 100:.1f}")

            traces.append(go.Scatter3d(x=s_x, y=s_y, z=s_z, mode='lines', name=f"{ang_deg}°", legendgroup=f"g_{ang_deg}",
                                       line=dict(width=6, color=ANGLE_COLORS[ang_deg]), hoverinfo='none'))
            traces.append(go.Scatter3d(x=t_x, y=t_y, z=t_z, mode='text', text=t_v, legendgroup=f"g_{ang_deg}",
                                       textfont=dict(color=ANGLE_COLORS[ang_deg], size=10), hoverinfo='none', showlegend=False))

        norm_title = "Per-Node Shape Focus" if norm_mode == 'per_node' else "Within-Loop Magnitude Focus"
        fig = go.Figure(data=traces)
        fig.update_layout(
            title=f"Coexistence Analysis: Rank {rank+1} H2 Sphere<br>Normalization: {norm_title} | Gold = H1 Loops ALIVE during lifespan",
            scene=dict(xaxis=dict(showbackground=False, showticklabels=False, title=''),
                       yaxis=dict(showbackground=False, showticklabels=False, title=''),
                       zaxis=dict(showbackground=False, showticklabels=False, title='')),
            legend=dict(title=dict(text="Angles"), x=0, y=1, bgcolor="rgba(0,0,0,0.5)"),
            margin=dict(l=0, r=0, b=0, t=50), template="plotly_dark"
        )
        if save_dir: fig.write_html(os.path.join(save_dir, f"Coexistence_Sphere_Rank{rank+1}_{norm_mode}.html"))
        else: fig.show()

# ==============================================================================
if __name__ == "__main__":
    
    H2_PERCENTILE = 95.0
    PLOT_SPECIFIC_LAMBDAS = [64]
    
    try:
        ids = get_target_stimuli(TSV_FILE_PATH, SESSIONS, REQUIRED_REPS)
        if not ids: exit()
            
        bank = build_gabor_bank(GABOR_FREQS, GABOR_ANGLES_RAD)
        unified_matrix = build_feature_matrix(ids, HDF5_FILE_PATH, DATASET_NAME, bank, GRID_SIZE)
        
        for current_lambda in PLOT_SPECIFIC_LAMBDAS:
            unified_topology_analysis(
                unified_matrix, ids, HDF5_FILE_PATH, DATASET_NAME, bank, 
                h2_percentile=H2_PERCENTILE, target_lambda=current_lambda, 
                save_dir=SAVE_DIR if SAVE_PLOTS else None,
                norm_mode=NORMALIZATION_MODE
            )
            
        if not SAVE_PLOTS:
            plt.show() 
                
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")