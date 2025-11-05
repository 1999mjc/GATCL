import torch
import pandas as pd
import numpy as np
import h5py
import anndata as ad
import scanpy as sc
import matplotlib.pyplot as plt

# --- Import necessary evaluation metrics (Assumed to be installed) ---
from sklearn.metrics import (
    homogeneity_score, mutual_info_score, v_measure_score,
    adjusted_mutual_info_score, normalized_mutual_info_score, adjusted_rand_score
)
# --- Rpy2 Imports (Used for mclust clustering) ---
from rpy2.robjects.conversion import localconverter 
import rpy2.robjects as ro
import rpy2.robjects.numpy2ri as rpynp 
from rpy2.robjects.packages import importr

def clr_normalize_each_cell(adata, inplace=True):
    """Normalize count vector for each cell, using Seurat's CLR method."""
    import numpy as np
    import scipy

    def seurat_clr(x):
        # Calculate geometric mean of log(x) and return CLR transformed value
        s = np.sum(np.log1p(x[x > 0]))
        exp = np.exp(s / len(x))
        return np.log1p(x / exp)

    if not inplace:
        adata = adata.copy()
    
    # Apply CLR transformation along axis 1 (per cell/row)
    adata.X = np.apply_along_axis(
        seurat_clr, 1, (adata.X.A if scipy.sparse.issparse(adata.X) else np.array(adata.X))
    )
    return adata    

def compute_scores(true, pred):
    """Calculates all supervised clustering evaluation metrics."""
    return {
        "Homogeneity": homogeneity_score(true, pred),
        "Mutual_info": mutual_info_score(true, pred),
        "V_measure": v_measure_score(true, pred),
        "AMI": adjusted_mutual_info_score(true, pred),
        "NMI": normalized_mutual_info_score(true, pred),
        "ARI": adjusted_rand_score(true, pred)
    }


def run_example():
    """Runs the full GATCL workflow: data loading, preprocessing, training, mclust clustering, and metric calculation."""
    
    # 2. Define Data Paths and Hyperparameters
    RNA_DATA_PATH = "./data/RNA_data.h5ad" # Path to the RNA data file
    PROT_DATA_PATH = "./data/PROT_data.h5ad" # Path to the Protein/ATAC data file
    K_NEIGHBORS = 10 # K-Nearest Neighbors used for graph construction
    TARGET_CLUSTERS = 10 # Target number of clusters for mclust (matching ground truth categories)
    
    print(f"--- Running GATCL Real Data Example ---")

    # 1. Load Data
    try:
        adata_rna = sc.read_h5ad(RNA_DATA_PATH)
        adata_prot = sc.read_h5ad(PROT_DATA_PATH)
    except FileNotFoundError:
        print("\nERROR: Data files not found. Please check paths.")
        return
        
    # --- Start Standard Data Preprocessing Pipeline ---
    
    # RNA: 1. Filter low-expressed genes
    sc.pp.filter_genes(adata_rna, min_cells=10)
    # RNA: 2. Normalize and log-transform
    sc.pp.normalize_total(adata_rna)
    sc.pp.log1p(adata_rna)
    # RNA: 3. Select highly variable genes (HVG)
    sc.pp.highly_variable_genes(adata_rna, n_top_genes=3000)
    adata_rna = adata_rna[:, adata_rna.var['highly_variable']].copy()
    
    # Protein/ATAC: 4. Apply CLR normalization
    clr_normalize_each_cell(adata_prot)
    
    # 5. Apply PCA for final feature generation (20 components used for the model)
    PCA_COMPONENTS = 20
    sc.tl.pca(adata_rna, n_comps=PCA_COMPONENTS)
    sc.tl.pca(adata_prot, n_comps=PCA_COMPONENTS)
    
    # 6. Set final feature representations in obsm['feat'] for model input
    adata_rna.obsm['feat'] = adata_rna.obsm["X_pca"]
    adata_prot.obsm['feat'] = adata_prot.obsm["X_pca"]
    
    # --- Check for Necessary Features (after preprocessing) ---
    if 'feat' not in adata_rna.obsm or 'spatial' not in adata_rna.obsm:
        print("\nERROR: AnnData objects must contain 'feat' (embeddings/PCA) and 'spatial' (coordinates) in .obsm")
        print("Please ensure spatial coordinates are available.")
        return

    N_SPOTS = adata_rna.n_obs
    print(f"Data Loaded: {N_SPOTS} Spots.")
    
    # 2. Graph Construction
    data_pkg = build_graphs(adata_rna, adata_prot, n_neighbors=K_NEIGHBORS)
    
    # 3. Model Instantiation and Training (Assumes GATCL_Trainer is defined)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Use GPU if available
    trainer = GATCL_Trainer(
        data=data_pkg,
        device=device,
        epochs=100, 
        dim_output=64 # Latent embedding dimension
    )
    
    # 4. Execute Training, get final embedding
    output = trainer.train()
    final_embedding = output['embedding']
    
    # 5. Save embedding result (for external use/mclust input)
    np.save("combine_embedding.npy", final_embedding)

    # --- Clustering and Evaluation (Using mclust via rpy2) ---
    print("\n--- Starting mclust Clustering and Metric Calculation ---")

    # A. Extract Ground Truth Labels (Assumes annotation.csv exists)
    try:
        annotation = pd.read_csv("annotation.csv")
        annotation.set_index("Barcode", inplace=True)
    except FileNotFoundError:
        print("ERROR: Ground truth file annotation.csv not found. Cannot calculate metrics.")
        return

    # Extract Spot names as index for matching
    barcodes = adata_prot.obs_names 
    
    # Assume annotation.csv contains 'manual-anno' labels
    # Match barcodes and extract true labels
    true_labels = annotation.loc[barcodes, "manual-anno"].astype('category').values

    # B. mclust Clustering 
    # Activate numpy to R conversion for the mclust call
    with localconverter(ro.default_converter + rpynp.numpy2rpy):
        mclust = importr('mclust')
        r_embedding = rpynp.numpy2rpy(final_embedding)
        # Fixed number of clusters G=10 (matching ground truth count)
        mclust_result = mclust.Mclust(r_embedding, G=TARGET_CLUSTERS) 
        r_labels = mclust_result.rx2('classification')
        
        # Convert R's 1-based labels to Python's 0-based labels (CRITICAL)
        pred_labels = np.array(r_labels, dtype=int) - 1 

    # C. Calculate Evaluation Metrics
    
    # Check if label lengths and order match (CRITICAL!)
    if len(true_labels) != len(pred_labels):
         print("ERROR: True labels and predicted labels length mismatch. Cannot calculate metrics.")
         return

    metrics_result = compute_scores(true_labels, pred_labels)
    
    # D. Print Results
    print("\n--- Clustering Evaluation Metrics ---")
    for metric, score in metrics_result.items():
          print(f"{metric}: {score:.4f}")
    print("----------------------")


if __name__ == "__main__":
    # Ensure all auxiliary functions (like GATCL_Trainer, build_graphs, etc.) are imported or defined before running
    run_example()
