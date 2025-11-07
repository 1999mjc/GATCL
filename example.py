import torch
import pandas as pd
import numpy as np
import h5py
import anndata as ad
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from build_graph import *
from model import *

# --- Import necessary evaluation metrics (Assumed to be installed) ---
from sklearn.metrics import (
    homogeneity_score, mutual_info_score, v_measure_score,
    adjusted_mutual_info_score, normalized_mutual_info_score, adjusted_rand_score
)

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
    """Runs the full GATCL workflow: data loading, preprocessing, training, clustering, and metric calculation."""
    
    # We selected some data from the human placenta dataset as an example
    RNA_DATA_PATH = "./data/RNA_data.h5ad" # Path to the RNA data file
    PROT_DATA_PATH = "./data/PROT_data.h5ad" # Path to the Protein/ATAC data file
    K_NEIGHBORS = 10 # K-Nearest Neighbors used for graph construction
    TARGET_CLUSTERS = 10 # Target number of clusters
    # Load Data
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
    
    # Protein:  Apply CLR normalization
    clr_normalize_each_cell(adata_prot)
    
    # 5. Apply PCA for final feature generation 
    PCA_COMPONENTS = adt_adata.n_vars-1  ##choose the proper number
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
        epochs=120, 
        dim_output=64 # Latent embedding dimension
    )
    
    # 4. Execute Training, get final embedding
    output = trainer.train()
    final_embedding = output['embedding']
    
    # 5. Save embedding result
    np.save("combine_embedding.npy", final_embedding)
    np.save(" alpha_1.npy", results['alpha_1'])
    np.save(" alpha_2.npy", results['alpha_2'])
    np.save(" alpha_cross.npy", results['alpha_cross'])
    np.savetxt("embedding.csv", embedding, delimiter=",",  fmt='%.6f')
    ########################################################################################
    # --- Clustering(use R)
    # Define the path to the input data file.
    data_file <- "embedding.csv"

     # 2. Install and load necessary R packages (if not already installed)
     # Check if the 'mclust' package is installed.
     if (!requireNamespace("mclust", quietly = TRUE)) {
      # If not installed, install the package.
        install.packages("mclust")
       }
     # Load the 'mclust' library for model-based clustering.
     library(mclust)

     # 3. Set a random seed to ensure reproducible results
     # Set a specific seed for random number generation.
     set.seed(2020) 

     # 4. Read the data
     # Print a message indicating data reading is in progress.
     cat("reading data...\n")
     # Read the data into an R data frame. header=FALSE indicates no header, sep="," uses comma as delimiter.
     data_for_clustering <- read.csv(data_file, header=FALSE, sep=",")

     # 5. Execute Mclust clustering
     # Define the desired number of clusters (K).
     K <- n # choose a value you want 
     # Print a message about the start of clustering, including the value of K.
     cat(paste(" Mclust clustering，K =", K, "...\n"))

     # Execute the clustering process using Mclust (Gaussian Mixture Models).
     # G = K sets the number of components. modelNames = "EEE" specifies an ellipsoidal model with equal shape and equal orientation.
     mclust_result <- Mclust(data_for_clustering, G = K, modelNames = "EEE")

     # 6. Output and save clustering results
     # Print a confirmation message.
     cat("Mclust finishe!\n\n")

     # View summary information (optional, for checking results)
     # Display a summary of the Mclust results.
     summary(mclust_result)

     # Extract the resulting cluster assignments (labels) into a vector.
     cluster_labels <- mclust_result$classification

     # 2. Define the output filename
     # Define the path for the output CSV file.
     csv_output_file <- "mclust_classification_results_R.csv"

     # 1) Convert the cluster labels vector into a single-column data frame.
     data_to_save <- data.frame(cluster_label = cluster_labels)

     # 2) Use write.csv() function to save the data frame to a CSV file.
     # The data frame to be saved.
     write.csv(
     # The output file path.
          data_to_save,
          file = csv_output_file,
        row.names = FALSE
        )
##########################################################################################################################
    
    # Load the predicted cluster labels saved from the R script (or previous step).
    labels = pd.read_csv('mclust_classification_results_R.csv')

    # Load the ground truth annotations from an Excel file.
    annotation = pd.read_excel(
    # Specify the full path to the annotation file.
           "./data/annotiation_demo.xlsx"
    )

    # Extract true labels (ground truth) from the 'manual' column of the annotation DataFrame.
    true_labels2 = annotation['manual']
    # Assign predicted labels (the entire 'labels' DataFrame/single column) to pred_labels2.
    pred_labels2 = labels  # The name corresponds to the column saved previously.
    # Convert the Pandas DataFrame (pred_labels2) to a NumPy array.
    pred_labels2 = pred_labels2.to_numpy()
    # Flatten the NumPy array to ensure it is a 1D vector, which is required by clustering evaluation functions.
    pred_labels2 = pred_labels2.flatten()  # Recommended, returns a copy.
    # Calculate clustering evaluation scores (ARI, NMI, etc.) using the custom function.
    scores_gatcl = compute_scores(true_labels2, pred_labels2)
    # Convert the dictionary of scores into a Pandas DataFrame for easy saving.
    df_scores = pd.DataFrame(list(scores_gatcl.items()), columns=['Metric', 'Score'])

    # Save the calculated evaluation metrics to an Excel file.
    # df_scores.to_excel(
    # # Specify the output path and filename for the results.
    #         "GATCL_result.xlsx",
    #    index=False
    #  )
    # Print a header for the evaluation metrics in the console.
    # Iterate through the dictionary of scores to print each metric and its value.
    for metric, score in scores_gatcl.items():
     # Print the metric and score.
          print(f"{metric}: {score:.4f}")  

if __name__ == "__main__":
    # Ensure all auxiliary functions (like GATCL_Trainer, build_graphs, etc.) are imported or defined before running
    run_example()
