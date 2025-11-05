import torch
import pandas as pd
import numpy as np
import scipy as sp
from scipy.sparse import coo_matrix
from sklearn.neighbors import NearestNeighbors, kneighbors_graph

# Docstring: Build graphs and package AnnData objects
def build_graphs(adata_1, adata_2, n_neighbors=10):
    """
    Constructs both spatial and feature graphs for two omics modalities (adata_1, adata_2)
    and packages them into a dictionary for training.
    """
    # --- Modality 1: Spatial Graph Construction ---
    # Extract physical coordinates from AnnData object 1
    positions_1 = adata_1.obsm['spatial']
    # Build the k-NN spatial graph based on coordinates
    adj_spatial_1 = construct_spatial_graph(positions_1, n_neighbors=n_neighbors)
    # Store the spatial adjacency dataframe in adata.uns (Unstructured data)
    adata_1.uns['adj_spatial'] = adj_spatial_1
    
    # --- Modality 2: Spatial Graph Construction ---
    # Extract physical coordinates from AnnData object 2
    positions_2 = adata_2.obsm['spatial']
    # Build the k-NN spatial graph based on coordinates
    adj_spatial_2 = construct_spatial_graph(positions_2, n_neighbors=n_neighbors)
    # Store the spatial adjacency dataframe in adata.uns
    adata_2.uns['adj_spatial'] = adj_spatial_2
    
    # --- Feature Graph Construction (Modality 1 & 2) ---
    # Build k-NN graphs based on feature correlation/similarity
    adj_feature_1, adj_feature_2 = construct_feature_graphs(adata_1, adata_2)
    # Store the feature adjacency matrix in adata.obsm (Observation-wise data)
    adata_1.obsm['adj_feature'] = adj_feature_1
    adata_2.obsm['adj_feature'] = adj_feature_2
    
    # Package AnnData objects into a dictionary
    data = {'adata_1': adata_1, 'adata_2': adata_2}
    return data


# Docstring: Construct feature-similarity k-NN graphs
def construct_feature_graphs(adata_1, adata_2, k=10, mode="connectivity", metric="correlation", include_self=False):
    """
    Constructs feature k-NN graphs for two omics using their 'feat' representations.
    
    metric="correlation" typically means the graph connects spots with highly correlated features.
    """
    # Build k-NN graph for Modality 1 based on its 'feat' representation
    adj_feature_1 = kneighbors_graph(
        adata_1.obsm['feat'], k, mode=mode, metric=metric, include_self=include_self
    )
    # Build k-NN graph for Modality 2 based on its 'feat' representation
    adj_feature_2 = kneighbors_graph(
        adata_2.obsm['feat'], k, mode=mode, metric=metric, include_self=include_self
    )
    return adj_feature_1, adj_feature_2


# Docstring: Construct k-NN spatial graph from coordinates
def construct_spatial_graph(positions, n_neighbors=6):
    """
    Constructs a spatial k-NN graph based on Euclidean distance between coordinates.
    Returns the graph as a Pandas DataFrame (Edge List format).
    """
    # Fit NearestNeighbors model to spatial coordinates (k+1 because the spot itself is the closest neighbor)
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(positions)
    # Find the distances and indices of k+1 neighbors
    _, indices = nbrs.kneighbors(positions)
    
    # The first column is the source node index
    source_nodes = indices[:, 0].repeat(n_neighbors)
    # The remaining columns are the target node indices (k neighbors, excluding self)
    target_nodes = indices[:, 1:].flatten()
    
    # Create an Edge List DataFrame
    adj_df = pd.DataFrame({
        'source': source_nodes,
        'target': target_nodes,
        'weight': np.ones(len(source_nodes)) # Assign a weight of 1 to all edges
    })
    return adj_df


# Docstring: Convert Edge List DataFrame to SciPy COO sparse matrix
def adj_df_to_coo(adj_df):
    """Converts the spatial graph DataFrame (Edge List) into a SciPy COO sparse matrix."""
    # Determine the total number of nodes (max index + 1)
    n_nodes = int(adj_df['source'].max() + 1)
    # Create COO matrix: (weights, (rows, columns)), shape=(N, N)
    adj_matrix = coo_matrix(
        (adj_df['weight'], (adj_df['source'], adj_df['target'])),
        shape=(n_nodes, n_nodes)
    )
    return adj_matrix


# Docstring: Convert SciPy sparse matrix to PyTorch sparse tensor
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Converts a SciPy sparse matrix into a PyTorch Sparse Tensor (required by PyG or GATs)."""
    # Convert to COO format (required by PyTorch) and cast to float32
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    # Get indices: stack row and column indices
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    # Get values
    values = torch.from_numpy(sparse_mx.data)
    # Get shape
    shape = torch.Size(sparse_mx.shape)
    # Create and return the PyTorch sparse tensor
    return torch.sparse.FloatTensor(indices, values, shape)


# Docstring: Perform GCN-style symmetric normalization
def normalize_graph_adj(adj_matrix):
    """Performs symmetric normalization: D^(-1/2) * (A + I) * D^(-1/2)"""
    # Ensure input is COO sparse format
    adj_matrix = sp.coo_matrix(adj_matrix)
    # Add self-loops (A + I)
    adj_with_self_loops = adj_matrix + sp.eye(adj_matrix.shape[0])
    # Calculate row sum (Degree Matrix D)
    row_sum = np.array(adj_with_self_loops.sum(1))
    # Calculate D^(-1/2)
    inv_sqrt_degree = np.power(row_sum, -0.5).flatten()
    # Handle division by zero (should be 0 where row sum is 0)
    inv_sqrt_degree[np.isinf(inv_sqrt_degree)] = 0.
    # Create the sparse D^(-1/2) diagonal matrix
    degree_matrix_inv_sqrt = sp.diags(inv_sqrt_degree)
    # Apply normalization: D^(-1/2) * (A + I) * D^(-1/2)
    normalized_adj = adj_with_self_loops.dot(degree_matrix_inv_sqrt).transpose().dot(degree_matrix_inv_sqrt).tocoo()
    # Convert the final normalized matrix to a PyTorch sparse tensor
    return sparse_mx_to_torch_sparse_tensor(normalized_adj)


# Docstring: Main function to process graphs before training
def process_adjacency_matrices(adata_1, adata_2):
    """
    Converts graph data from AnnData objects into normalized, symmetric,
    and sparse PyTorch tensors for GAT input.
    """
    # --- Spatial Graph Processing ---
    # Convert Modality 1 spatial Edge List (DataFrame) to COO matrix and then dense array
    adj_spatial_1_df = adata_1.uns['adj_spatial']
    adj_spatial_1 = adj_df_to_coo(adj_spatial_1_df).toarray()
    # Convert Modality 2 spatial Edge List (DataFrame) to COO matrix and then dense array
    adj_spatial_2_df = adata_2.uns['adj_spatial']
    adj_spatial_2 = adj_df_to_coo(adj_spatial_2_df).toarray()
    
    # Symmetrization and Binarization: Ensure A is symmetric (A + A.T) and binary (max value is 1)
    adj_spatial_1 = np.where(adj_spatial_1 + adj_spatial_1.T > 1, 1, adj_spatial_1 + adj_spatial_1.T)
    adj_spatial_2 = np.where(adj_spatial_2 + adj_spatial_2.T > 1
