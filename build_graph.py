import torch
import pandas as pd
import numpy as np
import scipy as sp
from scipy.sparse import coo_matrix
from sklearn.neighbors import NearestNeighbors, kneighbors_graph


def build_graphs(adata_1, adata_2, n_neighbors=10):
    positions_1 = adata_1.obsm['spatial']
    adj_spatial_1 = construct_spatial_graph(positions_1, n_neighbors=n_neighbors)
    adata_1.uns['adj_spatial'] = adj_spatial_1
    positions_2 = adata_2.obsm['spatial']
    adj_spatial_2 = construct_spatial_graph(positions_2, n_neighbors=n_neighbors)
    adata_2.uns['adj_spatial'] = adj_spatial_2
    adj_feature_1, adj_feature_2 = construct_feature_graphs(adata_1, adata_2)
    adata_1.obsm['adj_feature'] = adj_feature_1
    adata_2.obsm['adj_feature'] = adj_feature_2
    data = {'adata_1': adata_1, 'adata_2': adata_2}
    return data


def construct_feature_graphs(adata_1, adata_2, k=10, mode="connectivity", metric="correlation", include_self=False):
    adj_feature_1 = kneighbors_graph(
        adata_1.obsm['feat'], k, mode=mode, metric=metric, include_self=include_self
    )
    adj_feature_2 = kneighbors_graph(
        adata_2.obsm['feat'], k, mode=mode, metric=metric, include_self=include_self
    )
    return adj_feature_1, adj_feature_2


def construct_spatial_graph(positions, n_neighbors=6):
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(positions)
    _, indices = nbrs.kneighbors(positions)
    source_nodes = indices[:, 0].repeat(n_neighbors)
    target_nodes = indices[:, 1:].flatten()
    adj_df = pd.DataFrame({
        'source': source_nodes,
        'target': target_nodes,
        'weight': np.ones(len(source_nodes))
    })
    return adj_df


def adj_df_to_coo(adj_df):
    n_nodes = int(adj_df['source'].max() + 1)
    adj_matrix = coo_matrix(
        (adj_df['weight'], (adj_df['source'], adj_df['target'])),
        shape=(n_nodes, n_nodes)
    )
    return adj_matrix


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def normalize_graph_adj(adj_matrix):
    adj_matrix = sp.coo_matrix(adj_matrix)
    adj_with_self_loops = adj_matrix + sp.eye(adj_matrix.shape[0])
    row_sum = np.array(adj_with_self_loops.sum(1))
    inv_sqrt_degree = np.power(row_sum, -0.5).flatten()
    inv_sqrt_degree[np.isinf(inv_sqrt_degree)] = 0.
    degree_matrix_inv_sqrt = sp.diags(inv_sqrt_degree)
    normalized_adj = adj_with_self_loops.dot(degree_matrix_inv_sqrt).transpose().dot(degree_matrix_inv_sqrt).tocoo()
    return sparse_mx_to_torch_sparse_tensor(normalized_adj)


def process_adjacency_matrices(adata_1, adata_2):
    adj_spatial_1_df = adata_1.uns['adj_spatial']
    adj_spatial_1 = adj_df_to_coo(adj_spatial_1_df).toarray()
    adj_spatial_2_df = adata_2.uns['adj_spatial']
    adj_spatial_2 = adj_df_to_coo(adj_spatial_2_df).toarray()
    adj_spatial_1 = np.where(adj_spatial_1 + adj_spatial_1.T > 1, 1, adj_spatial_1 + adj_spatial_1.T)
    adj_spatial_2 = np.where(adj_spatial_2 + adj_spatial_2.T > 1, 1, adj_spatial_2 + adj_spatial_2.T)
    norm_adj_spatial_1 = normalize_graph_adj(adj_spatial_1)
    norm_adj_spatial_2 = normalize_graph_adj(adj_spatial_2)
    adj_feature_1 = adata_1.obsm['adj_feature'].toarray()
    adj_feature_2 = adata_2.obsm['adj_feature'].toarray()
    adj_feature_1 = np.where(adj_feature_1 + adj_feature_1.T > 1, 1, adj_feature_1 + adj_feature_1.T)
    adj_feature_2 = np.where(adj_feature_2 + adj_feature_2.T > 1, 1, adj_feature_2 + adj_feature_2.T)
    norm_adj_feature_1 = normalize_graph_adj(adj_feature_1)
    norm_adj_feature_2 = normalize_graph_adj(adj_feature_2)
    processed_adjs = {
        'adj_spatial_1': norm_adj_spatial_1,
        'adj_spatial_2': norm_adj_spatial_2,
        'adj_feature_1': norm_adj_feature_1,
        'adj_feature_2': norm_adj_feature_2,
    }

    return processed_adjs

