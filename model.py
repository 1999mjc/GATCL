import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_geometric.nn import GATConv
from torch.nn import LayerNorm
from torch.nn.modules.module import Module
from tqdm import tqdm
# Assuming 'preprocess' contains the 'adjacent_matrix_preprocessing' function
from preprocess import adjacent_matrix_preprocessing


class GATCL:
    def __init__(self,
                 data,
                 device=torch.device('cpu'),
                 random_seed=2025,
                 learning_rate=0.001,
                 weight_decay=0.00,
                 epochs=120,
                 dim_output=128,
                 weight_factors=[0.1, 0.1, 0.1]
                 ):
        self.data = data.copy()
        self.device = device
        self.random_seed = random_seed
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.dim_output = dim_output
        self.weight_factors = weight_factors

        self.CLloss = CL(
            temperature=0.01,
            neg_sample_ratio=30,
            temp_annealing=0.99
        )

        # Load data for modalities
        # Note: The keys 'adata_omics1' and 'adata_omics2' depend on your input data structure.
        self.adata_1 = self.data['adata_1']
        self.adata_2 = self.data['adata_2']

        # Preprocess adjacency matrices
        self.adj = adjacent_matrix_preprocessing(self.adata_1, self.adata_2)
        # Corrected keys to match the output of the likely preprocessing script
        self.adj_spatial_1 = self.adj['adj_spatial_1'].to(self.device)   # <-- FIXED KEY
        self.adj_spatial_2 = self.adj['adj_spatial_2'].to(self.device)   # <-- FIXED KEY
        self.adj_feature_1 = self.adj['adj_feature_1'].to(self.device)   # <-- FIXED KEY
        self.adj_feature_2 = self.adj['adj_feature_2'].to(self.device)   # <-- FIXED KEY

        # Load features
        self.features_1 = torch.FloatTensor(self.adata_1.obsm['feat'].copy()).to(self.device)
        self.features_2 = torch.FloatTensor(self.adata_2.obsm['feat'].copy()).to(self.device)

        # Get dimensions
        self.n_cells_1 = self.adata_1.n_obs
        self.n_cells_2 = self.adata_2.n_obs
        self.input_dim_1 = self.features_1.shape[1]
        self.input_dim_2 = self.features_2.shape[1]
        self.output_dim_1 = self.dim_output
        self.output_dim_2 = self.dim_output

    def train(self):
        # The main model class is named 'Overall', not 'Encoder_overall'
        self.model = Overall(                                          # <-- FIXED
            self.input_dim_1, self.output_dim_1,
            self.input_dim_2, self.output_dim_2
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate,
                                          weight_decay=self.weight_decay)
        for epoch in tqdm(range(self.epochs)):
            self.model.train()
            results = self.model(
                self.features_1, self.features_2,
                self.adj_spatial_1, self.adj_feature_1,
                self.adj_spatial_2, self.adj_feature_2
            )

            # Calculate loss components
            loss_recon_1 = F.mse_loss(self.features_1, results['emb_recon_1'])
            loss_recon_2 = F.mse_loss(self.features_2, results['emb_recon_2'])
            loss_cl = self.CLloss(results['emb_latent_1'], results['emb_latent_2'])

            # Combine losses
            loss = (self.weight_factors[0] * loss_recon_1 +
                    self.weight_factors[1] * loss_recon_2 +
                    self.weight_factors[2] * loss_cl)

            if epoch % 10 == 0:
                print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {loss.item():.4f}")

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        print("Model training finished!\n")

        with torch.no_grad():
            self.model.eval()
            results = self.model(
                self.features_1, self.features_2,
                self.adj_spatial_1, self.adj_feature_1,
                self.adj_spatial_2, self.adj_feature_2
            )
        emb_1 = F.normalize(results['emb_latent_1'], p=2, eps=1e-12, dim=1)
        emb_2 = F.normalize(results['emb_latent_2'], p=2, eps=1e-12, dim=1)
        emb_combined = F.normalize(results['emb_combined'], p=2, eps=1e-12, dim=1)

        output = {
            'emb_latent_1': emb_1.detach().cpu().numpy(),
            'emb_latent_2': emb_2.detach().cpu().numpy(),
            'embedding': emb_combined.detach().cpu().numpy(),
            'alpha_1': results['alpha_1'].detach().cpu().numpy(),
            'alpha_2': results['alpha_2'].detach().cpu().numpy(),
            'alpha_cross': results['alpha_cross'].detach().cpu().numpy()
        }

        return output


class Overall(Module):
    """
    Top-level module that encapsulates the entire dual-modality autoencoder architecture.
    """
    def __init__(self, input_dim_1, output_dim_1, input_dim_2, output_dim_2, act=F.relu):
        super(Overall, self).__init__()
        self.act = act

        # Initialize encoders and decoders for each modality
        self.encoder_1 = Encoder(input_dim_1, output_dim_1)
        self.decoder_1 = Decoder(output_dim_1, input_dim_1)
        self.encoder_2 = Encoder(input_dim_2, output_dim_2)
        self.decoder_2 = Decoder(output_dim_2, input_dim_2)

        # Initialize attention layers for intra- and inter-modality fusion
        self.attention_1 = Attention(output_dim_1, output_dim_1)
        self.attention_2 = Attention(output_dim_2, output_dim_2)
        self.attention_cross = Attention(output_dim_1, output_dim_2)

    def forward(self, features_1, features_2, adj_spatial_1, adj_feature_1, adj_spatial_2, adj_feature_2):
        # Encode features using both spatial and feature graphs for modality 1
        emb_spatial_1 = self.encoder_1(features_1, adj_spatial_1)
        emb_feature_1 = self.encoder_1(features_1, adj_feature_1)

        # Encode features using both spatial and feature graphs for modality 2
        emb_spatial_2 = self.encoder_2(features_2, adj_spatial_2)
        emb_feature_2 = self.encoder_2(features_2, adj_feature_2)

        # Fuse spatial and feature embeddings within each modality
        emb_latent_1, alpha_1 = self.attention_1(emb_spatial_1, emb_feature_1)
        emb_latent_2, alpha_2 = self.attention_2(emb_spatial_2, emb_feature_2)

        # Fuse the latent embeddings from both modalities
        emb_combined, alpha_cross = self.attention_cross(emb_latent_1, emb_latent_2)

        # Reconstruct original features from the combined embedding
        recon_features_1 = self.decoder_1(emb_combined, adj_spatial_1)
        recon_features_2 = self.decoder_2(emb_combined, adj_spatial_2)

        results = {
            'emb_latent_1': emb_latent_1,
            'emb_latent_2': emb_latent_2,
            'emb_combined': emb_combined,
            'emb_recon_1': recon_features_1,
            'emb_recon_2': recon_features_2,
            'alpha_1': alpha_1,
            'alpha_2': alpha_2,
            'alpha_cross': alpha_cross
        }
        return results


class Encoder(Module):
    def __init__(self, in_feat, out_feat, hidden_dim=None, heads=2, dropout=0.0, act=F.relu):
        super(Encoder, self).__init__()
        self.act = act
        hidden_dim = hidden_dim or out_feat

        # Layer 1
        self.gat1 = GATConv(in_channels=in_feat, out_channels=hidden_dim,
                            heads=heads, concat=True, dropout=dropout)
        dim1 = hidden_dim * heads
        self.norm1 = LayerNorm(dim1)
        self.res_proj1 = nn.Linear(in_feat, dim1) if in_feat != dim1 else nn.Identity()
        self.dropout1 = nn.Dropout(dropout)

        # Layer 2
        self.gat2 = GATConv(in_channels=dim1, out_channels=hidden_dim,
                            heads=heads, concat=True, dropout=dropout)
        dim2 = hidden_dim * heads
        self.norm2 = LayerNorm(dim2)
        self.res_proj2 = nn.Linear(dim1, dim2) if dim1 != dim2 else nn.Identity()
        self.dropout2 = nn.Dropout(dropout)

        # Layer 3 (Output Layer)
        self.gat3 = GATConv(in_channels=dim2, out_channels=out_feat,
                            heads=1, concat=False, dropout=dropout)
        self.norm3 = LayerNorm(out_feat)
        self.res_proj3 = nn.Linear(dim2, out_feat) if dim2 != out_feat else nn.Identity()
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, feat, edge_index):
        x1 = self.act(self.gat1(feat, edge_index))
        x1 = self.norm1(x1)
        x1 = self.dropout1(x1)
        x1 = x1 + self.res_proj1(feat)

        x2 = self.act(self.gat2(x1, edge_index))
        x2 = self.norm2(x2)
        x2 = self.dropout2(x2)
        x2 = x2 + self.res_proj2(x1)

        x3 = self.act(self.gat3(x2, edge_index))
        x3 = self.norm3(x3)
        x3 = self.dropout3(x3)
        x3 = x3 + self.res_proj3(x2)

        return x3


class Decoder(Module):
    def __init__(self, in_feat, out_feat, hidden_dim=None, heads=2, dropout=0.0, act=F.relu):
        super(Decoder, self).__init__()
        self.act = act
        hidden_dim = hidden_dim or out_feat

        # Layer 1
        self.gat1 = GATConv(in_channels=in_feat, out_channels=hidden_dim,
                            heads=heads, concat=True, dropout=dropout)
        dim1 = hidden_dim * heads
        self.norm1 = LayerNorm(dim1)
        self.res_proj1 = nn.Linear(in_feat, dim1) if in_feat != dim1 else nn.Identity()
        self.dropout1 = nn.Dropout(dropout)

        # Layer 2
        self.gat2 = GATConv(in_channels=dim1, out_channels=hidden_dim,
                            heads=heads, concat=True, dropout=dropout)
        dim2 = hidden_dim * heads
        self.norm2 = LayerNorm(dim2)
        self.res_proj2 = nn.Linear(dim1, dim2) if dim1 != dim2 else nn.Identity()
        self.dropout2 = nn.Dropout(dropout)

        # Layer 3
        self.gat3 = GATConv(in_channels=dim2, out_channels=out_feat,
                            heads=1, concat=False, dropout=dropout)
        self.norm3 = LayerNorm(out_feat)
        self.res_proj3 = nn.Linear(dim2, out_feat) if dim2 != out_feat else nn.Identity()
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, feat, edge_index):
        x1 = self.act(self.gat1(feat, edge_index))
        x1 = self.norm1(x1)
        x1 = self.dropout1(x1)
        x1 = x1 + self.res_proj1(feat)

        x2 = self.act(self.gat2(x1, edge_index))
        x2 = self.norm2(x2)
        x2 = self.dropout2(x2)
        x2 = x2 + self.res_proj2(x1)

        x3 = self.act(self.gat3(x2, edge_index))
        x3 = self.norm3(x3)
        x3 = self.dropout3(x3)
        x3 = x3 + self.res_proj3(x2)
        return x3


class Attention(Module):
    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        # The super() call must match the class name 'Attention'
        super(Attention, self).__init__()                             # <-- FIXED
        self.in_feat = in_feat
        self.out_feat = out_feat

        self.w_omega = Parameter(torch.FloatTensor(in_feat, out_feat))
        self.u_omega = Parameter(torch.FloatTensor(out_feat, 1))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.w_omega)
        torch.nn.init.xavier_uniform_(self.u_omega)

    def forward(self, emb1, emb2):
        emb_stack = torch.stack([emb1, emb2], dim=1)

        v = torch.tanh(torch.matmul(emb_stack, self.w_omega))
        vu = torch.matmul(v, self.u_omega)
        alpha = F.softmax(vu.squeeze(2) + 1e-6, dim=1)

        emb_combined = torch.sum(emb_stack * alpha.unsqueeze(-1), dim=1)

        return emb_combined, alpha


class CL(nn.Module):
    """
    Contrastive Loss module.
    """
    def __init__(self, temperature=0.2,
                 neg_sample_ratio=30,
                 temp_annealing=0.99,
                 eps=1e-8):
        super(CL, self).__init__()
        self.temperature = nn.Parameter(torch.tensor(temperature))
        self.neg_sample_ratio = neg_sample_ratio
        self.temp_annealing = temp_annealing
        self.eps = eps
        self.register_buffer('temp_update_count', torch.tensor(0))

    def forward(self, z_i, z_j):
        batch_size = z_i.size(0)

        z_i = F.normalize(z_i, p=2, dim=1)
        z_j = F.normalize(z_j, p=2, dim=1)

        pos_sim = torch.sum(z_i * z_j, dim=1)
        all_neg_pool = torch.cat([z_i, z_j], dim=0)
        neg_indices = self._sample_negatives(batch_size, total_size=2 * batch_size)
        z_neg = all_neg_pool[neg_indices]

        z_i_exp = z_i.unsqueeze(1)
        neg_sim = torch.sum(z_i_exp * z_neg, dim=-1)

        logits = torch.cat([
            pos_sim.unsqueeze(1),
            neg_sim
        ], dim=1) / self.temperature

        labels = torch.zeros(batch_size, dtype=torch.long, device=z_i.device)
        loss = F.cross_entropy(logits, labels)
        if self.training and self.temp_annealing < 1.0:
            self.temp_update_count += 1
            if self.temp_update_count % 100 == 0:
                self.temperature.data *= self.temp_annealing

        return loss

    def _sample_negatives(self, batch_size, total_size):
        num_negs = min(total_size - 2, self.neg_sample_ratio)
        neg_indices = []
        for i in range(batch_size):
            exclude = torch.tensor([i, i + batch_size], device=self.temperature.device)
            all_indices = torch.arange(total_size, device=self.temperature.device)
            possible = all_indices[~torch.isin(all_indices, exclude)]
            selected = possible[torch.randperm(possible.size(0))[:num_negs]]
            neg_indices.append(selected)
        return torch.stack(neg_indices)
