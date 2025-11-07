import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_geometric.nn import GATConv
from torch.nn import LayerNorm
from torch.nn.modules.module import Module
from tqdm import tqdm
from preprocess import process_adjacency_matrices

import torch
import torch.nn.functional as F
from torch.nn import Module, LayerNorm, Parameter, init  # Ensure these base classes are imported
from tqdm import tqdm
import torch.nn as nn
# Assumed dependencies: GATConv, process_adjacency_matrices, CL, Overall

class GATCL:
    """
    The main training class for the GATCL (Graph Attention Network Meets Contrastive Learning) model.
    Handles data initialization, hyperparameter setting, loss function configuration, and the training loop.
    """
    def __init__(self,
                 data,
                 device=torch.device('cpu'),
                 seed=2025,
                 lr=0.001,
                 weight_decay=0.00,
                 epochs=120,
                 dim_output=128,
                 loss_weight=[0.4, 0.4, 0.4]  # Loss weights: [recon_1, recon_2, cl_loss]
                 ):
        # Copy data to prevent modification of the original AnnData object
        self.data = data.copy()
        # Specify the computing device (CPU or GPU)
        self.device = device
        # Set random seed for experimental reproducibility
        self.seed = seed
        # Learning rate
        self.lr = lr
        # Weight decay (L2 regularization)
        self.weight_decay = weight_decay
        # Number of training epochs
        self.epochs = epochs
        # Dimension of the final embedding vector
        self.dim_output = dim_output
        # Weight factors for reconstruction and contrastive losses
        self.loss_weight = loss_weight

        # Initialize the Contrastive Learning (CL) loss function
        self.CLloss = CL(
            temperature=0.01,
            neg_sample_ratio=30,
            temp_annealing=0.99
        )
        
        # --- Data and Graph Initialization ---
        # Extract AnnData objects
        self.adata_1 = self.data['adata_1']
        self.adata_2 = self.data['adata_2']
        # Preprocess adjacency matrices (assuming process_adjacency_matrices function is defined)
        self.adj = process_adjacency_matrices(self.adata_1, self.adata_2)
        
        # Move the four adjacency matrices to the specified device
        self.adj_spatial_1 = self.adj['adj_spatial_1'].to(self.device)    
        self.adj_spatial_2 = self.adj['adj_spatial_2'].to(self.device)    
        self.adj_feature_1 = self.adj['adj_feature_1'].to(self.device)    
        self.adj_feature_2 = self.adj['adj_feature_2'].to(self.device)    
        
        # Move feature matrices to the specified device
        self.features_1 = torch.FloatTensor(self.adata_1.obsm['feat'].copy()).to(self.device)
        self.features_2 = torch.FloatTensor(self.adata_2.obsm['feat'].copy()).to(self.device)

        # Extract cell counts and input/output dimensions
        self.n_cells_1 = self.adata_1.n_obs
        self.n_cells_2 = self.adata_2.n_obs
        self.input_dim_1 = self.features_1.shape[1]
        self.input_dim_2 = self.features_2.shape[1]
        self.output_dim_1 = self.dim_output
        self.output_dim_2 = self.dim_output

    def train(self):
        """
        Executes the model training loop.
        """
        # Instantiate the overall model (Overall is a combination of Encoder-Decoder-Attention)
        self.model = Overall(                                 
            self.input_dim_1, self.output_dim_1,
            self.input_dim_2, self.output_dim_2
        ).to(self.device)

        # Initialize Adam optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr,
                                              weight_decay=self.weight_decay)
        
        # Start training loop, using tqdm for progress bar display
        for epoch in tqdm(range(self.epochs)):
            self.model.train() # Set model to training mode
            
            # Forward pass: input features and four adjacency matrices
            results = self.model(
                self.features_1, self.features_2,
                self.adj_spatial_1, self.adj_feature_1,
                self.adj_spatial_2, self.adj_feature_2
            )
            
            # --- Loss Calculation ---
            # 1. Reconstruction loss (Modality 1)
            loss_recon_1 = F.mse_loss(self.features_1, results['emb_recon_1'])
            # 2. Reconstruction loss (Modality 2)
            loss_recon_2 = F.mse_loss(self.features_2, results['emb_recon_2'])
            # 3. Contrastive Learning loss (Aligning Latent Space)
            loss_cl = self.CLloss(results['emb_latent_1'], results['emb_latent_2'])
            
            # Combined Loss (Weighted sum)
            loss = (self.loss_weight[0] * loss_recon_1 +
                    self.loss_weight[1] * loss_recon_2 +
                    self.loss_weight[2] * loss_cl)

            # Print loss every 10 epochs (for monitoring)
            if epoch % 10 == 0:
                print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {loss.item():.4f}")

            # Backpropagation and Optimization
            self.optimizer.zero_grad() # Clear gradients
            loss.backward() # Compute gradients
            self.optimizer.step() # Update parameters

        print("Model training finished!\n")

        # --- Evaluation and Output ---
        with torch.no_grad():
            self.model.eval() # Set model to evaluation mode
            
            # Run a final forward pass to get the results
            results = self.model(
                self.features_1, self.features_2,
                self.adj_spatial_1, self.adj_feature_1,
                self.adj_spatial_2, self.adj_feature_2
            )
        
        # L2 Normalization of final embeddings (ensures consistent vector length)
        emb_1 = F.normalize(results['emb_latent_1'], p=2, eps=1e-12, dim=1)
        emb_2 = F.normalize(results['emb_latent_2'], p=2, eps=1e-12, dim=1)
        emb_combined = F.normalize(results['emb_combined'], p=2, eps=1e-12, dim=1)

        # Move results from GPU (if used) to CPU and convert to NumPy array for saving
        output = {
            'emb_latent_1': emb_1.detach().cpu().numpy(),
            'emb_latent_2': emb_2.detach().cpu().numpy(),
            'embedding': emb_combined.detach().cpu().numpy(),  # The final combined embedding
            'alpha_1': results['alpha_1'].detach().cpu().numpy(),
            'alpha_2': results['alpha_2'].detach().cpu().numpy(),
            'alpha_cross': results['alpha_cross'].detach().cpu().numpy() # Cross-modality attention weights
        }

        return output


class Overall(Module):
    """
    Overall GATCL Architecture: Combines Encoder, Decoder, and Attention fusion modules.
    """
    def __init__(self, input_dim_1, output_dim_1, input_dim_2, output_dim_2, act=F.relu):
        super(Overall, self).__init__()
        self.act = act

        # Initialize Encoders and Decoders
        self.encoder_1 = Encoder(input_dim_1, output_dim_1)
        self.decoder_1 = Decoder(output_dim_1, input_dim_1)  # Decoder 1 reconstructs Modality 1 input dimension
        self.encoder_2 = Encoder(input_dim_2, output_dim_2)
        self.decoder_2 = Decoder(output_dim_2, input_dim_2)  # Decoder 2 reconstructs Modality 2 input dimension

        # Initialize Attention Layers: for intra- and inter-modality fusion
        self.attention_1 = Attention(output_dim_1, output_dim_1)  # Modality 1: Fuses spatial and feature embeddings
        self.attention_2 = Attention(output_dim_2, output_dim_2)  # Modality 2: Fuses spatial and feature embeddings
        self.attention_cross = Attention(output_dim_1, output_dim_2)  # Cross-modality fusion: Fuses emb_latent_1 and emb_latent_2

    def forward(self, features_1, features_2, adj_spatial_1, adj_feature_1, adj_spatial_2, adj_feature_2):
        
        # --- Encoder Stage ---
        # Modality 1: Spatial graph embedding
        emb_spatial_1 = self.encoder_1(features_1, adj_spatial_1)
        # Modality 1: Feature graph embedding
        emb_feature_1 = self.encoder_1(features_1, adj_feature_1)
        
        # Modality 2: Spatial graph embedding
        emb_spatial_2 = self.encoder_2(features_2, adj_spatial_2)
        # Modality 2: Feature graph embedding
        emb_feature_2 = self.encoder_2(features_2, adj_feature_2)
        
        # --- Intra-modality Attention Fusion ---
        # Fuses spatial and feature embeddings for Modality 1
        emb_latent_1, alpha_1 = self.attention_1(emb_spatial_1, emb_feature_1)
        # Fuses spatial and feature embeddings for Modality 2
        emb_latent_2, alpha_2 = self.attention_2(emb_spatial_2, emb_feature_2)
        
        # --- Cross-modality Attention Fusion ---
        # Fuses the latent embeddings of Modality 1 and Modality 2
        emb_combined, alpha_cross = self.attention_cross(emb_latent_1, emb_latent_2)
        
        # --- Decoder Stage (Reconstruction) ---
        # Reconstructs Modality 1 input features using combined embedding and spatial graph (for reconstruction loss)
        recon_features_1 = self.decoder_1(emb_combined, adj_spatial_1)
        # Reconstructs Modality 2 input features using combined embedding and spatial graph
        recon_features_2 = self.decoder_2(emb_combined, adj_spatial_2)
        
        # Result dictionary
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
    """
    GAT Encoder Module
    """
    def __init__(self, in_feat, out_feat, hidden_dim=None, heads=2, dropout=0.0, act=F.relu):
        super(Encoder, self).__init__()
        self.act = act
        # Use output dimension if hidden dimension is not specified
        hidden_dim = hidden_dim or out_feat 
        
        # --- GAT Layer 1 ---
        # Multi-head attention (heads=2, concat=True)
        self.gat1 = GATConv(in_channels=in_feat, out_channels=hidden_dim,
                             heads=heads, concat=True, dropout=dropout)
        dim1 = hidden_dim * heads  # Dimension after concatenation
        self.norm1 = LayerNorm(dim1)  # Layer Normalization
        # Residual projection (Linear projection needed if input and output dimension differ)
        self.res_proj1 = nn.Linear(in_feat, dim1) if in_feat != dim1 else nn.Identity()
        self.dropout1 = nn.Dropout(dropout)
        
        # --- GAT Layer 2 ---
        # Multi-head attention (heads=2, concat=True)
        self.gat2 = GATConv(in_channels=dim1, out_channels=hidden_dim,
                             heads=heads, concat=True, dropout=dropout)
        dim2 = hidden_dim * heads  # Dimension after concatenation
        self.norm2 = LayerNorm(dim2)
        self.res_proj2 = nn.Linear(dim1, dim2) if dim1 != dim2 else nn.Identity()
        self.dropout2 = nn.Dropout(dropout)
        
        # --- GAT Layer 3 (Output Layer) ---
        # Single-head attention (heads=1, concat=False) to get the final out_feat dimension
        self.gat3 = GATConv(in_channels=dim2, out_channels=out_feat,
                             heads=1, concat=False, dropout=dropout)
        self.norm3 = LayerNorm(out_feat)
        self.res_proj3 = nn.Linear(dim2, out_feat) if dim2 != out_feat else nn.Identity()
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, feat, edge_index):
        # Layer 1
        x1 = self.act(self.gat1(feat, edge_index))
        x1 = self.norm1(x1)
        x1 = self.dropout1(x1)
        x1 = x1 + self.res_proj1(feat)  # Residual connection
        
        # Layer 2
        x2 = self.act(self.gat2(x1, edge_index))
        x2 = self.norm2(x2)
        x2 = self.dropout2(x2)
        x2 = x2 + self.res_proj2(x1)  # Residual connection
        
        # Layer 3 (Final Output)
        x3 = self.act(self.gat3(x2, edge_index))
        x3 = self.norm3(x3)
        x3 = self.dropout3(x3)
        x3 = x3 + self.res_proj3(x2)  # Residual connection

        return x3


class Decoder(Module):
    """
    GAT Decoder Module (Symmetric to Encoder, typically used for reconstruction).
    """
    def __init__(self, in_feat, out_feat, hidden_dim=None, heads=2, dropout=0.0, act=F.relu):
        super(Decoder, self).__init__()
        self.act = act
        # Decoder maps from latent space (in_feat) back to original input dimension (out_feat)
        hidden_dim = hidden_dim or out_feat 
        
        # --- GAT Layer 1 ---
        self.gat1 = GATConv(in_channels=in_feat, out_channels=hidden_dim,
                             heads=heads, concat=True, dropout=dropout)
        dim1 = hidden_dim * heads
        self.norm1 = LayerNorm(dim1)
        self.res_proj1 = nn.Linear(in_feat, dim1) if in_feat != dim1 else nn.Identity()
        self.dropout1 = nn.Dropout(dropout)
        
        # --- GAT Layer 2 ---
        self.gat2 = GATConv(in_channels=dim1, out_channels=hidden_dim,
                             heads=heads, concat=True, dropout=dropout)
        dim2 = hidden_dim * heads
        self.norm2 = LayerNorm(dim2)
        self.res_proj2 = nn.Linear(dim1, dim2) if dim1 != dim2 else nn.Identity()
        self.dropout2 = nn.Dropout(dropout)

        # --- GAT Layer 3 (Output Layer) ---
        self.gat3 = GATConv(in_channels=dim2, out_channels=out_feat,
                             heads=1, concat=False, dropout=dropout)
        self.norm3 = LayerNorm(out_feat)
        self.res_proj3 = nn.Linear(dim2, out_feat) if dim2 != out_feat else nn.Identity()
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, feat, edge_index):
        # Decoder layers are structurally identical to Encoder layers
        
        # Layer 1
        x1 = self.act(self.gat1(feat, edge_index))
        x1 = self.norm1(x1)
        x1 = self.dropout1(x1)
        x1 = x1 + self.res_proj1(feat)
        
        # Layer 2
        x2 = self.act(self.gat2(x1, edge_index))
        x2 = self.norm2(x2)
        x2 = self.dropout2(x2)
        x2 = x2 + self.res_proj2(x1)

        # Layer 3 (Final Output)
        x3 = self.act(self.gat3(x2, edge_index))
        x3 = self.norm3(x3)
        x3 = self.dropout3(x3)
        x3 = x3 + self.res_proj3(x2)
        return x3


class Attention(Module):
    """
    General Attention Fusion Module (Inspired by SpatialGlue), used for intra- or inter-modality fusion.
    """
    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        super(Attention, self).__init__()                    
        self.in_feat = in_feat
        self.out_feat = out_feat

        # Trainable parameters: W_omega and U_omega (used to compute attention weights)
        self.w_omega = Parameter(torch.FloatTensor(in_feat, out_feat))
        self.u_omega = Parameter(torch.FloatTensor(out_feat, 1))

        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        # Use Xavier uniform initialization for stable training
        init.xavier_uniform_(self.w_omega)
        init.xavier_uniform_(self.u_omega)

    def forward(self, emb1, emb2):
        # Stack the two input embeddings [N, 2, D]
        emb_stack = torch.stack([emb1, emb2], dim=1)

        # 1. Compute attention score V = tanh(E * W_omega)
        v = torch.tanh(torch.matmul(emb_stack, self.w_omega))
        # 2. Compute vu = V * U_omega [N, 2, 1]
        vu = torch.matmul(v, self.u_omega)
        # 3. Compute Softmax weights Alpha across modality dimension 1
        # squeeze(2) removes the last dimension of size 1 [N, 2]
        # + 1e-6 prevents numerical instability
        alpha = F.softmax(vu.squeeze(2) + 1e-6, dim=1)

        # 4. Weighted Sum: emb_combined = sum(E * Alpha)
        # unsqueeze(-1) adds a dimension at the end [N, 2, 1] for broadcast multiplication
        emb_combined = torch.sum(emb_stack * alpha.unsqueeze(-1), dim=1)

        # Return the fused embedding and attention weights
        return emb_combined, alpha


class CL(nn.Module):
    def __init__(self, temperature=0.1,
                 neg_sample_ratio=30,
                 temp_annealing=0.99,
                 eps=1e-8):
        super(CL, self).__init__()
        # Initialize temperature as a trainable parameter.
        self.temperature = nn.Parameter(torch.tensor(temperature)) 
        # Store the desired ratio/number of negative samples (K).
        self.neg_sample_ratio = neg_sample_ratio       
        # Store the factor for temperature decay
        self.temp_annealing = temp_annealing
        # Store a small epsilon value 
        self.eps = eps 
        # Register a buffer to count temperature update steps (does not track gradients).
        self.register_buffer('temp_update_count', torch.tensor(0))   

    def forward(self, z_i, z_j):
        """
        Processes representations from two different views/modalities of the same batch.
        z_i, z_j: Representations from two modalities/views of the same batch samples [B, D].
        """
        batch_size = z_i.size(0)                                    

        # 1. L2 Normalization
        z_i = F.normalize(z_i, p=2, dim=1)                           
        z_j = F.normalize(z_j, p=2, dim=1)                           
        # 2. Positive sample similarity (corresponding index)
        pos_sim = torch.sum(z_i * z_j, dim=1)                        
        # 3. Jointly sample negatives from non-self samples in z_i and z_j
        # Concatenate z_i and z_j into a single pool of potential negatives [2B, D].
        all_neg_pool = torch.cat([z_i, z_j], dim=0)    
        # Get indices for K negative samples for each sample in the batch. Result shape: [B, K].
        neg_indices = self._sample_negatives(batch_size, total_size=2*batch_size) 
        # Select the negative samples from the pool. Result shape: [B, K, D].
        z_neg = all_neg_pool[neg_indices]                            

        # 4. Negative sample similarity
        # Expand z_i to allow broadcasting with negative samples [B, 1, D].
        z_i_exp = z_i.unsqueeze(1)
        # Calculate similarity between z_i and all its K negative samples. Result shape: [B, K].
        neg_sim = torch.sum(z_i_exp * z_neg, dim=-1)                 

        # 5. Construct logits
        logits = torch.cat([
            pos_sim.unsqueeze(1),                                    
            neg_sim                                                 
        ], dim=1) / self.temperature                              

        # 6. Labels
        labels = torch.zeros(batch_size, dtype=torch.long, device=z_i.device) 

        # 7. Contrastive Loss
        loss = F.cross_entropy(logits, labels)                      

        # 8. Temperature Annealing
        if self.training and self.temp_annealing < 1.0:              
            self.temp_update_count += 1                             
            if self.temp_update_count % 100 == 0:                    
                self.temperature.data *= self.temp_annealing         
        return loss                                                  
    def _sample_negatives(self, batch_size, total_size):
        """Samples K negative samples from the 2B pool, avoiding the corresponding positive pair indices."""
        # K is set based on self.neg_sample_ratio, ensuring it doesn't exceed available negatives.
        num_negs = min(total_size - 2, self.neg_sample_ratio)        
        for i in range(batch_size):
            # Indices of the positive pair to exclude for sample i.
            exclude = torch.tensor([i, i + batch_size], device=self.temperature.device)
            # Create all possible indices in the pool [0, 2B-1].
            all_indices = torch.arange(total_size, device=self.temperature.device)  
            # Get indices that are NOT the positive pair.
            possible = all_indices[~torch.isin(all_indices, exclude)] 
            # Randomly select K indices from the possible ones.
            selected = possible[torch.randperm(possible.size(0))[:num_negs]] 
            # Collect the selected indices.
            neg_indices.append(selected)                                               
        return torch.stack(neg_indices)







