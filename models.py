import os
import sys
import torch
import torch.nn as nn
import yaml
from constants import RIBNET_MODULES_PATH, RIBNET_WEIGHTS_PATH

# Add RibonanzaNet modules to path
sys.path.append(RIBNET_MODULES_PATH)
from Network import RibonanzaNet, MultiHeadAttention

class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)
        self.entries = entries

    def print(self):
        print(self.entries)

def load_config_from_yaml(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return Config(**config)

class SimpleStructureModule(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, pairwise_dimension, dropout=0.1):
        super(SimpleStructureModule, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, d_model//nhead, d_model//nhead, dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.pairwise2heads = nn.Linear(pairwise_dimension, nhead, bias=False)
        self.pairwise_norm = nn.LayerNorm(pairwise_dimension)

        self.distance2heads = nn.Linear(1, nhead, bias=False)
        
        self.activation = nn.GELU()

    def custom(self, module):
        def custom_forward(*inputs):
            inputs = module(*inputs)
            return inputs
        return custom_forward

    def forward(self, input):
        src, pairwise_features, pred_t, src_mask = input
        
        pairwise_bias = self.pairwise2heads(self.pairwise_norm(pairwise_features)).permute(0, 3, 1, 2)
        
        distance_matrix = pred_t[None, :, :] - pred_t[:, None, :]
        distance_matrix = (distance_matrix**2).sum(-1).clip(2, 37**2).sqrt()
        distance_matrix = distance_matrix[None, :, :, None]
        distance_bias = self.distance2heads(distance_matrix).permute(0, 3, 1, 2)

        pairwise_bias = pairwise_bias + distance_bias

        src2, attention_weights = self.self_attn(src, src, src, mask=pairwise_bias, src_mask=src_mask)
        
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        src = src + self.dropout3(src2)
        src = self.norm3(src)

        return src

class FinetunedRibonanzaNet(RibonanzaNet):
    def __init__(self, config, pretrained=False):
        config.dropout = 0.1
        config.use_grad_checkpoint = True
        super(FinetunedRibonanzaNet, self).__init__(config)
        
        if pretrained:
            self.load_state_dict(torch.load(
                os.path.join(RIBNET_WEIGHTS_PATH, "RibonanzaNet.pt"), 
                map_location='cpu'
            ))
            
        self.dropout = nn.Dropout(0.0)
        self.structure_module = SimpleStructureModule(
            d_model=256, 
            nhead=8, 
            dim_feedforward=1024, 
            pairwise_dimension=64
        )
        
        self.xyz_predictor = nn.Linear(256, 3)

    def custom(self, module):
        def custom_forward(*inputs):
            inputs = module(*inputs)
            return inputs
        return custom_forward
    
    def forward(self, src, mask=None):
        """
        Forward pass supporting batched processing
        
        Args:
            src: [batch_size, seq_len] tensor of sequences
            mask: [batch_size, seq_len] boolean mask to indicate valid positions
        
        Returns:
            List of coordinate predictions
        """
        # Create default mask if none provided
        if mask is None:
            mask = torch.ones_like(src).bool()
        
        # Get embeddings
        sequence_features, pairwise_features = self.get_embeddings(src, mask.long())
        
        batch_size, seq_len = src.shape
        
        # Initialize coordinates for each sequence in batch
        xyzs = []
        xyz = torch.zeros(batch_size, seq_len, 3).to(src.device)
        
        # Iteratively refine coordinates
        for i in range(9):
            new_xyz = xyz.clone()  # Create a new tensor for updated coordinates
            
            for b in range(batch_size):
                # Extract features for this sequence
                seq_feat = sequence_features[b:b+1, :mask[b].sum(), :]
                pair_feat = pairwise_features[b:b+1, :mask[b].sum(), :mask[b].sum(), :]
                curr_xyz = xyz[b, :mask[b].sum(), :]
                
                # Update features
                updated_features = self.structure_module([seq_feat, pair_feat, curr_xyz, None])
                
                # Update coordinates (avoid in-place operations)
                xyz_update = self.xyz_predictor(updated_features).squeeze(0)
                new_xyz[b, :mask[b].sum(), :] = curr_xyz + xyz_update
                
                # Update the corresponding part of sequence_features without in-place operations
                if i < 8:  # Only need to update for next iteration
                    # Create a new tensor for sequence features
                    new_seq = sequence_features[b:b+1].clone()
                    new_seq[0, :mask[b].sum(), :] = updated_features.squeeze(0)
                    sequence_features = torch.cat([
                        sequence_features[:b], 
                        new_seq, 
                        sequence_features[b+1:]
                    ], dim=0) if b < batch_size-1 else torch.cat([
                        sequence_features[:b], 
                        new_seq
                    ], dim=0)
            
            # Update xyz with the new values (no in-place operation)
            xyz = new_xyz
            xyzs.append(xyz)
        
        return xyzs

def get_model(config_path, pretrained=True):
    """Create and return the RNA structure prediction model"""
    model_config = load_config_from_yaml(config_path)
    model = FinetunedRibonanzaNet(model_config, pretrained=pretrained).cuda()
    return model 