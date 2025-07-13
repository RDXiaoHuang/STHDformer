import torch
import torch.nn as nn
import torch.nn.functional as F
from STHDformer_backbone.backbone import Temporal_Heterogeneity_SelfAttentionLayer, Spatial_Heterogeneity_SelfAttentionLayer, Dual_graph, Fusion_Model

class AttentionLayer(nn.Module):
    def __init__(self, model_dim, num_heads=8):
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        
        self.FC_Q = nn.Linear(model_dim, model_dim)
        self.FC_K = nn.Linear(model_dim, model_dim)
        self.FC_V = nn.Linear(model_dim, model_dim)
        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(self, query, key, value):
        batch_size = query.shape[0]
                
        query = self.FC_Q(query)
        key = self.FC_K(key)
        value = self.FC_V(value)
        
        query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0) 
        key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)       
        value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)   

        attn_score = (query @ key.transpose(-1, -2)) / self.head_dim**0.5  

        attn_score = F.softmax(attn_score, dim=-1)
        out = attn_score @ value  
        out = torch.cat(torch.split(out, batch_size, dim=0), dim=-1)  
        return self.out_proj(out)

class SelfAttentionLayer(nn.Module):
    def __init__(
        self, model_dim, feed_forward_dim=256, num_heads=4, 
                 dropout=0.1
    ):
        super().__init__()
        
        self.attn = AttentionLayer(
            model_dim=model_dim,
            num_heads=num_heads
        )
        
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        
        self.act_linear = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.GELU()
        )
        
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, enhanced=False, dim=-2):
        
        x = x.transpose(dim, -2)

        residual = x
        
        if enhanced is True:
            augment_residual = self.act_linear(residual)
            out = self.attn(x, x, x)
            out = self.dropout1(out)
            out = self.ln1(residual + out + augment_residual)
        else:
            out = self.attn(x, x, x)
            out = self.dropout1(out)
            out = self.ln1(residual + out)
            
        residual = out
        out = self.feed_forward(out)  
        out = self.dropout2(out)
        out = self.ln2(residual + out)

        out = out.transpose(dim, -2)
        return out

class Cross_AttentionLayer(nn.Module):
    def __init__(self, model_dim, num_heads=8):
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        
        self.FC_Q = nn.Linear(model_dim, model_dim)
        self.FC_K = nn.Linear(model_dim, model_dim)
        self.FC_V = nn.Linear(model_dim, model_dim)
        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(self, query, key, value):
        batch_size = query.shape[0]
                
        query = self.FC_Q(query)
        key = self.FC_K(key)
        value = self.FC_V(value)
        
        query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0) 
        key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)       
        value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)   

        attn_score = (query @ key.transpose(-1, -2)) / self.head_dim**0.5  

        attn_score = F.softmax(attn_score, dim=-1)
        out = attn_score @ value  
        out = torch.cat(torch.split(out, batch_size, dim=0), dim=-1)  
        return self.out_proj(out)

class Cross_SelfAttentionLayer(nn.Module):
    def __init__(
        self, model_dim, feed_forward_dim=256, num_heads=4, 
                 dropout=0.1
    ):
        super().__init__()
        
        self.attn = Cross_AttentionLayer(
            model_dim=model_dim,
            num_heads=num_heads
        )
        
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, y=None, dim=-2):
        
        x = x.transpose(dim, -2)

        residual = x
        if y is not None: 
            y = y.transpose(dim, -2)
            out = self.attn(x, y, y)
            out = self.dropout1(out)
            out = self.ln1(residual + out)
        else:
            out = self.attn(x, x, x)
            out = self.dropout1(out)
            out = self.ln1(residual + out)
            
        residual = out
        out = self.feed_forward(out)  
        out = self.dropout2(out)
        out = self.ln2(residual + out)

        out = out.transpose(dim, -2)
        return out       

class STHDformer(nn.Module):
    def __init__(
        self, num_nodes, in_steps, out_steps, steps_per_day, steps_per_week, input_dim, output_dim,
        input_embedding_dim, tod_embedding_dim, dow_embedding_dim, feed_forward_dim, num_heads,
        num_layers_t, num_layers_s, num_layers_c, node_dim, num_layers_mlp, dropout, adaptive_embedding_dim, use_mixed_proj, use_temporal_heterogeneity, use_spatial_heterogeneity, use_temporal_cross, use_spatial_cross, transition_matrix, pattern_matrix, semantic_matrix,
        ):
        super().__init__()
        self.num_nodes = num_nodes
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.steps_per_day = steps_per_day
        self.steps_per_week = steps_per_week
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_embedding_dim = input_embedding_dim
        self.tod_embedding_dim = tod_embedding_dim
        self.dow_embedding_dim = dow_embedding_dim
        self.model_dim = input_embedding_dim + dow_embedding_dim + tod_embedding_dim + adaptive_embedding_dim 
        self.num_heads = num_heads
        self.use_mixed_proj = use_mixed_proj
        self.adaptive_embedding_dim = adaptive_embedding_dim
        self.dropout = dropout
        self.use_temporal_heterogeneity = use_temporal_heterogeneity
        self.use_spatial_heterogeneity = use_spatial_heterogeneity
        self.use_temporal_cross = use_temporal_cross
        self.use_spatial_cross = use_spatial_cross
        self.transition_matrix = [torch.tensor(i, dtype=torch.float32) for i in transition_matrix]
        self.node_dim = node_dim
        
        if self.input_embedding_dim > 0:
            self.input_proj = nn.Linear(input_dim, input_embedding_dim)

        if tod_embedding_dim > 0:
            self.tod_embedding = nn.Embedding(steps_per_day, tod_embedding_dim)

        if dow_embedding_dim > 0:
            self.dow_embedding = nn.Embedding(steps_per_week, dow_embedding_dim)
            
        if adaptive_embedding_dim > 0:
            self.adaptive_embedding = nn.init.xavier_uniform_(
                nn.Parameter(torch.empty(in_steps, num_nodes, adaptive_embedding_dim))
            )
        
        self.attn_layers_structure = nn.ModuleList(
                [
                    SelfAttentionLayer(self.model_dim, feed_forward_dim, num_heads, dropout)
                    for _ in range(num_layers_s)
                ]
        )
        
        if use_temporal_heterogeneity: 
            self.attn_layers_temporal_heterogeneity = nn.ModuleList(
                [
                    Temporal_Heterogeneity_SelfAttentionLayer(
                        self.model_dim, feed_forward_dim, num_heads, dropout, pattern_matrix
                    )
                    for _ in range(num_layers_t)
                ]
            )
            
        if use_spatial_heterogeneity: 
            self.attn_layers_spatial_heterogeneity = nn.ModuleList(
                [
                    Spatial_Heterogeneity_SelfAttentionLayer(
                        self.model_dim, feed_forward_dim, num_heads, dropout, 
                        num_nodes, semantic_matrix
                    )
                    for _ in range(num_layers_s)
                ]
            )
        
        self.attn_layers_cross = nn.ModuleList(
            [
                Cross_SelfAttentionLayer(self.model_dim, feed_forward_dim, num_heads, dropout)
                for _ in range(num_layers_c)
            ]
        )
        
        if node_dim > 0:
            self.dual_graph = Dual_graph(num_nodes, node_dim)
            self.fusion_spatial_feat = Fusion_Model(input_dim=self.adaptive_embedding_dim + 2 * node_dim, output_dim=self.adaptive_embedding_dim, num_layers_mlp=num_layers_mlp)

        if use_mixed_proj:
            self.output_proj = nn.Linear(in_steps * self.model_dim, out_steps * output_dim)
        else: 
            self.temporal_proj = nn.Linear(in_steps, out_steps)
            self.output_proj = nn.Linear(self.model_dim, self.output_dim)
        
    def forward(self, x):
        batch_size, input_steps, num_nodes, _ = x.shape
        
        if self.tod_embedding_dim > 0:
            tod = x[..., 1]

        if self.dow_embedding_dim > 0:
            dow = x[..., 2]
        
        x = x[..., :self.input_dim]
        
        x = self.input_proj(x)
        features = [x]
        
        if self.tod_embedding_dim > 0:
            tod_emb = self.tod_embedding((tod * self.steps_per_day).long())
            features.append(tod_emb)
        
        if self.dow_embedding_dim > 0:
            dow_emb = self.dow_embedding(dow.long())
            features.append(dow_emb)
        
        if self.adaptive_embedding_dim > 0:
            adp_emb = self.adaptive_embedding.expand(
                size=(batch_size, *self.adaptive_embedding.shape)
            )
            features.append(adp_emb)

        x = torch.cat(features, dim=-1)

        structure_x_temporal = x.clone()
        heterogeneity_x_temporal = x.clone()
        
        for attn_t in self.attn_layers_structure:
            structure_x_temporal = attn_t(structure_x_temporal, dim=1)
        
        if self.use_temporal_heterogeneity:
            for attn_temporal_heterogeneity in self.attn_layers_temporal_heterogeneity:
                heterogeneity_x_temporal = attn_temporal_heterogeneity(heterogeneity_x_temporal, dim=1)
        
        if self.use_temporal_cross: 
            for attn_t_c in self.attn_layers_cross:
                temporal_x = attn_t_c(structure_x_temporal, heterogeneity_x_temporal, dim=1)
        
        structure_x_spatial = temporal_x.clone()
        heterogeneity_x_spatial = temporal_x.clone()
        
        if self.node_dim > 0:  
            dual_graph = self.dual_graph(self.transition_matrix, batch_size, input_steps)
            adp_graph = structure_x_spatial[..., -self.adaptive_embedding_dim:]
            other_graph = structure_x_spatial[..., :self.model_dim - self.adaptive_embedding_dim]
            structure_x_spatial = self.fusion_spatial_feat(dual_graph, adp_graph, other_graph)
        for attn_s in self.attn_layers_structure:
            structure_x_spatial = attn_s(structure_x_spatial, enhanced=True, dim=2)
        
        if self.use_spatial_heterogeneity:
            for attn_spatial_heterogeneity in self.attn_layers_spatial_heterogeneity:
                heterogeneity_x_spatial = attn_spatial_heterogeneity(heterogeneity_x_spatial, dim=2)
        
        if self.use_spatial_cross: 
            for attn_s_c in self.attn_layers_cross:
                x = attn_s_c(structure_x_spatial, heterogeneity_x_spatial, dim=2)
        
        if self.use_mixed_proj:
            out = x.transpose(1, 2)
            out = out.reshape(batch_size, self.num_nodes, self.in_steps * self.model_dim)
            out = self.output_proj(out).view(batch_size, self.num_nodes, self.out_steps, self.output_dim)
            out = out.transpose(1, 2)
        else:
            out = x.transpose(1, 3)
            out = self.temporal_proj(out)
            out = self.output_proj(out.transpose(1, 3))
        return out