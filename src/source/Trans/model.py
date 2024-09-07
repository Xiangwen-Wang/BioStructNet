import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm
from torch_geometric.nn import GCNConv, GATConv, global_max_pool as gmp, global_add_pool as gap, global_mean_pool, global_sort_pool
from torch_geometric.utils import dropout_adj
from torch.nn import TransformerEncoder, TransformerEncoderLayer


def entropy_logits(linear_output):
    p = F.softmax(linear_output, dim=1)
    loss_ent = -torch.sum(p * (torch.log(p + 1e-5)), dim=1)
    return loss_ent

class Main_model(nn.Module):
    def __init__(self, **config):
        super(Main_model, self).__init__()
        ligand_in_feats = config["LIGAND"]["NODE_IN_FEATS"]
        ligand_embedding = config["LIGAND"]["NODE_IN_EMBEDDING"]
        ligand_hidden_feats = config["LIGAND"]["HIDDEN_LAYERS"]
        protein_emb_dim = config["PROTEIN"]["EMBEDDING_DIM"]
        mlp_in_dim = config["DECODER"]["IN_DIM"]
        mlp_hidden_dim = config["DECODER"]["HIDDEN_DIM"]
        mlp_out_dim = config["DECODER"]["OUT_DIM"]
        ligand_padding = config["LIGAND"]["PADDING"]
        trans_emb_dim = config["TRANS"]["IN_DIM"]
        self.num_cand = config["PROTEIN"]["NUM_CAND"]
        Transformer_head = config["TRANS"]["HEAD"]
        self.ligand_extractor = MolecularGCNWithGRU(in_feats=ligand_in_feats, dim_embedding=ligand_embedding,
                                           padding=ligand_padding,
                                           hidden_feats=ligand_hidden_feats)
        self.protein_extractor = Protein_global(output_dim=protein_emb_dim, dropout=0.2)

        self.trans = TransformerBlock(model_dim=trans_emb_dim, output_dim = mlp_in_dim, num_heads=Transformer_head)

        self.mlp_regression = MLPDecoder(mlp_in_dim, mlp_hidden_dim, mlp_out_dim)
        self.output_layer = nn.Linear(mlp_out_dim, 1)

    def forward(self, bg_d, p_globals, mode="train"):
        v_d = self.ligand_extractor(bg_d)

        target_x, target_edge_index, target_batch = p_globals.x, p_globals.edge_index, 1
        v_p = self.protein_extractor(target_x, target_edge_index)


        f,attention_map = self.trans(v_d, v_p)

        score = self.output_layer(self.mlp_regression(f))

        if mode == "train":
            print("train")
            print()

            return v_d, v_p, f, score
        elif mode == "eval":
            return v_d, v_p, score, attention_map


# ligand
class MolecularGCNWithGRU(nn.Module):
    def __init__(self, in_feats, dim_embedding=128, padding=True, hidden_feats=None, activation=None):
        super(MolecularGCNWithGRU, self).__init__()
        self.init_transform = nn.Linear(in_feats, dim_embedding, bias=False)
        if padding:
            with torch.no_grad():
                self.init_transform.weight[-1].fill_(0)
        self.gnn = GCNWithGRU(in_feats=dim_embedding, hidden_feats=hidden_feats, activation=activation)
        self.output_feats = hidden_feats[-1]

    def forward(self, batch_graph):
        node_feats = batch_graph.ndata.pop('h')
        node_feats = self.init_transform(node_feats)
        node_feats = self.gnn(batch_graph, node_feats)
        batch_size = batch_graph.batch_size
        node_feats = node_feats.view(batch_size, -1, self.output_feats)
        return node_feats

class GCNWithGRU(nn.Module):
    def __init__(self, in_feats, hidden_feats, activation):
        super(GCNWithGRU, self).__init__()
        self.layers = nn.ModuleList()
        self.activation = activation

        # Add input layer
        self.layers.append(GCNLayerWithGRU(in_feats, hidden_feats[0], activation))

        # Add hidden layers
        for i in range(1, len(hidden_feats)):
            self.layers.append(GCNLayerWithGRU(hidden_feats[i - 1], hidden_feats[i], activation))

    def forward(self, g, features):
        for layer in self.layers:
            features = layer(g, features)
        return features

class GCNLayerWithGRU(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(GCNLayerWithGRU, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation
        self.gru = nn.GRUCell(out_feats, out_feats)  # GRU unit

    def message_func(self, edges):
        msg = edges.src['h']
        return {'msg': msg}

    def reduce_func(self, nodes):
        msgs = nodes.mailbox['msg']
        aggregated = torch.sum(msgs, dim=1)
        return {'aggregated': aggregated}

    def forward(self, g, features):
        g.ndata['h'] = features
        g.update_all(self.message_func, self.reduce_func)
        aggregated = g.ndata['aggregated']
        h = self.linear(aggregated)
        if self.activation:
            h = self.activation(h)
        # Apply GRU
        features_new = self.gru(h, features)
        return features_new

# ligand end

# protein
# contact map
def positional_encoding(length, d_model, device):

    pe = torch.zeros(length, d_model, device=device)
    position = torch.arange(0, length, dtype=torch.float32, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, device=device).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

class Protein_global(nn.Module):
    def __init__(self, output_dim, dropout=0.2, device='cpu'):
        super(Protein_global, self).__init__()
        self.device = device
        self.output_dim = output_dim
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        input_feature_dim = 26  

        self.pro_conv1 = GCNConv(input_feature_dim, input_feature_dim)
        self.pro_conv2 = GCNConv(input_feature_dim, input_feature_dim * 2)
        self.pro_conv3 = GCNConv(input_feature_dim * 2, input_feature_dim * 4)


        self.pro_fc_g1 = nn.Linear(input_feature_dim * 4, 1024)
        self.pro_fc_g2 = nn.Linear(1024, output_dim)

    def forward(self, target_x, target_edge_index):

        pos_enc = positional_encoding(target_x.size(0), target_x.size(1), self.device)
        target_x = target_x + pos_enc

        xt = self.pro_conv1(target_x, target_edge_index)
        xt = self.relu(xt)

        xt = self.pro_conv2(xt, target_edge_index)
        xt = self.relu(xt)

        xt = self.pro_conv3(xt, target_edge_index)
        xt = self.relu(xt)

        xt = xt.unsqueeze(0)  


        xt = self.relu(self.pro_fc_g1(xt))
        xt = self.dropout(xt)
        xt = self.relu(self.pro_fc_g2(xt))
        xt = self.dropout(xt)

        return xt


class TransformerBlock(nn.Module):
    def __init__(self, model_dim, output_dim, num_heads):
        super(TransformerBlock, self).__init__()
        self.protein_attention = nn.MultiheadAttention(model_dim, num_heads)
        self.cross_attention = nn.MultiheadAttention(model_dim, num_heads)
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, model_dim * 4),
            nn.ReLU(),
            nn.Linear(model_dim * 4, model_dim)
        )
        self.output_linear = nn.Linear(model_dim, output_dim)

    def forward(self, protein_features, ligand_features):

        protein_features = protein_features.transpose(0, 1)  # (sequence length, batch size, embedding dimension)
        ligand_features = ligand_features.transpose(0, 1)

        protein_features, _ = self.protein_attention(protein_features, protein_features, protein_features)
        protein_features = self.norm1(protein_features)

        protein_features, cross_attn_weights = self.cross_attention(protein_features, ligand_features, ligand_features)
        protein_features = self.norm2(protein_features)

        output_features = self.ffn(protein_features)

        output_features = output_features.transpose(0, 1)
        # Global average pooling
        pooled_features = output_features.mean(dim=1, keepdim=True)  
        final_output = self.output_linear(pooled_features.squeeze(1)) 
        return final_output, cross_attn_weights


#MLP
class MLPDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(MLPDecoder, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)  
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x) 
        return x
