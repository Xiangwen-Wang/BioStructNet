import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm
from torch_geometric.nn import GCNConv, GATConv, global_max_pool as gmp, global_add_pool as gap, global_mean_pool, global_sort_pool
from torch_geometric.utils import dropout_adj


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
        num_filters = config["PROTEIN"]["NUM_FILTERS"]
        mlp_in_dim = config["DECODER"]["IN_DIM"]
        mlp_hidden_dim = config["DECODER"]["HIDDEN_DIM"]
        mlp_out_dim = config["DECODER"]["OUT_DIM"]
        ligand_padding = config["LIGAND"]["PADDING"]
        ban_heads = config["BCN"]["HEADS"]
        input_channels = config["PROTEIN"]["CHANNEL"]
        self.num_cand = config["PROTEIN"]["NUM_CAND"]

        self.ligand_extractor = MolecularGCNWithGRU(in_feats=ligand_in_feats, dim_embedding=ligand_embedding,
                                           padding=ligand_padding,
                                           hidden_feats=ligand_hidden_feats)

        self.protein_extractor = Protein_global(output_dim=protein_emb_dim, dropout=0.2)

        self.bcn = weight_norm(
            BANLayer(v_dim=ligand_hidden_feats[-1], q_dim=num_filters[-1], h_dim=mlp_in_dim, h_out=ban_heads),
            name='h_mat', dim=None)

        self.mlp_regression = MLPDecoder(mlp_in_dim, mlp_hidden_dim, mlp_out_dim)
        self.output_layer = nn.Linear(mlp_out_dim, 1)

    def forward(self, bg_d, p_globals, mode="train"):

        v_d = self.ligand_extractor(bg_d)

        target_x, target_edge_index, target_batch = p_globals.x, p_globals.edge_index, 1
        v_p = self.protein_extractor(target_x, target_edge_index)

        f, att = self.bcn(v_d, v_p)
        
        score = self.output_layer(self.mlp_regression(f))
        if mode == "train":
            return v_d, v_p, f, score
        elif mode == "eval":
            return v_d, v_p, score, att



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

#BCN module
class BANLayer(nn.Module):
    def __init__(self, v_dim, q_dim, h_dim, h_out, act='ReLU', dropout=0.2, k=3):
        super(BANLayer, self).__init__()

        self.c = 32
        self.k = k
        self.v_dim = v_dim
        self.q_dim = q_dim
        self.h_dim = h_dim
        self.h_out = h_out

        self.v_net = FCNet([v_dim, h_dim * self.k], act=act, dropout=dropout)
        self.q_net = FCNet([q_dim, h_dim * self.k], act=act, dropout=dropout)
        
        if 1 < k:
            self.p_net = nn.AvgPool1d(self.k, stride=self.k)

        if h_out <= self.c:
            self.h_mat = nn.Parameter(torch.Tensor(1, h_out, 1, h_dim * self.k).normal_())
            self.h_bias = nn.Parameter(torch.Tensor(1, h_out, 1, 1).normal_())
        else:
            self.h_net = weight_norm(nn.Linear(h_dim * self.k, h_out), dim=None)


    def attention_pooling(self, v, q, att_map):
        fusion_logits = torch.einsum('bvk,bvq,bqk->bk', (v, att_map, q))
        if 1 < self.k:
            fusion_logits = fusion_logits.unsqueeze(1)  # b x 1 x d
            fusion_logits = self.p_net(fusion_logits).squeeze(1) * self.k  # sum-pooling
        return fusion_logits

    def forward(self, v, q, softmax=False):
        v_num = v.size(1)
        q_num = q.size(1)
        if self.h_out <= self.c:
            v_ = self.v_net(v)
            q_ = self.q_net(q)
            att_maps = torch.einsum('xhyk,bvk,bqk->bhvq', (self.h_mat, v_, q_)) + self.h_bias
        else:
            v_ = self.v_net(v).transpose(1, 2).unsqueeze(3)
            q_ = self.q_net(q).transpose(1, 2).unsqueeze(2)
            d_ = torch.matmul(v_, q_)  
            att_maps = self.h_net(d_.transpose(1, 2).transpose(2, 3))  
            att_maps = att_maps.transpose(2, 3).transpose(1, 2)  
        if softmax:
            p = nn.functional.softmax(att_maps.view(-1, self.h_out, v_num * q_num), 2)
            att_maps = p.view(-1, self.h_out, v_num, q_num)
        logits = self.attention_pooling(v_, q_, att_maps[:, 0, :, :])
        for i in range(1, self.h_out):
            logits_i = self.attention_pooling(v_, q_, att_maps[:, i, :, :])
            logits += logits_i
    
        return logits, att_maps

class FCNet(nn.Module):
    """Simple class for non-linear fully connect network
    Modified from https://github.com/jnhwkim/ban-vqa/blob/master/fc.py
    """

    def __init__(self, dims, act='ReLU', dropout=0):
        super(FCNet, self).__init__()

        layers = []
        for i in range(len(dims) - 2):
            in_dim = dims[i]
            out_dim = dims[i + 1]
            if 0 < dropout:
                layers.append(nn.Dropout(dropout))
            layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
            if '' != act:
                layers.append(getattr(nn, act)())
        if 0 < dropout:
            layers.append(nn.Dropout(dropout))
        layers.append(weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None))
        if '' != act:
            layers.append(getattr(nn, act)())

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

class BCNet(nn.Module):
    """Simple class for non-linear bilinear connect network
    Modified from https://github.com/jnhwkim/ban-vqa/blob/master/bc.py
    """

    def __init__(self, v_dim, q_dim, h_dim, h_out, act='ReLU', dropout=[.2, .5], k=3):
        super(BCNet, self).__init__()

        self.c = 32
        self.k = k
        self.v_dim = v_dim;
        self.q_dim = q_dim
        self.h_dim = h_dim;
        self.h_out = h_out

        self.v_net = FCNet([v_dim, h_dim * self.k], act=act, dropout=dropout[0])
        self.q_net = FCNet([q_dim, h_dim * self.k], act=act, dropout=dropout[0])
        self.dropout = nn.Dropout(dropout[1])  # attention
        if 1 < k:
            self.p_net = nn.AvgPool1d(self.k, stride=self.k)

        if None == h_out:
            pass
        elif h_out <= self.c:
            self.h_mat = nn.Parameter(torch.Tensor(1, h_out, 1, h_dim * self.k).normal_())
            self.h_bias = nn.Parameter(torch.Tensor(1, h_out, 1, 1).normal_())
        else:
            self.h_net = weight_norm(nn.Linear(h_dim * self.k, h_out), dim=None)

    def forward(self, v, q):
        if None == self.h_out:
            v_ = self.v_net(v)
            q_ = self.q_net(q)
            logits = torch.einsum('bvk,bqk->bvqk', (v_, q_))
            return logits

        elif self.h_out <= self.c:
            v_ = self.dropout(self.v_net(v))
            q_ = self.q_net(q)
            logits = torch.einsum('xhyk,bvk,bqk->bhvq', (self.h_mat, v_, q_)) + self.h_bias
            return logits  # b x h_out x v x q

        else:
            v_ = self.dropout(self.v_net(v)).transpose(1, 2).unsqueeze(3)
            q_ = self.q_net(q).transpose(1, 2).unsqueeze(2)
            d_ = torch.matmul(v_, q_)  # b x h_dim x v x q
            logits = self.h_net(d_.transpose(1, 2).transpose(2, 3))  # b x v x q x h_out
            return logits.transpose(2, 3).transpose(1, 2)  # b x h_out x v x q

    def forward_with_weights(self, v, q, w):
        v_ = self.v_net(v)  # b x v x d
        q_ = self.q_net(q)  # b x q x d
        logits = torch.einsum('bvk,bvq,bqk->bk', (v_, w, q_))
        if 1 < self.k:
            logits = logits.unsqueeze(1)  # b x 1 x d
            logits = self.p_net(logits).squeeze(1) * self.k  # sum-pooling
        return logits
#BCN module end

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
