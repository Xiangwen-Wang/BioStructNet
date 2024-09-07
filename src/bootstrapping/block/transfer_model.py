import torch
import torch.nn as nn
from source_model import Main_model  
from torch.nn.utils.weight_norm import weight_norm
from time import time
import torch.nn.functional as F
from configs import get_cfg_defaults
from feature import loadData
import dgl
import argparse
import pandas as pd
import random
import copy
import os
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from prettytable import PrettyTable
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold
from sklearn.utils import resample


def set_seed(seed=1000):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# BAN module
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
        # self.dropout = nn.Dropout(dropout[1])
        if 1 < k:
            self.p_net = nn.AvgPool1d(self.k, stride=self.k)

        if h_out <= self.c:
            self.h_mat = nn.Parameter(torch.Tensor(1, h_out, 1, h_dim * self.k).normal_())
            self.h_bias = nn.Parameter(torch.Tensor(1, h_out, 1, 1).normal_())
        else:
            self.h_net = weight_norm(nn.Linear(h_dim * self.k, h_out), dim=None)

        # self.bn = nn.BatchNorm1d(h_dim)

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
            d_ = torch.matmul(v_, q_)  # b x h_dim x v x q
            att_maps = self.h_net(d_.transpose(1, 2).transpose(2, 3))  # b x v x q x h_out
            att_maps = att_maps.transpose(2, 3).transpose(1, 2)  # b x h_out x v x q
        if softmax:
            p = nn.functional.softmax(att_maps.view(-1, self.h_out, v_num * q_num), 2)
            att_maps = p.view(-1, self.h_out, v_num, q_num)
        logits = self.attention_pooling(v_, q_, att_maps[:, 0, :, :])
        for i in range(1, self.h_out):
            logits_i = self.attention_pooling(v_, q_, att_maps[:, i, :, :])
            logits = logits + logits_i
        # logits = self.bn(logits)
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
                layers.append(getattr(nn, act)(inplace=False))
        if 0 < dropout:
            layers.append(nn.Dropout(dropout))
        layers.append(weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None))
        if '' != act:
            layers.append(getattr(nn, act)(inplace=False))

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

        # low-rank bilinear pooling using einsum
        elif self.h_out <= self.c:
            v_ = self.dropout(self.v_net(v))
            q_ = self.q_net(q)
            logits = torch.einsum('xhyk,bvk,bqk->bhvq', (self.h_mat, v_, q_)) + self.h_bias
            return logits  # b x h_out x v x q

        # batch outer product, linear projection
        # memory efficient but slow computation
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
# BAN module end

# classification task
class SimpleClassifier(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout):
        super(SimpleClassifier, self).__init__()
        layers = [
            weight_norm(nn.Linear(in_dim, hid_dim), dim=None),
            nn.ReLU(inplace=False),
            nn.Dropout(dropout, inplace=False),
            weight_norm(nn.Linear(hid_dim, out_dim), dim=None)
        ]
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.main(x)
        return logits


class TransferLearningModel(nn.Module):
    def __init__(self, source_model, **config):
        super(TransferLearningModel, self).__init__()

        ban_heads = config["BCN"]["HEADS"]
        mlp_in_dim = config["DECODER"]["IN_DIM"]
        self.mlp_in_dim = mlp_in_dim
        mlp_hidden_dim = config["DECODER"]["HIDDEN_DIM"]
        mlp_out_dim = config["DECODER"]["OUT_DIM"]
        input_channels = config["PROTEIN"]["CHANNEL"]
        protein_emb_dim = config["PROTEIN"]["EMBEDDING_DIM"]
        num_filters = config["PROTEIN"]["NUM_FILTERS"]
        kernel_size = config["PROTEIN"]["KERNEL_SIZE"]
        num_classes = config["CLASSIFIER"]["NUM_CLASS"]
        ligand_hidden_feats = config["LIGAND"]["HIDDEN_LAYERS"]


        self.source_model = source_model
        self.ligand_extractor = self.source_model.ligand_extractor
        self.protein_extractor = self.source_model.protein_extractor

        self.ligand_extractor.eval()
        self.protein_extractor.eval()
        for param in self.ligand_extractor.parameters():
            param.requires_grad = False
        for param in self.protein_extractor.parameters():
            param.requires_grad = False

        self.bcn = self.source_model.bcn

        for param in self.bcn.parameters():
            param.requires_grad = False

        self.classifer = SimpleClassifier(mlp_in_dim, mlp_hidden_dim, num_classes, dropout = 0.2)


    def forward(self, substrate_features, p_globals, mode="train"):

        with torch.no_grad():
            v_d = self.ligand_extractor(substrate_features)

        with torch.no_grad():
            target_x, target_edge_index, target_batch = p_globals.x, p_globals.edge_index, 1
            v_p = self.protein_extractor(target_x, target_edge_index)

        bcn_output, att = self.bcn(v_d, v_p)

        # classification
        score = self.classifer(bcn_output)
        return score, att



def main():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')


    torch.cuda.empty_cache()
    cfg = get_cfg_defaults()
    set_seed(cfg.SOLVER.SEED)
    suffix = str(int(time() * 1000))[6:]
    print(f"Hyperparameters: {dict(cfg)}")
    print(f"Running on: {device}", end="\n\n")

    dataFolder = cfg.DIR.DATASET
    featureFolder = cfg.DIR.FEATURE
    outputFolder = cfg.RESULT.OUTPUT_DIR

    data_path = os.path.join(dataFolder, 'dataset.csv')

    df_data = pd.read_csv(data_path)

    threshold, center_residue_index, subgraph_radius = cfg.PROTEIN.THERSHOLD,  cfg.PROTEIN.CENTER, cfg.PROTEIN.RADIUS

    num_candidates = cfg.PROTEIN.NUM_CAND

    n_bootstraps = 100  
    results = []
    for i in range(n_bootstraps):
        with open(os.path.join(cfg.RESULT.OUTPUT_DIR, 'realtime_monitoring.txt'), 'a') as file:
            file.write('Bootstrap iteration {} / {}.\n'.format(i + 1, n_bootstraps))
        print(f"Bootstrap iteration {i + 1}/{n_bootstraps}")
        sample_indices = resample(np.arange(len(df_data)), replace=True)
        oob_indices = np.setdiff1d(np.arange(len(df_data)), sample_indices, assume_unique=True)
        train_dataset = loadData(sample_indices, df_data.iloc[sample_indices], featureFolder, center_residue_index,
                                 subgraph_radius, threshold, num_candidates)
        if len(oob_indices) > 0:
            val_dataset = loadData(oob_indices, df_data.iloc[oob_indices], featureFolder, center_residue_index,
                                   subgraph_radius, threshold, num_candidates)

        source_model = Main_model(**cfg, pretrained=True).to(device) 


        model_weights = torch.load(cfg.DIR.SOURCE + '/model')

        source_model.load_state_dict(model_weights)

        transfer_model = TransferLearningModel(source_model, **cfg).to(device)
        criterion = nn.CrossEntropyLoss()

        opt = torch.optim.Adam([
            {'params': transfer_model.classifer.parameters(), 'lr': cfg.SOLVER.DA_LR}
        ], lr=cfg.SOLVER.LR)


        trainer = Trainer(transfer_model, opt, device, train_dataset, val_dataset, i, **cfg)
        result = trainer.train()
        results.append(result)

    mean_performance = np.mean([r['accuracy'] for r in results])
    std_performance = np.std([r['accuracy'] for r in results])
    mean_AUC = np.mean([r['AUC'] for r in results])
    std_AUC = np.std([r['AUC'] for r in results])
    mean_RE5 = np.mean([r['RE5'] for r in results])
    std_RE5 = np.std([r['RE5'] for r in results])
    mean_RE1 = np.mean([r['RE1'] for r in results])
    std_RE1 = np.std([r['RE1'] for r in results])
    mean_RE2 = np.mean([r['RE2'] for r in results])
    std_RE2 = np.std([r['RE2'] for r in results])
    mean_RE05 = np.mean([r['RE05'] for r in results])
    std_RE05 = np.std([r['RE05'] for r in results])
    print(f"Mean Accuracy: {mean_performance}, Standard Deviation: {std_performance}")
    with open(os.path.join(cfg.RESULT.OUTPUT_DIR, 'realtime_monitoring.txt'), 'a') as file:
        file.write('Mean Accuracy: {}, SD Accuracy:  {}, Mean AUC: {}, SD AUC:  {}, Mean RE05: {}, SD RE05:  {}, Mean RE1: {}, SD RE1:  {}, Mean RE2: {}, SD RE2:  {}, Mean RE5: {}, SD RE5:  {}.\n'.format(mean_performance, std_performance, mean_AUC, std_AUC, mean_RE05, std_RE05, mean_RE1, std_RE1, mean_RE2, std_RE2, mean_RE5, std_RE5))

    with open(os.path.join(cfg.RESULT.OUTPUT_DIR, "model_architecture.txt"), "w+") as wf:
        wf.write(str(transfer_model))

    print("Source Model  Weights:")
    for name, param in source_model.protein_extractor.named_parameters():
        print(name, param.data)

    print("\nTransfer Learning Model  Weights:")
    for name, param in transfer_model.protein_extractor.named_parameters():
        print(name, param.data)



class Trainer(object):
    def __init__(self, model, optim, device, train_dataloader, val_dataloader, fold, **config):
        self.model = model
        self.optim = optim
        self.device = device
        self.epochs = config["SOLVER"]["MAX_EPOCH"]
        self.current_epoch = 0
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.batch_size = config["SOLVER"]["BATCH_SIZE"]
        self.nb_training = len(self.train_dataloader)
        self.step = 0
        self.criterion = nn.CrossEntropyLoss()
        self.best_model = model # Initialize with first model
        self.best_epoch = None
        self.best_accuracy = -float('inf')
        self.best_val_los = float('inf')
        self.best_AUC = 100
        self.best_RE5 = 100
        self.best_RE1 = 100
        self.best_RE2 = 100
        self.best_RE05 = 100
        self.fold = fold
        self.train_loss_epoch = []
        self.val_loss_epoch = []
        self.val_accuracy_epoch = []
        self.val_precision_epoch = []
        self.val_recall_epoch = []
        self.val_f1_epoch = []
        self.test_metrics = {}
        self.config = config
        self.output_dir = config["RESULT"]["OUTPUT_DIR"]

        train_metric_header = ["# Epoch", "Train_loss", "Train_Accuracy"]
        valid_metric_header = ["# Epoch", "Val_loss", "Val_Accuracy", "Val_Precision", "Val_Recall", "Val_f1"]

        self.train_table = PrettyTable(train_metric_header)
        self.val_table = PrettyTable(valid_metric_header)

    def train(self):
        float2str = lambda x: '%0.4f' % x
        for epoch in range(self.epochs):
            self.current_epoch = self.current_epoch + 1
            train_loss, train_accuracy = self.train_epoch()  
            val_loss, val_accuracy, val_precision, val_recall, val_f1, AUC, RE5, RE1, RE2, RE05 = self.test(dataloader="val")  # 更新test以返回准确率

            train_lst = ["Epoch " + str(self.current_epoch)] + list(map(float2str, [train_loss, train_accuracy]))
            self.train_table.add_row(train_lst)

            val_lst = ["Epoch " + str(self.current_epoch)] + list(map(float2str, [val_loss, val_accuracy, val_precision, val_recall, val_f1]))
            self.val_table.add_row(val_lst)

            self.train_loss_epoch.append(train_loss)
            self.val_loss_epoch.append(val_loss)
            self.val_accuracy_epoch.append(val_accuracy)
            self.val_precision_epoch.append(val_precision)
            self.val_recall_epoch.append(val_recall)
            self.val_f1_epoch.append(val_f1)

            if val_accuracy > self.best_accuracy:
                self.best_model.load_state_dict(self.model.state_dict())
                self.best_accuracy = val_accuracy
                self.best_epoch = self.current_epoch
                self.best_AUC = AUC
                self.best_RE5 = RE5
                self.best_RE1 = RE1
                self.best_RE2 = RE2
                self.best_RE05 = RE05

            with open(os.path.join(self.output_dir, 'realtime_monitoring.txt'), 'a') as file:
                file.write('Validation at Epoch {}: Loss: {}, Val Accuracy: {}, Precision: {}, Recall: {}, F1: {}, AUC: {}, RE5: {}, RE1 : {}, RE2 : {}, RE05: {}.\n'.format(self.current_epoch, val_loss, val_accuracy, val_precision, val_recall, val_f1, AUC, RE5, RE1, RE2, RE05))
            print('Validation at Epoch {}: Loss: {}, Val Accuracy: {}'.format(self.current_epoch, val_loss, val_accuracy))

            if val_loss < self.best_val_los:
                self.best_val_los = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= 5:
                    print(f"Early stopping triggered after {epoch + 1} epochs.")
                    with open(os.path.join(self.output_dir, 'realtime_monitoring.txt'), 'a') as file:
                        file.write('Early stopping triggered.')
                    break

        results = {
            'accuracy': self.best_accuracy,  # you could also use np.mean(self.val_accuracy_epoch) for average accuracy
            'loss': self.best_val_los,
            'precision': np.mean(self.val_precision_epoch),
            'recall': np.mean(self.val_recall_epoch),
            'f1': np.mean(self.val_f1_epoch),
            'AUC':self.best_AUC,
            'RE5':self.best_RE5,
            'RE1':self.best_RE1,
            'RE2':self.best_RE2,
            'RE05':self.best_RE05
        }
        self.save_result()
        return results

    def save_result(self):
        if self.config["RESULT"]["SAVE_MODEL"]:
            torch.save(self.best_model.state_dict(),
                       os.path.join(self.output_dir, f"best_model_epoch_{self.best_epoch}_boots_{self.fold}.pth"))
            torch.save(self.model.state_dict(), os.path.join(self.output_dir, f"model_epoch_{self.current_epoch}.pth"))
        state = {
            "train_loss": self.train_loss_epoch,
            "val_loss": self.val_loss_epoch,
            "val_accuracy": self.val_accuracy_epoch,
            "val_precision": self.val_precision_epoch,
            "val_recall": self.val_recall_epoch,
            "val_f1": self.val_f1_epoch,
            "test_metrics": self.test_metrics,
            "config": self.config
        }
        torch.save(state, os.path.join(self.output_dir, f"result_metrics.pt"))


    def train_epoch(self):
        self.model.train()
        loss_epoch = 0
        correct_predictions = 0
        total_predictions = 0
        num_batches = len(self.train_dataloader)


        for i, (bg_d, p_globals, labels) in enumerate(tqdm(self.train_dataloader)):
            self.step += 1
            bg_d = bg_d.to(self.device)
            labels = torch.Tensor([labels]).long().to(self.device)

            self.optim.zero_grad()
            score, f = self.model(bg_d, p_globals)
            loss = self.criterion(score, labels)
            loss.backward()
            self.optim.step()
            loss_epoch += loss.item()
            _, predictions = torch.max(score, 1)
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)

        loss_epoch = loss_epoch / num_batches
        accuracy = correct_predictions / total_predictions * 100
        print('Training at Epoch {}: Loss: {}  Accuracy: {}'.format(self.current_epoch, loss_epoch, accuracy))
        with open(os.path.join(self.output_dir, 'realtime_monitoring.txt'), 'a') as file:
            file.write(
                'raining at Epoch {}: Loss: {}  Accuracy: {}.\n'.format(self.current_epoch, loss_epoch, accuracy))

        return loss_epoch, accuracy

    def test(self, dataloader="val"):
        self.model.eval()
        total_loss = 0
        y_true, y_pred, y_scores = [], [], []
        correct_predictions = 0
        total_predictions = 0
        data_loader = self.val_dataloader

        num_batches = len(data_loader)
        with torch.no_grad():
            for i, (bg_d, p_globals, labels) in enumerate(data_loader):
                bg_d = bg_d.to(self.device)
                labels = torch.Tensor([labels]).long().to(self.device)

                if dataloader == "val":
                    score, f = self.model(bg_d, p_globals)

                loss = self.criterion(score, labels)
                total_loss += loss.item()
                _, predicted = torch.max(score.data, 1)
                total_predictions += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
                y_scores.extend(score.cpu().numpy()[:, 1])

        avg_loss = total_loss / len(data_loader)
        accuracy = 100 * (np.array(y_true) == np.array(y_pred)).mean()
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        f1 = f1_score(y_true, y_pred, average='macro')
        auc = roc_auc_score(y_true, y_scores)
        def relative_error(y_true, y_pred, threshold):
            y_true_bin = np.array(y_true) >= threshold
            y_pred_bin = np.array(y_pred) >= threshold
            abs_errors = np.logical_xor(y_true_bin, y_pred_bin)
            relative_errors = abs_errors / np.maximum(np.abs(y_true_bin), 1)  # Avoid division by zero
            return relative_errors.mean()

        re_05 = relative_error(y_true, y_scores, 0.005)
        re_1 = relative_error(y_true, y_scores, 0.01)
        re_2 = relative_error(y_true, y_scores, 0.02)
        re_5 = relative_error(y_true, y_scores, 0.05)

        return avg_loss, accuracy, precision, recall, f1,auc,re_5,re_1,re_2,re_05

if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    s = time()
    main()
    e = time()
    print(f"Total running time: {round(e - s, 2)}s")
