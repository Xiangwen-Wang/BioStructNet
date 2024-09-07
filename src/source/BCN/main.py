from model import Main_model
from time import time
from configs import get_cfg_defaults
from feature import loadData
from torch_geometric.data import Data, Batch
import pickle
from torch.utils.data import DataLoader
import torch
import dgl
import torch.nn as nn
import argparse
import pandas as pd
import random
import copy
import os
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from prettytable import PrettyTable
from tqdm import tqdm
count = 0
"""CPU or GPU."""
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')



def set_seed(seed=1000):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def main():
    torch.cuda.empty_cache()
    cfg = get_cfg_defaults()
    set_seed(cfg.SOLVER.SEED)
    suffix = str(int(time() * 1000))[6:]
    print(f"Hyperparameters: {dict(cfg)}")
    print(f"Running on: {device}", end="\n\n")


    dataFolder = cfg.DIR.DATASET
    featureFolder = cfg.DIR.FEATURE
    outputFolder = cfg.RESULT.OUTPUT_DIR
    train_path = os.path.join(dataFolder, 'train.csv')
    val_path = os.path.join(dataFolder, 'val.csv')
    test_path = os.path.join(dataFolder, 'test.csv')
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)


    threshold = cfg.PROTEIN.THERSHOLD #contact map distance
    train_dataset = loadData(df_train.index.values, df_train, featureFolder, threshold)
    val_dataset = loadData(df_val.index.values, df_val, featureFolder, threshold)
    test_dataset = loadData(df_test.index.values, df_test, featureFolder, threshold)


    torch.manual_seed(cfg.SOLVER.SEED)
    model = Main_model(**cfg).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.LR)
    trainer = Trainer(model, opt, device, train_dataset, val_dataset, test_dataset, **cfg)
    result = trainer.train()

    with open(os.path.join(cfg.RESULT.OUTPUT_DIR, "model_architecture.txt"), "w+") as wf:
        wf.write(str(model))

    print()
    print(f"Directory for saving result: {cfg.RESULT.OUTPUT_DIR}")

    return result

def load_features(dir):
    with open(dir+'v_d.pkl', 'rb') as data_file:
        v_d = pickle.load(data_file)
    with open(dir+'protein_graph.pkl', 'rb') as data_file:
        protein_graph = pickle.load(data_file)
    interactions = load_tensor(dir + 'interactions.npy', torch.FloatTensor)
    dataset = list(zip(v_d, protein_graph, interactions))
    return dataset

def load_tensor(file_name, dtype):
    return [dtype(d).to(device) for d in np.load(file_name,allow_pickle = True)]


class Trainer(object):
    def __init__(self, model, optim, device, train_dataloader, val_dataloader, test_dataloader, **config):
        self.model = model
        self.optim = optim
        self.device = device
        self.epochs = config["SOLVER"]["MAX_EPOCH"]
        self.current_epoch = 0
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.batch_size = config["SOLVER"]["BATCH_SIZE"]
        self.nb_training = len(self.train_dataloader)
        self.step = 0

        self.best_model = None
        self.best_epoch = None
        self.best_r2_score = -float('inf')  

        self.train_loss_epoch = []
        self.val_loss_epoch = []
        self.val_rmse_epoch = []
        self.test_metrics = {}
        self.config = config
        self.output_dir = config["RESULT"]["OUTPUT_DIR"]

        train_metric_header = ["# Epoch", "Train_loss"]
        valid_metric_header = ["# Epoch", "rmse", "r2", "Val_loss"]

        self.train_table = PrettyTable(train_metric_header)
        self.val_table = PrettyTable(valid_metric_header)

    def train(self):
        float2str = lambda x: '%0.4f' % x
        for i in range(self.epochs):
            self.current_epoch += 1
            train_loss = self.train_epoch()
            train_lst = ["epoch " + str(self.current_epoch)] + list(map(float2str, [train_loss]))
            self.train_table.add_row(train_lst)
            self.train_loss_epoch.append(train_loss)

            rmse, r2_score_val, val_loss = self.test(dataloader="val")
            val_lst = ["epoch " + str(self.current_epoch)] + list(map(float2str, [rmse, r2_score_val, val_loss]))
            self.val_table.add_row(val_lst)
            self.val_loss_epoch.append(val_loss)
            self.val_rmse_epoch.append(rmse)

            if r2_score_val >= self.best_r2_score: 
                self.best_model = copy.deepcopy(self.model)
                self.best_r2_score = r2_score_val
                self.best_epoch = self.current_epoch
                self.save_model(self.model, self.output_dir)

            with open(os.path.join(self.output_dir, 'realtime_monitoring.txt'), 'a') as file:
                file.write('Validation at Epoch {}: Loss: {}, R2 Score: {}, RMSE Score: {}\n'.format(self.current_epoch, val_loss, r2_score_val, rmse))
            print('Validation at Epoch {}: Loss: {}, R2 Score: {}, RMSE Score: {}'.format(self.current_epoch, val_loss, r2_score_val, rmse))

        self.test_metrics = self.test()
        self.save_final()
        return self.test_metrics

    def save_final(self):
        state = {
            "train_epoch_loss": self.train_loss_epoch,
            "val_epoch_loss": self.val_loss_epoch,
            "test_metrics": self.test_metrics,
            "config": self.config
        }
        torch.save(state, os.path.join(self.output_dir, f"result_metrics.pt"))

        val_prettytable_file = os.path.join(self.output_dir, "valid_markdowntable.txt")
        train_prettytable_file = os.path.join(self.output_dir, "train_markdowntable.txt")
        with open(val_prettytable_file, 'w') as fp:
            fp.write(self.val_table.get_string())
        with open(train_prettytable_file, "w") as fp:
            fp.write(self.train_table.get_string())

    def train_epoch(self):
        self.model.train()
        loss_epoch = 0
        num_batches = len(self.train_dataloader)
        criterion = nn.MSELoss()

        for i, (bg_d, p_globals, labels) in enumerate(tqdm(self.train_dataloader)):
            self.step += 1
            bg_d = bg_d.to(self.device)
            labels = torch.Tensor([labels]).float().to(self.device)

            self.optim.zero_grad()
            v_d, v_p, f, score = self.model(bg_d, p_globals)
            loss = criterion(score, labels)
            loss.backward()
            self.optim.step()
            loss_epoch += loss.item()

        loss_epoch = loss_epoch / num_batches
        print('Training at Epoch {}: Loss: {}'.format(self.current_epoch, loss_epoch))

        return loss_epoch

    def test(self, dataloader="test"):
        self.model.eval()
        test_loss = 0
        y_true, y_pred = [], []
        if dataloader == "test":
            data_loader = self.test_dataloader
        elif dataloader == "val":
            data_loader = self.val_dataloader

        num_batches = len(data_loader)
        with torch.no_grad():
            criterion = nn.MSELoss()
            for i, (bg_d, p_globals, labels) in enumerate(data_loader):
                bg_d = bg_d.to(self.device)
                labels = torch.Tensor([labels]).float().to(self.device)

                if dataloader == "val":
                    v_d, v_p, f, score = self.model(bg_d, p_globals)
                elif dataloader == "test":
                    v_d, v_p, f, score = self.best_model(bg_d, p_globals)

                loss = criterion(score, labels)
                test_loss += loss.item()

                y_true.extend(labels.cpu().numpy())
                y_pred.extend(score.cpu().numpy())

            test_loss = test_loss / len(self.test_dataloader)
            r2_score_test = r2_score(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mae ** 2)

        if dataloader == "test":
            test_metrics = {
                "test_loss": test_loss,
                "r2_score": r2_score_test,
                "mae": mae,
                "rmse": rmse,
                "best_epoch": self.best_epoch
            }
            return test_metrics
        if dataloader == "val":
            return rmse, r2_score_test, test_loss

    def save_model(self, model, dir):
        filename = dir + '/model'
        torch.save(model.state_dict(), filename)

if __name__ == '__main__':
    s = time()
    result = main()
    e = time()
    print(f"Total running time: {round(e - s, 2)}s")
