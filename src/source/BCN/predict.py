import torch
import pandas as pd
from model import Main_model
from feature_predict import SingleDataFeatureGenerator
from configs import get_cfg_defaults
from torch_geometric.data import Data, Batch
import numpy as np

import dgl

def load_model(model_path, config):

    model = Main_model(**config)
    model.load_state_dict(torch.load(model_path))
    model.eval()  
    return model


def predict_new_data(model, new_data, device, featureFolder, threshold):
    predictions = []
    attention_maps = []
    seqs = []
    for index, row in new_data.iterrows():
        feature_generator = SingleDataFeatureGenerator(smiles=row['substrate_smiles'], sequence=row['protein_seq'],
                                                       value=row['kcat'], pdb_id=row['pname'],
                                                       dir_path=featureFolder, threshold=8)
        ligand_graph, protein_graph, interaction, seq = feature_generator[0]


        ligand_batch = dgl.batch([ligand_graph])  
        protein_batch = Batch.from_data_list([protein_graph]) 


        ligand_batch = ligand_batch.to(device)
        protein_batch = protein_batch.to(device)


        _, _, score, att = model(ligand_batch, protein_batch, mode='eval')
        predictions.append(score.cpu().item())
        attention_maps.append(att.detach().cpu().numpy())
        seqs.append(seq)

    return predictions, attention_maps, seqs



def main_prediction():

    config = get_cfg_defaults()  
    model_path = 'path to model' 
    featureFolder = 'path to pdb' 
    threshold = 8 


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, config)


    path = 'path to predicted dataset'
    new_data_path = path + '/dataset.csv'  
    new_data = pd.read_csv(new_data_path)


    predictions, attention_maps, seqs = predict_new_data(model, new_data, device, featureFolder, threshold)

    n= len(predictions)
    for i in range(n):
        docking_name = str(new_data.loc[i, 'label'])
        with open(path + '/predict_results.txt', 'a') as pre_file:
            pre_file.write('{}:  {}\n'.format(docking_name, predictions[i]))




if __name__ == '__main__':
    main_prediction()
