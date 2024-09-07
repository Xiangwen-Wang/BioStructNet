from Bio import PDB
import dgl
import numpy as np
from moleculekit.tools.voxeldescriptors import getVoxelDescriptors,getCenters, rotateCoordinates
from moleculekit.tools.atomtyper import prepareProteinForAtomtyping
from moleculekit.readers import Molecule
import torch.utils.data as data
import torch
from functools import partial
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer
from torch_geometric import data as DATA

def extract_subgraph(structure, center_residue_id, radius, threshold):
    """
    Extracts a subgraph centered at a specific residue within a given radius.

    :param structure: PDB structure as parsed by BioPython PDBParser.
    :param center_residue_id: The identifier for the center residue (e.g., (' ', 105, ' ')).
    :param radius: Radius in angstroms to consider for the subgraph.
    :return: A tuple containing the indices of nodes within the radius and the adjacency matrix for these nodes.
    """
    # Extract Cα atoms and their coordinates
    ca_atoms_by_residue = {}
    residue_ids = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.has_id("CA"):
                    residue_id = residue.get_id()
                    residue_ids.append(residue_id)
                    ca_atom = residue["CA"]
                    ca_atoms_by_residue[residue_id] = ca_atom.get_coord()

    # Find the center residue's coordinates
    center_coord = ca_atoms_by_residue.get(center_residue_id)
    if center_coord is None:
        raise ValueError("Center residue ID not found in structure.")

    # Find all residues within the specified radius
    nearby_residues = []
    for residue_id, coord in ca_atoms_by_residue.items():
        if np.linalg.norm(coord - center_coord) <= radius:
            nearby_residues.append(residue_id)

    # Create a subgraph adjacency matrix
    num_residues = len(nearby_residues)
    adjacency_matrix = np.zeros((num_residues, num_residues), dtype=int)

    # Map old indices to new indices in the subgraph
    index_map = {residue_id: index for index, residue_id in enumerate(nearby_residues)}

    # Build adjacency matrix for the subgraph
    for i, residue_id_i in enumerate(nearby_residues):
        coord_i = ca_atoms_by_residue[residue_id_i]
        for j, residue_id_j in enumerate(nearby_residues):
            coord_j = ca_atoms_by_residue[residue_id_j]
            distance = np.linalg.norm(coord_i - coord_j)
            if distance <= threshold:
                adjacency_matrix[i, j] = 1
                adjacency_matrix[j, i] = 1

    return index_map, adjacency_matrix

# Load the PDB file for the structure
def load_pdb(pdb_file, pname):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure(pname, pdb_file)
    return structure

def protein_features(data_dir, pname, candidate, target_sequence, center_residue_index, subgraph_radius, threshold):
    structure_dir = data_dir + '/md_pdb/'
    candidate = candidate + 1
    candidate_file = structure_dir + pname + '.pdb'
    candidate_structure = load_pdb(candidate_file, pname)
    target_size, target_features, target_edge_index = protein_global(pname, candidate_structure, target_sequence, center_residue_index, subgraph_radius, threshold, data_dir)
    return target_size, target_features, target_edge_index


def dic_normalize(dic):
    # print(dic)
    max_value = dic[max(dic, key=dic.get)]
    min_value = dic[min(dic, key=dic.get)]
    # print(max_value)
    interval = float(max_value) - float(min_value)
    for key in dic.keys():
        dic[key] = (dic[key] - min_value) / interval
    dic['X'] = (max_value + min_value) / 2.0
    return dic

def residue_features(residue):

    pro_res_aliphatic_table = ['A', 'I', 'L', 'M', 'V']
    pro_res_aromatic_table = ['F', 'W', 'Y']
    pro_res_polar_neutral_table = ['C', 'N', 'Q', 'S', 'T']
    pro_res_acidic_charged_table = ['D', 'E']
    pro_res_basic_charged_table = ['H', 'K', 'R']

    res_weight_table = {'A': 71.08, 'C': 103.15, 'D': 115.09, 'E': 129.12, 'F': 147.18, 'G': 57.05, 'H': 137.14,
                        'I': 113.16, 'K': 128.18, 'L': 113.16, 'M': 131.20, 'N': 114.11, 'P': 97.12, 'Q': 128.13,
                        'R': 156.19, 'S': 87.08, 'T': 101.11, 'V': 99.13, 'W': 186.22, 'Y': 163.18}

    res_pka_table = {'A': 2.34, 'C': 1.96, 'D': 1.88, 'E': 2.19, 'F': 1.83, 'G': 2.34, 'H': 1.82, 'I': 2.36,
                     'K': 2.18, 'L': 2.36, 'M': 2.28, 'N': 2.02, 'P': 1.99, 'Q': 2.17, 'R': 2.17, 'S': 2.21,
                     'T': 2.09, 'V': 2.32, 'W': 2.83, 'Y': 2.32}

    res_pkb_table = {'A': 9.69, 'C': 10.28, 'D': 9.60, 'E': 9.67, 'F': 9.13, 'G': 9.60, 'H': 9.17,
                     'I': 9.60, 'K': 8.95, 'L': 9.60, 'M': 9.21, 'N': 8.80, 'P': 10.60, 'Q': 9.13,
                     'R': 9.04, 'S': 9.15, 'T': 9.10, 'V': 9.62, 'W': 9.39, 'Y': 9.62}

    res_pkx_table = {'A': 0.00, 'C': 8.18, 'D': 3.65, 'E': 4.25, 'F': 0.00, 'G': 0, 'H': 6.00,
                     'I': 0.00, 'K': 10.53, 'L': 0.00, 'M': 0.00, 'N': 0.00, 'P': 0.00, 'Q': 0.00,
                     'R': 12.48, 'S': 0.00, 'T': 0.00, 'V': 0.00, 'W': 0.00, 'Y': 0.00}

    res_pl_table = {'A': 6.00, 'C': 5.07, 'D': 2.77, 'E': 3.22, 'F': 5.48, 'G': 5.97, 'H': 7.59,
                    'I': 6.02, 'K': 9.74, 'L': 5.98, 'M': 5.74, 'N': 5.41, 'P': 6.30, 'Q': 5.65,
                    'R': 10.76, 'S': 5.68, 'T': 5.60, 'V': 5.96, 'W': 5.89, 'Y': 5.96}


    res_weight_table = dic_normalize(res_weight_table)
    res_pka_table = dic_normalize(res_pka_table)
    res_pkb_table = dic_normalize(res_pkb_table)
    res_pkx_table = dic_normalize(res_pkx_table)
    res_pl_table = dic_normalize(res_pl_table)

    res_property2 = [res_weight_table[residue], res_pka_table[residue], res_pkb_table[residue], res_pkx_table[residue],
                     res_pl_table[residue]] 

    return np.array(res_property2)


def seq_feature(pro_seq):
    pro_res_table = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
                     'X']
    pro_hot = np.zeros((len(pro_seq), len(pro_res_table)))
    pro_property = np.zeros((len(pro_seq), 5))
    for i in range(len(pro_seq)):

        pro_hot[i,] = one_of_k_encoding(pro_seq[i], pro_res_table)
        pro_property[i,] = residue_features(pro_seq[i])
    return np.concatenate((pro_hot, pro_property), axis=1)  

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        # print(x)
        raise Exception('input {0} not in allowable set{1}:'.format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def global_node_feature(target_key, target_sequence, dir):
    feature = seq_feature(target_sequence)
    return feature


def protein_global(target_key, target_structure, target_sequence, center_residue_index, subgraph_radius, threshold, data_dir):

    center_residue_id = (' ', center_residue_index, ' ')  

    
    index_map, subgraph = extract_subgraph(target_structure, center_residue_id, subgraph_radius, threshold)


    subgraph_sequence = [target_sequence[idx] for idx in sorted(index_map.values())]
    subgraph_seq_line =  ''.join(subgraph_sequence)

    node_feature = global_node_feature(target_key, ''.join(subgraph_sequence), data_dir)

    target_edge_index = []
    index_row, index_col = np.where(subgraph >= 0.5)
    for i, j in zip(index_row, index_col):
        target_edge_index.append([i, j])

    target_size = len(subgraph_sequence)
    target_edge_index = np.array(target_edge_index)

    return target_size, node_feature, target_edge_index



class ligand_features(data.Dataset):
    def __init__(self):
        super(ligand_features, self).__init__()
        self.atom_featurizer = CanonicalAtomFeaturizer()
        self.bond_featurizer = CanonicalBondFeaturizer(self_loop=True)
        self.fc = partial(smiles_to_bigraph, add_self_loop=True)
    def forward(self, ligand_smiles):
        # Generate features of ligand
        max_ligand_nodes = 290
        v_d = self.fc(smiles=ligand_smiles, node_featurizer=self.atom_featurizer, edge_featurizer=self.bond_featurizer)
        actual_node_feats = v_d.ndata.pop('h')
        num_actual_nodes = actual_node_feats.shape[0]
        num_virtual_nodes = self.max_ligand_nodes - num_actual_nodes
        virtual_node_bit = torch.zeros([num_actual_nodes, 1])
        actual_node_feats = torch.cat((actual_node_feats, virtual_node_bit), 1)
        v_d.ndata['h'] = actual_node_feats
        virtual_node_feat = torch.cat((torch.zeros(num_virtual_nodes, 74), torch.ones(num_virtual_nodes, 1)), 1)
        v_d.add_nodes(num_virtual_nodes, {"h": virtual_node_feat})
        v_d = v_d.add_self_loop()
        return v_d


class loadData(data.Dataset):
    def __init__(self, list_IDs, df, dir_path, center_residue_index, subgraph_radius, threshold, num_candidates, max_ligand_nodes=290):
        self.list_IDs = list_IDs
        self.df = df
        self.max_ligand_nodes = max_ligand_nodes
        self.structure_dir = dir_path
        self.threshold = threshold
        self.center_residue_index=center_residue_index
        self.subgraph_radius = subgraph_radius
        self.num_candiates = num_candidates
        self.atom_featurizer = CanonicalAtomFeaturizer()
        self.bond_featurizer = CanonicalBondFeaturizer(self_loop=True)
        self.fc = partial(smiles_to_bigraph, add_self_loop=True)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        index = self.list_IDs[index]
        num_graph_types = self.num_candiates  

        threshold_distance = self.threshold
        center_residue_index, subgraph_radius = self.center_residue_index, self.subgraph_radius
        dir = self.structure_dir


        y = self.df.iloc[index]["Value"]
        v_d = self.df.iloc[index]['Smiles']
        v_d = self.fc(smiles=v_d, node_featurizer=self.atom_featurizer, edge_featurizer=self.bond_featurizer)
        actual_node_feats = v_d.ndata.pop('h')
        num_actual_nodes = actual_node_feats.shape[0]
        num_virtual_nodes = self.max_ligand_nodes - num_actual_nodes
        virtual_node_bit = torch.zeros([num_actual_nodes, 1])
        actual_node_feats = torch.cat((actual_node_feats, virtual_node_bit), 1)
        v_d.ndata['h'] = actual_node_feats
        virtual_node_feat = torch.cat((torch.zeros(num_virtual_nodes, 74), torch.ones(num_virtual_nodes, 1)), 1)
        v_d.add_nodes(num_virtual_nodes, {"h": virtual_node_feat})
        v_d = v_d.add_self_loop()


        p_name = self.df.iloc[index]['PDBID'] 
        protein_sequence = self.df.iloc[index]['Sequence']  
        num_rows = self.df.shape[0]
        p_globals = []
        p_locals = []
        for j in range(num_graph_types):
            g_size, g_features, g_edge_index = protein_features(dir, p_name, j, protein_sequence, center_residue_index, subgraph_radius, threshold_distance)
            protein_graph = DATA.Data(x=torch.Tensor(g_features),
                                    edge_index=torch.LongTensor(g_edge_index).transpose(1, 0),
                                    y=torch.FloatTensor([y]))

            p_globals.append(protein_graph)


        p_globals = [pg.to('cuda' if torch.cuda.is_available() else 'cpu') for pg in p_globals]

        return v_d, p_globals[0], float(y)



