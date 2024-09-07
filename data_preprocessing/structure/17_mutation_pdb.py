import json
from Bio import pairwise2
from Bio import PDB
from Bio.SubsMat import MatrixInfo as matlist
from Bio.pairwise2 import format_alignment
import os


def mutation_seq(uni_seq, pdb_seq, mutation):
    matrix = matlist.blosum62
    i = 0
    j = 0
    t = 0
    mutation = mutation[7:]
    output_seq = ''
    mutation_list = mutation.split('/')
    for a in pairwise2.align.globaldx(uni_seq, pdb_seq, matrix):
        if i == 0:
            print('****************************************')
            align = format_alignment(*a)
            seq_list = align.split('\n')
            i = i+1
            uni_align = seq_list[0]
            pdb_align = seq_list[2]
            print(mutation)
            print(uni_align)
            print(pdb_align)
            for char in uni_align:
                j = j+1
                if char != '-':
                    t = t+1
                for mutantSite in mutation_list:
                    if t == int(mutantSite[1:-1]):
                        pdb_align = list(pdb_align)
                        if char == pdb_align[j-1]:
                            pdb_align[j-1] = mutantSite[-1]
                            output_seq = ''.join(pdb_align)
                        else:
                            output_seq = ''
                        # 在pdb fasta里面能和这个位置wildtype对上的才mutation，对不上如果是'-'的话就会记录none然后clean掉。
    output_seq = output_seq.replace("-", "")
    return output_seq


def get_pdb_fasta(pname,dir):
    pdb_file = dir + pname + '-ChainA.pdb'
    if os.path.exists(pdb_file) and os.path.getsize(pdb_file) > 0:
        dictionary = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
             'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
             'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
             'ALA': 'A', 'VAL': 'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}
        print(pname)
        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure(pname, pdb_file)
        res_list = structure.get_residues()
        seq = []
        for residue in res_list:
            seq.append(dictionary[residue.resname])
        seq =''.join(seq)
        sequence = seq
    else:
        sequence = ''
    return sequence

if __name__ == '__main__':

    pdb_dir = '/Users/xw0201/Desktop/postdoc/CPI_structure_CALB/Large_Dataset/BRENDA/pdb/'
    new_list = []
    with open('/Users/xw0201/Desktop/postdoc/CPI_structure_CALB/Large_Dataset/BRENDA/KM/KM_withPDB.json', 'r') as infile:
        datasets = json.load(infile)

    for data in datasets:
        PDBID = data['PDBID']
        pdb_seq = get_pdb_fasta(PDBID, pdb_dir)
        if pdb_seq:
            if data['Type'] == 'wildtype':
                new_data = {
                    'UniprotID': data['UniprotID'],
                    'Substrate': data['Substrate'],
                    'SMILES': data['SMILES'],
                    'Sequence': pdb_seq,
                    'Organism': data['Organism'],
                    'EC_number': data['EC_number'],
                    'Type': data['Type'],
                    'PDBID': data['PDBID'],
                    'Value': data['Value']
                }
            else:
                mutation_sequence = mutation_seq(data['Sequence'], pdb_seq, data['Type'])
                if mutation_sequence:
                    new_data = {
                        'UniprotID': data['UniprotID'],
                        'Substrate': data['Substrate'],
                        'SMILES': data['SMILES'],
                        'Sequence': mutation_sequence,
                        'Organism': data['Organism'],
                        'EC_number': data['EC_number'],
                        'Type': data['Type'],
                        'PDBID': data['PDBID'],
                        'Value': data['Value']
                    }
            new_list.append(new_data)

    with open('/Users/xw0201/Desktop/postdoc/CPI_structure_CALB/Large_Dataset/BRENDA/KM/KM_PDBseq.json',
                  'w') as outfile:
        json.dump(new_list, outfile, indent=4)

