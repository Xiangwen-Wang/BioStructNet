#!/usr/bin/python
# coding: utf-8
# author: Xiangwen Wang. Last edited: 2022-04-10
# This python script is to obtain canonical SMILES just by chemical name using PubChem API
import json
import io
from multiprocessing.dummy import Pool
from Bio import ExPASy
from Bio import SwissProt
from Bio.PDB import PDBList
from urllib.error import HTTPError


def get_pdb(uniprot_id):

    with ExPASy.get_sprot_raw(uniprot_id) as handle:
        data = handle.read().strip()  
        if not data:  #
            raise ValueError("No data found for UniProt ID.")
        
        handle = io.StringIO(data)
        record = SwissProt.read(handle)
        pdb_one_uni = [xref[1] for xref in record.cross_references if xref[0] == 'PDB' and len(xref) > 1]
        name_pdb[uniprot_id] = pdb_one_uni


name_pdb = dict()


def create_dic():

    with open('/Users/xw0201/Desktop/postdoc/CPI_structure_CALB/Large_Dataset/allec/KCAT/KCAT_uniseq.json', 'r') as file:
        datasets = json.load(file)
    UniprotID = [data['UniprotID'].upper() for data in datasets]

    unique_Uniprots = list(set(UniprotID))
    print(len(unique_Uniprots))

    thread_pool = Pool(4)
    thread_pool.map(get_pdb, unique_Uniprots)

    with open('/Users/xw0201/Desktop/postdoc/CPI_structure_CALB/Large_Dataset/allec/KCAT/uniprot_pdb_dic.json', 'w') as outfile:
         json.dump(name_pdb, outfile, indent=2)

def unique_pdb():
    with open('/Users/xw0201/Desktop/postdoc/CPI_structure_CALB/Large_Dataset/allec/KCAT/withPDBID.json', 'r') as infile:
         datasets = json.load(infile)
    PDBID = [data['PDBID'] for data in datasets]
    unique_pdb_ids = list(set(PDBID))
    newlist_data = []

    for data in datasets:
        i = data['PDBID']
        if i:
            new_data = {
                'UniprotID': data['UniprotID'],
                'Substrate': data['Substrate'].lower(),
                'SMILES': data['SMILES'],
                'Sequence': data['Sequence'],
                'Organism': data['Organism'].lower(),
                'EC_number': data['EC_number'],
                'Type': data['Type'],
                'PDBID': i,
                "Value": data["Value"]
            }
            newlist_data.append(new_data)

    count_non_empty_pdb = sum(1 for pdb_id in PDBID if pdb_id)
    print("Number of non-empty PDBID entries:", count_non_empty_pdb)

    print("The number of unique PDBID:", len(unique_pdb_ids)-1)

    # 将唯一的PDB ID写入文件
    with open('/Users/xw0201/Desktop/postdoc/CPI_structure_CALB/Large_Dataset/allec/KCAT/unique_pdb.txt', "w") as file:
        for pdb_id in unique_pdb_ids:
            if pdb_id:
                file.write(pdb_id + "\n")

    print("Unique PDB list has been saved.")

    with open('/Users/xw0201/Desktop/postdoc/CPI_structure_CALB/Large_Dataset/allec/KCAT/KCAT_withPDB.json', 'w') as outfile:
        json.dump(newlist_data, outfile, indent=4)


def updated_with_pdb():
    with open('/Users/xw0201/Desktop/postdoc/CPI_structure_CALB/Large_Dataset/allec/KCAT/KCAT_uniseq.json', 'r') as file:
        datasets = json.load(file)

    with open('/Users/xw0201/Desktop/postdoc/CPI_structure_CALB/Large_Dataset/allec/KCAT/uniprot_pdb_dic.json',
              'r') as infile:
        uniprot_pdb_dic = json.load(infile)

    combined_data = []


    for data in datasets:
        UniprotID = data['UniprotID'].upper()
        PDBID = uniprot_pdb_dic.get(UniprotID, "")  
        if not PDBID:
            pdb = None
        else:
            pdb = PDBID[0]

        new_data = {
            'UniprotID': UniprotID,
            'Substrate': data['Substrate'].lower(),
            'SMILES': data['Smiles'],
            'Sequence': data['Sequence'],
            'Organism': data['Organism'].lower(),
            'EC_number': data['ECNumber'],
            'Type': data['Type'],
            'PDBID': pdb,
            "Value": data["Value"]
        }


        combined_data.append(new_data)


    with open('/Users/xw0201/Desktop/postdoc/CPI_structure_CALB/Large_Dataset/allec/KCAT/withPDBID.json',
              'w') as outfile:
        json.dump(combined_data, outfile, indent=4)


if __name__ == '__main__':
    create_dic()
    updated_with_pdb()
    unique_pdb()