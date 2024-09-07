#!/usr/bin/python
# coding: utf-8

# This python script is to obtain canonical SMILES just by chemical name using PubChem API
import json
import time
import requests
import multiprocessing as mp
from multiprocessing.dummy import Pool
import codecs
from pubchempy import Compound, get_compounds


name_smiles = dict()

# One method to obtain SMILES by PubChem API using the website
def get_smiles(name):
    # smiles = redis_cli.get(name)
    # if smiles is None:
    try :
        url = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/%s/property/CanonicalSMILES/TXT' % name
        req = requests.get(url)
        if req.status_code != 200:
            smiles = None
        else:
            smiles = req.content.splitlines()[0].decode()

    except :
        smiles = None

    name_smiles[name] = smiles


def main():


    with codecs.open("/Users/xw0201/Desktop/postdoc/CPI_structure_CALB/Large_Dataset/allec/KCAT/KCAT_sabio_clean.tsv", "r", encoding='utf-8') as file :
        lines = file.readlines()[1:]

    substrates = [line.strip().split('\t')[2] for line in lines]

    print(len(substrates)) # 18243

    names = list(set(substrates))
    print(len(names))  # 3100


    thread_pool = Pool(4)
    thread_pool.map(get_smiles, names)

    with open('/Users/xw0201/Desktop/postdoc/CPI_structure_CALB/Large_Dataset/allec/KCAT/Sabio_smiles.json', 'w') as outfile:
        json.dump(name_smiles, outfile, indent=2)


if __name__ == '__main__':
    main()