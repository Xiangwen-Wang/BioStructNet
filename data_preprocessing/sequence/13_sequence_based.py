#!/usr/bin/python
# coding: utf-8

# This python script is to obtain protein sequence for each Kcat entries

import os
import re
import csv
import json
import requests
import time
import urllib2
from SOAPpy import SOAPProxy
import hashlib
import codecs
# import string
# import hashlib
# from SOAPpy import WSDL
# from SOAPpy import SOAPProxy ## for usage without WSDL file


# This function is to obtain the protein sequence according to the protein id from Uniprot API
# https://www.uniprot.org/uniprot/A0A1D8PIP5.fasta
# https://www.uniprot.org/help/api_idmapping
def uniprot_sequence(id) :
    url = "https://www.uniprot.org/uniprot/{}.fasta".format(id)
    IdSeq = dict()

    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.text
            respdata = data.strip()
            IdSeq[id] = "".join(respdata.split("\n")[1:])
        else:
            print("HTTP request failed with status code:", response.status_code)
    except requests.exceptions.RequestException as e:
        print("An error occurred:", e)

    return IdSeq[id]

def uniprotID_entry() :
    with codecs.open("/Users/xw0201/Desktop/postdoc/CPI_structure_CALB/Large_Dataset/BRENDA/KcatKm/Km_combination.tsv", "r", encoding='utf-8') as file :
        combination_lines = file.readlines()[1:]

    uniprotID_list = list()
    uniprotID_seq = dict()
    uniprotID_noseq = list()

    i=0
    for line in combination_lines :
        data = line.strip().split('\t')
        uniprotID = data[5]

        if uniprotID :
            if ' ' in uniprotID:
                uniprotID_list += uniprotID.split(' ')
            else :
                uniprotID_list.append(uniprotID)
    print('uniprotID')
    print(len(uniprotID_list))
    uniprotID_unique = list(set(uniprotID_list))
    print('unique uniprotID')
    print(len(uniprotID_unique))

    for uniprotID in uniprotID_unique :
        i += 1
        print(i)
        sequence = uniprot_sequence(uniprotID)
        if sequence :
            uniprotID_seq[uniprotID] = sequence
        else :
            uniprotID_noseq.append(uniprotID)

    print('sequence')
    print(len(uniprotID_seq))

    with open('/Users/xw0201/Desktop/postdoc/CPI_structure_CALB/Large_Dataset/BRENDA/KcatKm/uniprotID_entry.json', 'w') as outfile :
        json.dump(uniprotID_seq, outfile, indent=4)

def uniprotID_noseq() :
    with open('/Users/xw0201/Desktop/postdoc/CPI_structure_CALB/Large_Dataset/BRENDA/KcatKm/uniprotID_entry.json', 'r') as infile :
        uniprotID_seq = json.load(infile)

    print(len(uniprotID_seq))

    uniprotID_noseq = {'P0A5R0':'P9WIL4', 'P0C5C1':'P9WKD2', 'P51698':'A0A1L5BTC1', 'P96807':'P9WNP2', 'Q01745':'I1S2N3', 'P00892':'P0DP89', 
    'Q02469':'P0C278', 'P96223':'P9WNF8', 'P0A4Z2':'P9WPY2', 'P0A4X4':'P9WQ86', 'P96420':'P9WQB2', 'Q47741':'F2MMN9', 'O05783':'P9WIQ2', 
    'P0A4X6':'P9WQ80', 'P56967':'F2MMP0', 'O60344':'P0DPD6', 'P04804':'P60906', 'O52310':'P0CL72'}

    for uniprotID, mappedID in uniprotID_noseq.items() :
        sequence = uniprot_sequence(mappedID)
        print(uniprotID)
        print(sequence)
        if sequence :
            uniprotID_seq[uniprotID] = sequence
        else :
            print('No sequence found!---------------------------')

    print(len(uniprotID_seq))

    with open('/Users/xw0201/Desktop/postdoc/CPI_structure_CALB/Large_Dataset/BRENDA/KcatKm/uniprotID_entry_all.json', 'w') as outfile :
        json.dump(uniprotID_seq, outfile, indent=4)


def seq_by_ec_organism(ec, organism) :
    IdSeq = dict()

    params = {"query": "ec:%s AND organism:%s AND reviewed:yes" % (ec, organism), "format": "fasta"}
    response = requests.get("http://www.uniprot.org/uniprot/", params=params)


    try :
        # respdata = response.text.strip()
        # # print(respdata) 
        # IdSeq[ec+'&'+organism] =  "".join(respdata.split("\n")[1:])

        respdata = response.text
        # print(respdata)
        sequence = list()
        seq = dict()
        i = 0
        for line in respdata.split('\n') :
            if line.startswith('>') :
                name=line
                seq[name] = ''
            else :
                seq[name] += line.replace('\n', '').strip()
        IdSeq[ec+'&'+organism] =  list(seq.values())

    except :
        print(ec+'&'+organism, "can not find from uniprot!")
        IdSeq[ec+'&'+organism] = None
    print(IdSeq[ec+'&'+organism])
    return IdSeq[ec+'&'+organism]

# Run in python 2.7
def seq_by_brenda(ec, organism) :
    # E-mail in BRENDA:
    email = 'shuohan_liU@outlook.com'
    # Password in BRENDA:
    password = 'wxw9521'

    endpointURL = "https://www.brenda-enzymes.org/soap/brenda_server.php"
    client      = SOAPProxy(endpointURL)
    password    = hashlib.sha256(password).hexdigest()
    credentials = email + ',' + password

    parameters = credentials+","+"ecNumber*%s#organism*%s" %(ec, organism)
    entries = client.getSequence(parameters)
    sequences = list()
    if entries :
        parts = entries.split('#')
        for part in parts:
            if part[:8] == 'sequence':
                sequences.append(part[9:])

    return sequences

def nouniprotID_entry_uniprot() :


    with codecs.open("/Users/xw0201/Desktop/postdoc/CPI_structure_CALB/Large_Dataset/BRENDA/KcatKm/Km_combination.tsv", "r", encoding='utf-8') as file :
        combination_lines = file.readlines()[1:]

    IdSeq = dict()
    entries = list()
    i=0
    for line in combination_lines :
        data = line.strip().split('\t')
        ec = data[0]
        organism = data[2]
        uniprotID = data[5]

        if not uniprotID :
            entries.append((ec,organism))

    # print(len(entries))  # 28104
    entries_unique = set(entries)
    # print(len(entries_unique)) # 7258

    for entry in list(entries_unique) :
        # print(entry)
        ec, organism = entry[0], entry[1]
        i += 1
        print('This is', str(i)+'------------')
        IdSeq[ec+'&'+organism] = seq_by_ec_organism(ec, organism)
    # print(len(IdSeq)
        if i%10 == 0 :
            time.sleep(3)

    with open('/Users/xw0201/Desktop/postdoc/CPI_structure_CALB/Large_Dataset/BRENDA/KcatKm/nouniprotID_entry_all.json', 'w') as outfile :
        json.dump(IdSeq, outfile, indent=4)

# Run in python 2.7
def nouniprotID_entry_brenda() :
    with open("/Users/xw0201/Desktop/postdoc/CPI_structure_CALB/Large_Dataset/BRENDA/KcatKm/Km_combination.tsv", "r") as file :
        combination_lines = file.readlines()[1:]

    IdSeq = dict()
    entries = list()
    i=0
    j=0
    batch_size = 10
    for line in combination_lines :
        data = line.strip().split('\t')
        ec = data[0]
        organism = data[2]
        uniprotID = data[5]

        if not uniprotID:
            entries.append((ec,organism))

    # print(len(entries))  # 28104
    entries_unique = set(entries)
    print(len(entries_unique))

    new_list =list()

    for entry in list(entries_unique) :
        # print(entry)
        ec, organism = entry[0], entry[1]
        i += 1
        new_list.append((ec, organism))
        print('This is', str(j)+'.'+str(i)+'------------', ec+' '+organism)
        # print(ec)
        # print(organism)
        if i>=batch_size:
            process_batch(new_list, IdSeq)
            new_list = []
            j += 1
            i=0
    if new_list:
        process_batch(new_list, IdSeq)

    with open('/Users/xw0201/Desktop/postdoc/CPI_structure_CALB/Large_Dataset/BRENDA/KcatKm/nouniprotID_entry_brenda.json', 'w') as outfile :
        json.dump(IdSeq, outfile, indent=4)

def process_batch(entries, IdSeq):
    batch_results = {}
    for entry in entries:
        ec, organism = entry[0], entry[1]
        sequences = seq_by_brenda(ec, organism)
        batch_results[ec + '&' + organism] = sequences
    time.sleep(1)
    IdSeq.update(batch_results)

def combine_sequence() :
    with open('/Users/xw0201/Desktop/postdoc/CPI_structure_CALB/Large_Dataset/BRENDA/KcatKm/uniprotID_entry.json', 'r') as file1:
        uniprot_file1 = json.load(file1)

    # with open('/Users/xw0201/Desktop/postdoc/CPI_structure_CALB/Large_Dataset/BRENDA/KcatKm/nouniprotID_entry_all.json', 'r') as file2:  # By Uniprot API
    #     nouniprot_file2 = json.load(file2)

    with open('/Users/xw0201/Desktop/postdoc/CPI_structure_CALB/Large_Dataset/BRENDA/KcatKm/nouniprotID_entry_brenda.json', 'r') as file3:  # By BRENDA API
        nouniprot_file3 = json.load(file3)

    with codecs.open("/Users/xw0201/Desktop/postdoc/CPI_structure_CALB/Large_Dataset/BRENDA/KcatKm/Km_combination.tsv", "r", encoding='utf-8') as file4 :
        Kcat_lines = file4.readlines()[1:]

    # i = 0
    # for proteinKey, sequence in nouniprot_file2.items() :
    #     if sequence :
    #         if len(sequence) == 1 :  # 1178 BRENDA  1919 Uniprot
    #         # if sequence :  # 1784 BRENDA  3363 Uniprot
    #             i += 1   
    #             print(i)
    # print(len(nouniprot_file3))

    i = 0
    j = 0
    n = 0
    entries = list()
    for line in Kcat_lines :
        data = line.strip().split('\t')
        ECNumber, EnzymeType, Organism, Smiles = data[0], data[1], data[2], data[3]
        Substrate, UniprotID, Value, Unit = data[4], data[5], data[6], data[7]

        RetrievedSeq = ''
        entry = dict()
        if UniprotID :
            try :  # a few (maybe four) UniprotIDs have no ID as the key
                if ' ' not in UniprotID :
                    RetrievedSeq = [uniprot_file1[UniprotID]]
                    # print(RetrievedSeq)
                else :
                    # print(UniprotID)
                    RetrievedSeq1 = [uniprot_file1[UniprotID.split(' ')[0]]]
                    RetrievedSeq2 = [uniprot_file1[UniprotID.split(' ')[1]]]
                    if RetrievedSeq1 == RetrievedSeq2 :
                        RetrievedSeq = RetrievedSeq1

            except :
                continue



        try:  # local variable 'RetrievedSeq' referenced before assignment
            if len(RetrievedSeq) == 1 and EnzymeType == 'wildtype':
                sequence = RetrievedSeq
                i += 1

                entry = {
                    'ECNumber': ECNumber,
                    'Organism': Organism,
                    'Smiles': Smiles,
                    'Substrate': Substrate,
                    'UniprotID': UniprotID,
                    'Sequence': sequence[0],
                    'Type': 'wildtype',
                    'Value': Value,
                    'Unit': Unit,
                }

                entries.append(entry)

            if len(RetrievedSeq) == 1 and EnzymeType != 'wildtype':
                sequence = RetrievedSeq[0]

                mutantSites = EnzymeType.split('/')
                # print(mutantSites)

                mutant1_1 = [mutantSite[1:-1] for mutantSite in mutantSites]
                mutant1_2 = [mutantSite for mutantSite in mutantSites]
                mutant1 = [mutant1_1, mutant1_2]
                mutant2 = set(mutant1[0])
                if len(mutant1[0]) != len(mutant2) :
                    print(mutant1)
                    n += 1
                    print(str(n) + '---------------------------')  # some are mapped, some are not mapped. R234G/R234K (60, 43 mapped, 17 not mapped)

                mutatedSeq = sequence
                for mutantSite in mutantSites :
                    # print(mutantSite)
                    # print(mutatedSeq[int(mutantSite[1:-1])-1])
                    # print(mutantSite[0])
                    # print(mutantSite[-1])
                    if mutatedSeq[int(mutantSite[1:-1])-1] == mutantSite[0] :
                        # pass
                        mutatedSeq = list(mutatedSeq)
                        mutatedSeq[int(mutantSite[1:-1])-1] = mutantSite[-1]
                        mutatedSeq = ''.join(mutatedSeq)
                        # if not mutatedSeq :
                        #  print('-------------')
                    else :
                        # n += 1
                        # print(str(n) + '---------------------------')
                        mutatedSeq = ''

                if mutatedSeq :
                    # j += 1
                    # print(str(j) + '---------------------------')          
                    entry = {
                        'ECNumber': ECNumber,
                        'Organism': Organism,
                        'Smiles': Smiles,
                        'Substrate': Substrate,
                        'UniprotID': UniprotID,
                        'Sequence': mutatedSeq,
                        'Type': EnzymeType,
                        'Value': Value,
                        'Unit': Unit,
                    }

                    entries.append(entry)

        except:
            continue

    # mutatedSeq.replace([int(mutantSite[1:-1])-1], mutantSite[-1])
    print(i)

    print(len(entries))   # 17010  including 9529 wildtype and 7481 mutant

    # with open('../../Data/database/Kcat_combination_0918.json', 'w') as outfile :
    #     json.dump(entries, outfile, indent=4)
    with open('/Users/xw0201/Desktop/postdoc/CPI_structure_CALB/Large_Dataset/BRENDA/KcatKm/KM_results.json', 'w') as outfile :
        json.dump(entries, outfile, indent=4)

def check_substrate_seq() :
    with open('/Users/xw0201/Desktop/postdoc/CPI_structure_CALB/Large_Dataset/BRENDA/KcatKm/KM_results.json', 'r') as file :
        datasets = json.load(file)
    UniprotID = [data['UniprotID'].upper() for data in datasets]
    substrate = [data['Substrate'].lower() for data in datasets]
    sequence = [data['Sequence'] for data in datasets]
    organism = [data['Organism'].lower() for data in datasets]
    EC_number = [data['ECNumber'] for data in datasets]

    unique_UniprotID = len(set(UniprotID))
    unique_substrate = len(set(substrate))
    unique_sequence = len(set(sequence))
    unique_organism = len(set(organism))
    unique_EC_number = len(set(EC_number))

    print('The number of unique substrate:', unique_substrate)
    print('The number of unique sequence:', unique_sequence)
    print('The number of unique organism:', unique_organism)
    print('The number of unique EC Number:', unique_EC_number)
    print('The number of unique UniprotID:', unique_UniprotID)



    with open("/Users/xw0201/Desktop/postdoc/CPI_structure_CALB/Large_Dataset/BRENDA/KcatKm/unique_uniprot.txt", 'w') as file:
        for value in set(UniprotID):
            file.write(value + '\n')

if __name__ == "__main__" :
    """
    # nouniprotID_entry_uniprot()
    """

    # nouniprotID_entry_brenda()
    # uniprotID_entry()
    combine_sequence()
    # check_substrate_seq()


