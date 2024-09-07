#!/usr/bin/python
# coding: utf-8


import os
import re
import csv
import codecs



with codecs.open("/Users/xw0201/Desktop/postdoc/CPI_structure_CALB/Large_Dataset/allec/KCAT/KCAT_sabio_clean.tsv", "r", encoding='utf-8') as file :
    lines = file.readlines()[1:]

clean_mutant = list()
for line in lines :
    # print(line)
    data = line.strip().split('\t')
    Type = data[0]
    ECNumber = data[1]
    Substrate = data[2]
    EnzymeType = data[3]
    PubMedID = data[4]
    Organism =data[5]
    UniprotID = data[6]
    Value = data[7]
    Unit = data[8]

    if 'wildtype' in EnzymeType :
        enzymeType = 'wildtype'
    else :
    # if 'mutant' in EnzymeType or 'mutated' in EnzymeType:
        print(EnzymeType)
        mutant = re.findall('[A-Z]\d+[A-Z]', EnzymeType)  # re is of great use
        enzymeType = '/'.join(mutant)

    print(enzymeType)
    if enzymeType:
        clean_mutant.append([Type, ECNumber, Substrate, enzymeType, PubMedID, Organism, UniprotID, Value, Unit])


# print(enzymeType_entries)
print(len(clean_mutant))  # 17384


with open("/Users/xw0201/Desktop/postdoc/CPI_structure_CALB/Large_Dataset/allec/KCAT/KCAT_sabio_clean_mutant.tsv", "w") as outfile :
    records = ['Type', 'ECNumber', 'Substrate', 'EnzymeType', 'PubMedID', 'Organism', 'UniprotID', 'Value', 'Unit']
    outfile.write('\t'.join(records) + '\n')
    for line in clean_mutant :
        outfile.write('\t'.join(line) + '\n')

