#!/usr/bin/python
# coding: utf-8

import csv
import json
import codecs

with open('/Users/xw0201/Desktop/postdoc/CPI_structure_CALB/Large_Dataset/allec/KCAT/Sabio_smiles.json', 'r') as infile:
    sabio_name_smiles = json.load(infile)

with open('/Users/xw0201/Desktop/postdoc/CPI_structure_CALB/Large_Dataset/allec/KCAT/Brenda_smiles.json', 'r') as infile:
    brenda_name_smiles = json.load(infile)

with codecs.open("/Users/xw0201/Desktop/postdoc/CPI_structure_CALB/Large_Dataset/allec/KCAT/KCAT_sabio_clean_mutant.tsv", "r", encoding='utf-8') as file1 :
    sabio_lines = file1.readlines()[1:]

with codecs.open("/Users/xw0201/Desktop/postdoc/CPI_structure_CALB/Large_Dataset/allec/KCAT/KCAT_brenda_clean.tsv", "r", encoding='utf-8') as file2 :
    brenda_lines = file2.readlines()[1:]

Kcat_data = list()
Kcat_data_include_value = list()
Substrate_name = dict()
Substrate_smiles = dict()
entry_uniprot = dict()

for line in brenda_lines :
    # print(line)
    data = line.strip().split('\t')
    ECNumber = data[1]
    Substrate = data[2]
    EnzymeType = data[3]
    Organism =data[4]
    Value = data[5]
    Unit = data[6]

    smiles = brenda_name_smiles[Substrate]
    # print(smiles)
    if smiles is not None :
        # print(smiles)
        Substrate_name[Substrate.lower()] = Substrate
        Substrate_smiles[Substrate.lower()+'&smiles'] = smiles

        Kcat_data_include_value.append([ECNumber, Substrate.lower(), EnzymeType, Organism, Value, Unit])
        Kcat_data.append([ECNumber, Substrate.lower(), EnzymeType, Organism])

for line in sabio_lines :
    # print(line)
    data = line.strip().split('\t')
    ECNumber = data[1]
    Substrate = data[2]
    EnzymeType = data[3]
    Organism =data[5]
    UniprotID = data[6]
    Value = data[7]
    Unit = data[8]

    smiles = sabio_name_smiles[Substrate]

    if smiles is not None :

        Substrate_name[Substrate.lower()] = Substrate
        Substrate_smiles[Substrate.lower()+'&smiles'] = smiles
        entry_uniprot[ECNumber + EnzymeType + Organism] = UniprotID

        Kcat_data_include_value.append([ECNumber, Substrate.lower(), EnzymeType, Organism, Value, Unit])
        Kcat_data.append([ECNumber, Substrate.lower(), EnzymeType, Organism])

print(len(Kcat_data))

new_lines = list()
for line in Kcat_data:
    if line not in new_lines:
        new_lines.append(line)

print(len(new_lines))

i = 0
clean_Kcat = list()
for new_line in new_lines :
    # print(new_line)
    i += 1
    # print(i)
    value_unit = dict()
    Kcat_values = list()
    for line in Kcat_data_include_value :
        if line[:-2] == new_line :
            value = line[-2]
            value_unit[str(float(value))] = line[-1]
            # print(type(value))  # <class 'str'>
            Kcat_values.append(float(value))
    # print(value_unit)
    # print(Kcat_values)
    max_value = max(Kcat_values) # choose the maximum one for duplication Kcat value under the same entry as the data what we use
    unit = value_unit[str(max_value)]
    # print(max_value)
    # print(unit)
    Substrate = Substrate_name[new_line[1]]
    Smiles = Substrate_smiles[new_line[1]+'&smiles']
    try :
        UniprotID = entry_uniprot[new_line[0]+new_line[2]+new_line[3]]
    except :
        UniprotID = ''
    new_line[1] = Substrate
    new_line = new_line + [Smiles, UniprotID, str(max_value), unit]
    # print(new_line)
    # new_line.append(str(max_value))
    # new_line.append(unit)
    clean_Kcat.append(new_line)

# print(clean_Kcat)
print(len(clean_Kcat))  # 44166

with open("/Users/xw0201/Desktop/postdoc/CPI_structure_CALB/Large_Dataset/allec/KCAT/KCAT_combination_temp.tsv", "w") as outfile :
    records = ['ECNumber', 'Substrate', 'EnzymeType', 'Organism', 'Smiles', 'UniprotID', 'Value', 'Unit']
    outfile.write('\t'.join(records) + '\n')
    for line in clean_Kcat :
        outfile.write('\t'.join(line) + '\n')

# The above is to eliminate the dupliaction entries by Substrate name
print('-----------------------------------------------')

# This is to eliminate the duplication entries by Smiles
with codecs.open("/Users/xw0201/Desktop/postdoc/CPI_structure_CALB/Large_Dataset/allec/KCAT/KCAT_combination_temp.tsv", "r", encoding='utf-8') as infile :
    Kcat_lines = infile.readlines()[1:]

Kcat_data = list()
Kcat_data_include_value = list()
Substrate_name = dict()
entry_uniprot = dict()

for line in Kcat_lines :
    # print(line)
    data = line.strip().split('\t')
    ECNumber = data[0]
    Substrate = data[1]
    EnzymeType = data[2]
    Organism =data[3]
    Smiles = data[4]
    UniprotID = data[5]
    Value = data[6]
    Unit = data[7]

    Substrate_name[Smiles] = Substrate
    entry_uniprot[ECNumber + EnzymeType + Organism] = UniprotID

    Kcat_data_include_value.append([ECNumber, EnzymeType, Organism, Smiles, Value, Unit])
    Kcat_data.append([ECNumber, EnzymeType, Organism, Smiles])

print(len(Kcat_data))


new_lines = list()
for line in Kcat_data :
    if line not in new_lines :
        new_lines.append(line)

print(len(new_lines))

i = 0
clean_Kcat = list()
unique = list()
for new_line in new_lines :
    # print(new_line)
    i += 1
    # print(i)
    value_unit = dict()
    Kcat_values = list()
    for line in Kcat_data_include_value :
        if line[:-2] == new_line :
            value = line[-2]
            value_unit[str(float(value))] = line[-1]
            # print(type(value))  # <class 'str'>
            Kcat_values.append(float(value))
    # print(value_unit)
    # print(Kcat_values)
    max_value = max(Kcat_values) # choose the maximum one for duplication Kcat value under the same entry as the data what we use
    unit = value_unit[str(max_value)]
    # print(max_value)
    # print(unit)
    Substrate = Substrate_name[new_line[3]]
    try :
        UniprotID = entry_uniprot[new_line[0]+new_line[1]+new_line[2]]
    except :
        UniprotID = ''
    if ' ' in UniprotID:
        unique += UniprotID.split(' ')
    if (UniprotID!= '') and (UniprotID not in unique):
        unique.append(UniprotID)
    new_line = new_line + [Substrate, UniprotID, str(max_value), unit]
    # print(new_line)
    # new_line.append(str(max_value))
    # new_line.append(unit)
    clean_Kcat.append(new_line)

with open("/Users/xw0201/Desktop/postdoc/CPI_structure_CALB/Large_Dataset/allec/KCAT/unique_Uniprot.txt", "w") as outfile1:
    for item in unique:
        outfile1.write("%s\n" % item)
# print(clean_Kcat)
print(len(clean_Kcat))  # 41558, in which 13454 entries have UniprotID

with open("/Users/xw0201/Desktop/postdoc/CPI_structure_CALB/Large_Dataset/allec/KCAT/KCAT_combination.tsv", "w") as outfile :
    records = ['ECNumber', 'EnzymeType', 'Organism', 'Smiles', 'Substrate', 'UniprotID', 'Value', 'Unit']
    outfile.write('\t'.join(records) + '\n')
    for line in clean_Kcat :
        outfile.write('\t'.join(line) + '\n')