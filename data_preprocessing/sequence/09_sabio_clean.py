#!/usr/bin/python
# coding: utf-8



import csv
import codecs

with codecs.open("/Users/xw0201/Desktop/postdoc/CPI_structure_CALB/Large_Dataset/allec/KCAT/KCAT_sabio_4_unisubstrate.tsv", "r", encoding='utf-8') as file :
    lines = file.readlines()[1:]


Kcat_data = list()
Kcat_data_include_value = list()
for line in lines :
    # print(line)
    data = line.strip().split('\t')
    Type = data[1]
    ECNumber = data[2]
    Substrate = data[3]
    EnzymeType = data[4]
    PubMedID = data[5]
    Organism =data[6]
    UniprotID = data[7]
    Value = data[8]
    Unit = data[9]
    Kcat_data_include_value.append([Type, ECNumber, Substrate, EnzymeType, PubMedID, Organism, UniprotID, Value, Unit])
    Kcat_data.append([Type, ECNumber, Substrate, EnzymeType, PubMedID, Organism, UniprotID])

print(len(Kcat_data))


new_lines = list()
for line in Kcat_data :
    if line not in new_lines :
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

    if unit in ['mol*s^(-1)*mol^(-1)', 's^(-', '-'] :
        unit = 's^(-1)'
    new_line.append(str(max_value))
    new_line.append(unit)

    clean_Kcat.append(new_line)

# print(clean_Kcat)
print(len(clean_Kcat))  # 18243 after unifing the Kcat value unit to 's^(-1)', in which 16825 has a specific Unipro ID


with open("/Users/xw0201/Desktop/postdoc/CPI_structure_CALB/Large_Dataset/allec/KCAT/KCAT_sabio_clean.tsv", "w") as outfile :
    records = ['Type', 'ECNumber', 'Substrate', 'EnzymeType', 'PubMedID', 'Organism', 'UniprotID', 'Value', 'Unit']
    outfile.write('\t'.join(records) + '\n')
    for line in clean_Kcat :
        outfile.write('\t'.join(line) + '\n')
