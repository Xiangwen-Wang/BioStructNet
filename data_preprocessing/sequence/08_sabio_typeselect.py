#!/usr/bin/python
# coding: utf-8



import os
import csv
import codecs

# with open("./Kcat_sabio_4_new/%s" %('1.1.1.184.txt'), 'r', encoding="utf-8") as file :
#     lines = file.readlines()

# for line in lines[1:] :
#     data = line.strip().split('\t')
#     print(data)


outfile = open("/Users/xw0201/Desktop/postdoc/CPI_structure_CALB/Large_Dataset/allec/KCAT/KCAT_sabio_4_unisubstrate.tsv", "wt")
# with open("./Kcat_sabio.tsv", "wt") as outfile :
tsv_writer = csv.writer(outfile, delimiter="\t")
tsv_writer.writerow(["EntryID", "Type", "ECNumber", "Substrate", "EnzymeType", "PubMedID", 
    "Organism", "UniprotID", "Value", "Unit"])

filenames = os.listdir('/Users/xw0201/Desktop/postdoc/CPI_structure_CALB/Large_Dataset/allec/KCAT/sabio_4')

i = 0
j=0
for filename in filenames :
    # print(filename[1:-4])

    if filename != '.DS_Store' :
        with codecs.open("/Users/xw0201/Desktop/postdoc/CPI_structure_CALB/Large_Dataset/allec/KCAT/sabio_4/%s" % filename, 'r', encoding="utf-8") as file :
            lines = file.readlines()

        for line in lines[1:] :
            data = line.strip().split('\t')
            try :
                # if data[7] == 'kcat' and data[9]:
                if data[7] == 'kcat' and data[9]:
                    i += 1
                    entryID = data[0]
                    """ # 'KM; kcat/Km' :
                    for line in lines[1:] :
                        data2 = line.strip().split('\t')
                        if data2[0] == entryID and data2[7] == 'kcat/Km' :
                            j += 1
                            tsv_writer.writerow([j, data[7], data[6], data2[8], data[2], data[3], data[4], data[5], data[9], data[-1]])
                    """
                    if ';H2O' in data[1]:
                        data[1] = data[1].replace(';H2O', '')
                    elif 'H2O;' in data[1]:
                        data[1] = data[1].replace('H2O;', '')
                    elif ';Hydroxylamine' in data[1]:
                        data[1] = data[1].replace(';Hydroxylamine', '')
                    elif 'Hydroxylamine;' in data[1]:
                        data[1] = data[1].replace('Hydroxylamine;', '')
                    elif 'L-Homocysteine;' in data[1]:
                        data[1] = data[1].replace('Hydroxylamine;', '')
                    elif ';L-Homocysteine' in data[1]:
                        data[1] = data[1].replace(';L-Homocysteine', '')
                    elif ';Ethanol' in data[1]:
                        data[1] = data[1].replace(';Ethanol', '')
                    elif 'Ethanol;' in data[1]:
                        data[1] = data[1].replace('Ethanol;', '')
                    elif 'Pentaglycine;' in data[1]:
                        data[1] = data[1].replace('Pentaglycine;', '')
                    elif ';Pentaglycine' in data[1]:
                        data[1] = data[1].replace(';Pentaglycine', '')
                    elif 'beta-D-Glucose;' in data[1]:
                        data[1] = data[1].replace('beta-D-Glucose;', '')
                    elif ';' in data[1]:
                        print(data[1]+'\t'+data[6])
                    tsv_writer.writerow(
                        [j, data[7], data[6], data[1], data[2], data[3], data[4], data[5], data[9], data[-1]])

            except :
                continue

outfile.close()

