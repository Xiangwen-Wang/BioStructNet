# This code is to summary all the unique pdb id for further download and clean.

file1 = '/Users/xw0201/Desktop/postdoc/CPI_structure_CALB/Large_Dataset/BRENDA/Kcat/unique_pdb.txt'
file2 = '/Users/xw0201/Desktop/postdoc/CPI_structure_CALB/Large_Dataset/BRENDA/KM/unique_pdb.txt'
file3 = '/Users/xw0201/Desktop/postdoc/CPI_structure_CALB/Large_Dataset/BRENDA/KcatKm/unique_pdb.txt'
outputfile = '/Users/xw0201/Desktop/postdoc/CPI_structure_CALB/Large_Dataset/BRENDA/sum_pdbid.txt'


unique_strings = set()

file_paths = [file1, file2, file3]
for file_path in file_paths:
    with open(file_path, 'r') as file:

        lines = file.read().split('\n')
        for line in lines:

            unique_strings.add(line)


with open(outputfile, 'w') as output_file:
    for unique_string in unique_strings:
        output_file.write(unique_string + '\n')


