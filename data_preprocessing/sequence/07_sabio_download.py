#!/usr/bin/python
# coding: utf-8



import requests
import codecs

# Extract EC number list from ExPASy, which is a repository of information relative to the nomenclature of enzymes.
def eclist():
    with open('/Users/xw0201/Desktop/postdoc/CPI_structure_CALB/Large_Dataset/allec/KCAT/enzyme.dat', 'r') as outfile :
        lines = outfile.readlines()

    mark = 0
    ec_list = list()
    for line in lines :
        if line.startswith('ID') :
            ec = line.strip().split('  ')[1]
            if (ec[1:8] == '7.1.2.2'):
                mark = 1
            if mark:
                ec_list.append(ec)
    # print(ec_list)
    # print(len(ec_list))
    return ec_list

def sabio_info(allEC):
    QUERY_URL = 'http://sabiork.h-its.org/sabioRestWebServices/kineticlawsExportTsv'

    # specify search fields and search terms

    # query_dict = {"Organism":'"lactococcus lactis subsp. lactis bv. diacetylactis"', "Product":'"Tyrosine"'}
    # query_dict = {"Organism":'"lactococcus lactis subsp. lactis bv. diacetylactis"',} #saccharomyces cerevisiae  escherichia coli
    # query_dict = {"ECNumber":'"1.1.1.1"',}
    i = 0
    for EC in allEC :
        i += 1
        print('This is %d ----------------------------' %i)
        print(EC)
        query_dict = {"ECNumber":'%s' %EC,}
        query_string = ' AND '.join(['%s:%s' % (k,v) for k,v in query_dict.items()])


        # specify output fields and send request

        query = {'fields[]':['EntryID', 'Substrate', 'EnzymeType', 'PubMedID', 'Organism', 'UniprotID','ECNumber','Parameter'], 'q':query_string}
        # the 'Smiles' keyword could get all the smiles included in substrate and product

        request = requests.post(QUERY_URL, params = query)
        # request.raise_for_status()


        # results
        results = request.text
        print(results)
        print('---------------------------------------------')

        if check_second_line_empty(results):
            pass
        else:
            with codecs.open('/Users/xw0201/Desktop/postdoc/CPI_structure_CALB/Large_Dataset/allec/KCAT/sabio_4/%s.txt' %EC, 'w', encoding="utf-8") as ECfile :
                ECfile.write(results)

def check_second_line_empty(parameters):
    lines = parameters.split('\n')
    if len(lines) < 2:
        return False  # 参数少于两行，返回 False
    second_line = lines[1].strip()
    return second_line == ''  # 检查第二行是否仅包含回车字符

if __name__ == '__main__' :
    allEC = eclist()
    sabio_info(allEC)


