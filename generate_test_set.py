'''
Created on Apr 12, 2014

@author: Adam
'''

import csv
import os
import PPI_cite_main

class A1_File(object):
    
    def __init__(self):
        self.dict = None
        
        self.paper_ID = None
        self.proteins = None
        
        self.protein1 = None
        self.protein2 = None
        
    def get_paper_ID(self, data):
        for key, value in data.iteritems() :
            return key.rstrip(".a1")
        
        

def make_a1_file_object(my_file, dir_entry, first_two_proteins):
    
    data = {}
    data[dir_entry] = my_file.read()
                   
    a1_file = A1_File()
    a1_file.dict = data
    a1_file.paper_ID = a1_file.get_paper_ID(data)
    a1_file.proteins = first_two_proteins
    
    a1_file.protein1 = first_two_proteins[0]
    a1_file.protein2 = first_two_proteins[1]
    
    return a1_file
  
            
def main(size_of_test_set, articles, max_sentences):
    
    path = r'C:\Users\Adam\workspace\Wiki Pi NLP\Test_Set_Files_BIONLP09\dot_a1_files'
    count = 0
    for dir_entry in os.listdir(path):
        count += 1
        if count > size_of_test_set:
            break
        dir_entry_path = os.path.join(path, dir_entry)
        if os.path.isfile(dir_entry_path):
            with open(dir_entry_path, 'r') as my_file:
                reader=csv.reader(my_file,delimiter='\t')
                rows = []
                for row in reader:
                    rows.append(row)
                first_two_proteins = []
                for lst in rows[:2]:
                    first_two_proteins.append(lst[2])
                if len(first_two_proteins) != 2:
                    continue
                first_two_proteins = [x.lower() for x in first_two_proteins]
                #first_two_proteins = [x.replace('-',' ') for x in first_two_proteins]
                if first_two_proteins[0] == first_two_proteins[1]:
                    continue
                a1_file = make_a1_file_object(my_file, dir_entry,first_two_proteins)

            print a1_file.proteins, count
            PPI_cite_main.index(a1_file, articles, max_sentences) 

        
        
if __name__=='__main__':
    x = 800
    main(x)
    
    