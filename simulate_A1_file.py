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
        
        

def make_a1_file_object(first_two_proteins):
    a1_file = A1_File()
    a1_file.proteins = first_two_proteins
    
    a1_file.protein1 = first_two_proteins[0]
    a1_file.protein2 = first_two_proteins[1]
    
    a1_file.paper_ID = None
    
    return a1_file
  
            
def main(q1, q2, articles, max_sentences):
    q1= q1.lower()
    q2 = q2.lower()
    first_two_proteins = [q1, q2]
    a1_file = make_a1_file_object(first_two_proteins)
    print a1_file.proteins
    PPI_cite_main.index(a1_file, articles, max_sentences) 

        
        
if __name__=='__main__':
    main(x)
    
    