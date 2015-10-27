'''
Created on Feb 13, 2014

@author: Adam
'''

import Papers
import queries
import Syntax_Scorer
import Semantics_Scorer
import Organize
import operator   #returns a callable object that fetches attr from its operand - see below
import generate_test_set
import simulate_A1_file
import csv



def no_papers_with_queries(sorted_sentences_with_score2, q1_syns, q1, q2_syns, q2, a_f):
    print 'no papers with queries found'
    print_output_to_file("No papers with co-occurrence found", None, None, None, None, a_file)

def no_cooc_sent(sorted_sentences_with_score2, q1_syns, q1, q2_syns, q2, a_f):
    print_output_to_file("No sentences with co-occurrence found", None, None, None, None, a_file)

def print_output_to_file(sorted_sentences_with_score2, q1_syns, q1, q2_syns, q2, a1_file):

    for sent_obj in sorted_sentences_with_score2[:1]:    #Organizes output
        if sorted_sentences_with_score2 == "No Papers with both queries were found on PubMed":
            with open(r'text_files\generated_test_data.txt','a') as f:
                f.write('\n'+a1_file.protein1+'\t')
                f.write(a1_file.protein2+'\t')
                if a1_file.paper_ID:
                    f.write(a1_file.paper_ID+'\n')
                if not a1_file.paper_ID:
                    f.write('\n')
                f.write("No Papers with both queries were found on PubMed!"+'\n') 
                f.close()
                       
        elif sorted_sentences_with_score2 == "No sentences with co-occurrence found":
            with open(r'text_files\generated_test_data.txt','a') as f:
                f.write('\n'+a1_file.protein1+'\t')
                f.write(a1_file.protein2+'\t')
                if a1_file.paper_ID:
                    f.write(a1_file.paper_ID+'\n')
                if not a1_file.paper_ID:
                    f.write('\n')
                f.write('No sentences with co-occurrence found!'+'\n')
                f.close()
        else:        
            score = sent_obj.score
            method_scored = str(sent_obj.method_scored)
            sent= sent_obj.sentence
            PMID = sent_obj.paper_id
            sent_w_replaced_queries = Organize.insert_syns(sent,q1,q1_syns,q2,q2_syns)
            with open(r'text_files\generated_test_data.txt','a') as f:
                f.write('\n'+a1_file.protein1+'\t')
                f.write(a1_file.protein2+'\t')
                f.write(str(score)+'\t')
                f.write(str(method_scored) + '\t')
                if a1_file.paper_ID:
                    f.write(a1_file.paper_ID+'\n')
                if not a1_file.paper_ID:
                    f.write('\n')
                f.write(sent_w_replaced_queries+'\n')
                f.close()
#            print "Likely interaction sentence w/ score: "
#            print score, sent_w_replaced_queries.capitalize()

    
def index(a1_file, articles, max_sentences):
   
    global a_file
    a_file = a1_file

    q1= a1_file.protein1
    q2= a1_file.protein2 
    query = queries.main(q1,q2)    # Creates Queries
    q1_syns = query.q1_syns        # Retrieves Q1 and Q2 synonyms
    q2_syns = query.q2_syns
    print a1_file.protein1, ' synonyms = ', q1_syns
    print a1_file.protein2, ' synonyms = ', q2_syns
    
    ID_sentence_position_list = Papers.main(query, articles)
    if len(ID_sentence_position_list) > 0:
        print str(len(ID_sentence_position_list)) + " sentences with co-occurrence found"
    
    sentences_with_score1 = Syntax_Scorer.main(ID_sentence_position_list, query, max_sentences) 
    sentences_with_score2 = Semantics_Scorer.main(sentences_with_score1, query)
    sorted_sentences_with_score2 = sorted(sentences_with_score2, key=operator.attrgetter('score'), reverse=True)
    if sorted_sentences_with_score2:
        with open (r'txt_files_Testing\calibration unlimited sentences','a') as f:
            f.write(query.q1+'\t'+query.q2+'\n')
            for sent in sorted_sentences_with_score2:
                sent_w_replaced_queries = Organize.insert_syns(sent.sentence,q1,q1_syns,q2,q2_syns)
                if str(sent.sentence)[0] != '<': 
                    f.write(str(sent.score) +' '+ str(sent.method_scored)+'\t'+ sent_w_replaced_queries + '\n')
                    print str(sent.score) +' '+ sent_w_replaced_queries
            f.write('\n') 
            
    print_output_to_file(sorted_sentences_with_score2, q1_syns, q1, q2_syns, q2, a1_file) 
    print ""
    
if __name__=='__main__':
    DataSet_type = 'Manual'    #Manual, Random, 50examples, madhavi_split
    q1= 'CLIP-170'
    q2= 'LIS1'
    articles = 300                       # maximum number of articles to get from PubMed
    max_sentences = 10                  # maximum number of sentences to return
    
    if DataSet_type == 'Manual':
        simulate_A1_file.main(q1, q2, articles, max_sentences)
    if DataSet_type == 'Random':
        size_of_test_set = 1000
        generate_test_set.main(size_of_test_set, articles, max_sentences)
    if DataSet_type == '50examples':
        dir_entry = r'text_files\madhavi_example_protein_interactions.txt'
        list_of_protein_pairs = []
        with open(dir_entry, 'r') as my_file:
            reader = csv.reader(my_file, delimiter='\t')
            for row in reader:
                list_of_protein_pairs.append(row)
        for protein_pair in list_of_protein_pairs:
            q1 = protein_pair[0]
            q2 = protein_pair[1]    
            simulate_A1_file.main(q1, q2, articles, max_sentences)       
    if DataSet_type == 'madhavi_split':
        dir_entry = r'C:\Users\Adam\workspace\BIoNLP_1.13\BioNLP_1.13\text_files\madhavi_split.txt'
        list_of_protein_pairs = []
        with open(dir_entry, 'r') as my_file:
            reader = csv.reader(my_file, delimiter='\t')
            for row in reader:
                list_of_protein_pairs.append(row)
        for protein_pair in list_of_protein_pairs:
            q1 = protein_pair[0]
            q2 = protein_pair[1]    
            simulate_A1_file.main(q1, q2, articles, max_sentences)   
