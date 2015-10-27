import URL
from nltk.tokenize import RegexpTokenizer
import PPI_cite_main
import queries
import time
import sys

class Paper(object):
    
    def __init__(self):
        self.id = None
        self.parsed = None
        self.title = None
        self.authors = None
        self.mesh_terms = None
        self.abstract = None
        self.all_sentences = None
        # classified sentences is a list of sentence objects]
        self.word_tokenized = None
        # list of tuples (sentence, pos in abstract)
        self.coocurring_sentences = None
        # list of tuples (sentence, pos in abstract)
        self.non_interactive_sents = None

    def make_sentence_id_tuples(self, coocurring_sentences):
        """ makes list of tuples of sentences in format
            (paper_id, sentence, order in absract) """
        list_of_tuples = []
        for sentence in coocurring_sentences:
            current_tuple = (self.id, sentence[0], sentence[1])
            list_of_tuples.append(current_tuple)
        return list_of_tuples

    def split_abstract_into_sentences(self, query):  # Splits abstract into sentences and adds paper title to this list 
        sentence_list = []
        sentence_list.append(self.title)
        if self.abstract:
            abstract_sentence_split = self.abstract.split(". ")
            sentence_list.extend(abstract_sentence_split)
        sentence_list = [x.lower() for x in sentence_list]
        self.all_sentences = sentence_list

    def find_query_in_sentence(self, sentence, query_str, syns_list):
        if query_str in sentence:
            return True
        if query_str.replace('-','') in sentence:
            return True
        if query_str.replace(' ', '') in sentence:
            return True
        if query_str in sentence.replace('-',' '):
            return True
        elif syns_list:
            for syn in syns_list:
                if syn in sentence:
                    return True
                if syn in sentence.replace('-',' '):
                    return True
        elif sentence:
            tokenizer = RegexpTokenizer("\s+", gaps = True)
            tokenized_sentence = tokenizer.tokenize(sentence)  
                
            for word in tokenized_sentence:
                if word == query_str:
                    return True
                elif syns_list:
                    if any(word == syn for syn in syns_list):
                        return True
                   
        elif sentence:
            sentence2 = sentence.replace("."," ").lower().split()
               
            if any(query_str.lower() == word.lower() for word in sentence2):
                return True
            
            else:
                if syns_list:
                    for syn in syns_list:
                        if any(syn.lower() == word.lower() for word in sentence2):
                            return True
                        
        else: return False
        
        
        
            
#            elif any(query_str.lower() == val for val in sentence2):
##                    if any(syn.lower() == val.lower() for val in syns_list):                    
#                print "GOTTTT ITTTTTTTTTTT!!!!!!!!", query_str, sentence
#                return True
                                    
 #                       elif syn.lower == any(val.lower() for val in sentence2):
 #                           return True


    def find_sentences_with_both_queries(self, sentence_list, query):
        """takes list of sentences, tests if both queries are in each 
            sentence. If so, keeps, if not, adds to non-interactive list"""
            
        coocurrence_list = []
        non_interactive_list = []
        for i, sentence in enumerate(sentence_list):
            q1_or_syns_found = self.find_query_in_sentence(sentence, query.q1, query.q1_syns)
            q2_or_syns_found = self.find_query_in_sentence(sentence, query.q2, query.q2_syns)
            if q1_or_syns_found and q2_or_syns_found:
                sentence_and_position = (sentence, i)
                coocurrence_list.append(sentence_and_position)
            elif q1_or_syns_found or q2_or_syns_found:              
                sentence_and_position = (sentence, i)
                non_interactive_list.append(sentence_and_position)
        self.non_interactive_sents = non_interactive_list
        return coocurrence_list

    def word_tokenize(self):
        """ splits sentences into list of words by spaces """
        tokenizer = RegexpTokenizer("\s+", gaps=True)
        
        if self.coocurring_sentences:
            self.word_tokenized = []
            for sentence in self.coocurring_sentences:
                tokenized_words = tokenizer.tokenize(sentence[0])
                self.word_tokenized.append(tokenized_words)
        else:
            self.word_tokenized = None

def sent_with_cooccur(ID_paper_obj_dict, query):  # returns a list of all sentences that contain both queries
    
    ID_sentence_lists = []
    for key in iter(ID_paper_obj_dict):
        if not ID_paper_obj_dict[key].all_sentences:
            ID_paper_obj_dict[key].split_abstract_into_sentences(query)
            ID_paper_obj_dict[key].word_tokenize()
            coocurrence_list = ID_paper_obj_dict[key].find_sentences_with_both_queries(ID_paper_obj_dict[key].all_sentences, query)  
            sentence_list = ID_paper_obj_dict[key].make_sentence_id_tuples(coocurrence_list)      
            ID_sentence_lists.extend(sentence_list)
        else: 
            coocurrence_list = ID_paper_obj_dict[key].find_sentences_with_both_queries(ID_paper_obj_dict[key].all_sentences, query)
            sentence_list = ID_paper_obj_dict[key].make_sentence_id_tuples(coocurrence_list)
            ID_sentence_lists.extend(sentence_list)
             
    if not ID_sentence_lists:
        print "No sentences with co-occurance found"
        time.sleep(3)
        PPI_cite_main.no_cooc_sent("No sentences with co-occurance found", None, None, None, None, None)
        return ID_sentence_lists
    else:
        return ID_sentence_lists

def make_paper_objects(dict_of_info):
    """takes in dict of info, returns dictionary paper objects like
        ["paper_id#"]: paper object """
    ID_paper_obj_dict = {}
    if "fetched_id_list" in dict_of_info:
        fetched_id_list = dict_of_info["fetched_id_list"]
        title_list = dict_of_info["title_list"]
        abstract_list = dict_of_info["abstract_list"]
           
        for i, paper_id in enumerate(fetched_id_list):
            paper = Paper()
            paper.id = fetched_id_list[i]
            paper.title = title_list[i]
            paper.abstract = abstract_list[i]
            ID_paper_obj_dict[paper_id] = paper
    return ID_paper_obj_dict    

def main(query, articles):
    dict_of_info = URL.main(query, articles)    #Gets info from PubMed
    ID_paper_obj_dict = make_paper_objects(dict_of_info)
    ID_sentence_lists = sent_with_cooccur(ID_paper_obj_dict, query)
    return ID_sentence_lists
