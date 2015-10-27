'''
Created on Mar 2, 2014

@author: Adam
'''
import re
import stat_parser

def make_list_of_syns(text_file):
    word_list = []
    f = open(text_file)
    for line in f:
        line = line.strip()
        word_list.append(line)
    f.close()
    return word_list

class Sentence(object):

    def __init__(self):
        self.paper_id = None
        self.sentence = None
        self.order_in_abstract = None
        self.blinded_sentence = None
        
        self.tree = None
        
        self.pos_list = []
        self.q1_indexes = []
        self.q2_indexes = []
        self.s_indexes = []
        self.vp_indexes = []

        self.different_clauses = False
        self.vp_between_queries = False
        self.stimulatory_words_present = None
        self.inhibitory_words_present = None
        self.negation_words_present = None
        
        self.stimulatory = False
        self.inhibitory = False
        self.neutral = False
        self.parallel = False

        self.overall_classification = None
        self.general_classification = None


    def make_tree(self, sentence):    
        parsed_sentence = stat_parser.Parser().parse(sentence)
        return parsed_sentence
    
    def blind_sentence(self, sentence, query):
        q1_first_char_index = sentence.find(query.blinded_q1)
        q2_first_char_index = sentence.find(query.blinded_q2)
        
        q1_length = len(query.blinded_q1)
        q2_length = len(query.blinded_q2)
        
        q1_last_char = q1_first_char_index + q1_length
        q2_last_char = q2_first_char_index + q2_length
        
        sentence = sentence[:q2_first_char_index] + 'cat' + sentence[q2_last_char:]
        sentence = sentence[:q1_first_char_index] + 'dog' + sentence[q1_last_char:]
        
        return sentence

    def find_syns(self, thing_to_search_in, syn_list):
        if syn_list:
            for word in syn_list:
                if word in thing_to_search_in:
                    return True
                for word in syn_list:
                    if word in thing_to_search_in:
                        return True
                return False 
        


    def traverse(self, t, query):
        """ given a parse tree, traverses each node and leaf
            and makes an ordered list of each entity of interest
            (query terms, 'S' (notes simple declarative clauses), and 'VP's (verb phrase)
            also creates lists of indexes of each type of thing in list 
            tags:
            http://web.mit.edu/6.863/www/PennTreebankTags.html"""
        s_find = re.compile("^S")
        vp_find = re.compile("^VP")
#        print "tree and branches: "
#        result = t
#        result.draw()
#        print "t"
        try:
            t.label()
        
        except AttributeError:
            if query.blinded_q1 in t or self.find_syns(t, query.q1_syns):
                self.pos_list.append(t)
                pos_length = len(self.pos_list)
                self.q1_indexes.append(pos_length-1)
            elif query.blinded_q2 in t or self.find_syns(t, query.q2_syns):
                self.pos_list.append(t)
                pos_length = len(self.pos_list)
                self.q2_indexes.append(pos_length-1)

        else:
            # t.node is defined sp tests for regex match objects --- was replaced with .label()
            
            if s_find.match(t.label()):
                self.pos_list.append(t.label())
                pos_length = len(self.pos_list)
                self.s_indexes.append(pos_length-1)
            elif vp_find.match(t.label()):
                self.pos_list.append(t.label())
                pos_length = len(self.pos_list)
                self.vp_indexes.append(pos_length-1)
            for child in t:
                self.traverse(child, query) 


    def is_pos_between(self, q_index_list, pos_indexes):
        """Given two numbers in a list, tests if any indexes of
            pos_indexes are between the first two numbers """ 
        is_pos_between_queries = any(pos_idx in range(q_index_list[0], q_index_list[1]) for pos_idx in pos_indexes)
        return is_pos_between_queries
    
    def find_the_things(self,q1_indexes, q2_indexes, vp_indexes, s_indexes): #searched indexes and finds whether N or V there
        for idxq1 in q1_indexes:
            for idxq2 in q2_indexes:
                index_list = [idxq1, idxq2]
                index_list.sort()
                if self.is_pos_between(index_list, s_indexes):
                    self.different_clauses = True
                    self.score += 10
                    method_scored = self.method_scored
                    method_scored.append(27)
                    self.method_scored= method_scored
                    break            
            for idxq2 in q2_indexes:
                index_list = [idxq1, idxq2]
                index_list.sort()
                if self.is_pos_between(index_list, vp_indexes):
                    self.vp_between_queries = True
                    self.score += 20
                    method_scored = self.method_scored 
                    method_scored.append(28)
                    self.method_scored= method_scored
                    break

#        print self.pos_list
#        print self.different_clauses, "is s between"
#        print self.vp_between_queries, "is vp between"

    def make_list_of_syns(self,text_file):
        word_list = []
        f = open(text_file)
        for line in f:
            line = line.strip()
            word_list.append(line)
        return word_list

    #===========================================================================
    # def test_for_presence_of_words(self, sentence):
    #     stimulatory_words = make_list_of_syns("text_files\stim_words.txt")
    #     inhibitory_words = make_list_of_syns("text_files\inhib_words.txt")
    #     negation_words = [" not ", " no ", " none ", " did not ", " does not "]
    #     
    #     
    #     self.stimulatory_words_present = any(word in sentence for word in stimulatory_words)
    #     self.inhibitory_words_present = any(word in sentence for word in inhibitory_words)
    #     self.negation_words_present = any(word in sentence for word in negation_words)
    #===========================================================================


        
def make_sentence_objects(sentences_with_scores, query):
    
    sentence_list = []
    for i, sentence in enumerate(sentences_with_scores):

        sentence_obj = Sentence()
        sentence_obj.method_scored = sentence[0][1]
        sentence_obj.score = sentence[0][0]
        sentence_obj.paper_id = sentence[1]
        sentence_obj.sentence = sentence[2]
#        sentence_obj.blinded_sentence= sentence_obj.blind_sentence(sentence_obj.sentence, query)
        sentence_obj.order_in_abstract = sentence[3]
        sentence_obj.tree = sentence_obj.make_tree(sentence_obj.sentence)
        sentence_obj.traverse(sentence_obj.tree, query)
        sentence_obj.find_the_things(sentence_obj.q1_indexes, sentence_obj.q2_indexes, 
                                     sentence_obj.vp_indexes, sentence_obj.s_indexes)
#        sentence_obj.test_for_presence_of_words(sentence_obj.sentence)
        sentence_list.append(sentence_obj)   

    return sentence_list

def main(sentences_with_scores, query):

    sentence_list = make_sentence_objects(sentences_with_scores, query)
    
#    print 'sentence1 list', sentences_with_scores
#    print 'sentence2', sentence_list[0].score, sentence_list[0].method_scored
    

    return sentence_list

