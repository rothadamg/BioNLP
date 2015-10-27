'''
Created on Aug 6, 2014

@author: Adam
'''
import heuristic_scores
import sk_learn_testing
import xml.etree.ElementTree as ET
import queries
import itertools
import Syntax_Scorer
import Semantics_Scorer
import operator
from collections import OrderedDict
import numpy as np
from sklearn import metrics
import pylab as pl
from pyroc import *
from nltk.tokenize import RegexpTokenizer
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn.svm import SVC
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn import grid_search
from collections import Counter
from sklearn.grid_search import GridSearchCV


def xml_to_dict(root):

    document_dic = {}
    ids = []
    protein_entity_dicts = []
    abstracts = []
    
    for document in root.findall('.//document'):
        id = document.find('.//id').text
        ids.append(id)
    
    for passage in iter(root.findall('.//passage')):
        txt = passage.find('text')
        abstracts.append(txt)
        annotations = passage.findall('annotation[@id]')
        id_text_dict = {}
        annotation_location_dict = {}
        for annotation in annotations:
            id_text_dict[annotation.get('id')]= annotation.find('text').text
            offset_length_tuples_list = []
            for node in annotation:
                if node.tag == 'location':
                    offset_length_tuples_list.append((node.attrib['offset'], node.attrib['length']))
            annotation_location_dict[annotation.get('id')] = offset_length_tuples_list
        id_entity_dict = {}
        relations = passage.findall('relation[@id]')
        for relation in relations:
            relation_id = relation.get('id')
            refids = []
            for node in relation:
                if node.tag == 'node':
                    refids.append(node.attrib['refid'])
                id_entity_dict[relation_id]= refids
        t = (id_text_dict,id_entity_dict, annotation_location_dict)
        protein_entity_dicts.append(t)
    
    abstract_entity_tuples = zip(abstracts, protein_entity_dicts)
    xml_dict = dict(zip(ids, abstract_entity_tuples))
    return xml_dict

#===============================================================================
# def get_abstract_queries(xml_dict):
#     abstract_queries_full_list = []
#     for key in xml_dict:
#         abstract_queries_ind_tuples = ()
#         text_rel_tuple = xml_dict[key]
#         abstract = text_rel_tuple[0]
#         query_list = []
#         for key in text_rel_tuple[1][0]:
#             query_list.append(key)
#         abstract_queries_ind_tuples = (abstract, query_list)
#         abstract_queries_full_list.append(abstract_queries_ind_tuples)
#     return abstract_queries_full_list
#===============================================================================

def get_doc_ids(xml_dict):
    doc_ids = []
    for key in xml_dict:
        doc_ids.append(key)
    return doc_ids
 
#===============================================================================
# def get_queries(doc_ids, abstract_queries):
#       
#     query_tuples_per_abstract = []
#     abstract_list = [] 
#     
#     query_tuples_per_abstract = []
#     for abs_query in abstract_queries:
#         query_tuples_ind_abstract = []
#         doc_queries_list = abs_query[1]
#         abstract_list.append(abs_query[0])
#         abstract_query_combinations = itertools.combinations(doc_queries_list, 2)
#         
#         for combination in abstract_query_combinations:
#             query_tuples_ind_abstract.append(combination)
#         query_tuples_per_abstract.append(query_tuples_ind_abstract)
#             
#     docID_query_combinations = zip(doc_ids, query_tuples_per_abstract)
#     return docID_query_combinations, abstract_list
#===============================================================================

def get_id_sent_lists(doc_ID, abstract, query):  
    abstract_text = abstract.text

class hprd50_Paper(object):
    
    def __init__(self):
        
        self.relations = None
        self.entity_dict = None
        self.id = None
        self.abstract = None
        self.xml_dict_value = None
        self.possible_relations_Tnumber = None
        self.possible_relations_names = None
        self.known_relations_Tnumber = None
        self.known_relations_names= None
        self.abstract_replaced_names = None
        
        self.all_sentences = None
        
    def get_known_relations_Tnumber(self):
        relation_dict = self.xml_dict_value[1][1]
        known_relations_Tnumber = []
        for key in relation_dict:
            known_relations_Tnumber.append(tuple(relation_dict[key]))
        return known_relations_Tnumber
    
    def get_known_relations_names(self):
        known_relations_names = []
        for relation in self.known_relations_Tnumber:
            entity1 = relation[0]
            entity2 = relation[1]
            if entity1 in self.entity_dict:
                entity1 = self.entity_dict[entity1]
            if entity2 in self.entity_dict:
                entity2 = self.entity_dict[entity2]
            relation_tuple = (entity1, entity2)
            known_relations_names.append(relation_tuple)
        return known_relations_names
    
    def get_entity_dict(self):
        return self.xml_dict_value[1][0]
    
    def get_possible_relations(self):
        possible_relations_Tnumber = []
        possible_relations_names= []
        for key in self.xml_dict_value[1][0]:
            if key not in possible_relations_Tnumber:
                possible_relations_Tnumber.append(key)
            if self.xml_dict_value[1][0][key] not in possible_relations_names:
                possible_relations_names.append(self.xml_dict_value[1][0][key])
        possible_relations_Tnumber_before = list(itertools.combinations(possible_relations_Tnumber, 2))
        self.possible_relations_Tnumber = [x[::-1] for x in possible_relations_Tnumber_before]
#        self.possible_relations_Tnumber = list(itertools.combinations(possible_relations_Tnumber, 2))
        self.possible_relations_names = list(itertools.combinations(possible_relations_names, 2))


    def replace_names(self):
        abstract = self.abstract
        removal_dict = {}
        for key in self.xml_dict_value[1][2]:
            first_char = int(self.xml_dict_value[1][2][key][0][0])
            offset = int(self.xml_dict_value[1][2][key][0][1])
            last_char = first_char + offset
            removal_dict[key] = (first_char, last_char)
        sorted_removal_dict = OrderedDict(sorted(removal_dict.items(), key=lambda t: t[1][0], reverse=True))
        for key in sorted_removal_dict:
            first_char = sorted_removal_dict[key][0]
            last_char = sorted_removal_dict[key][1]
            abstract = abstract[:first_char] + key + abstract[last_char:]
        self.abstract_replaced_names = abstract
        return abstract



def make_hprd50_papers(xml_dict):
    
    hprd50_paper_dict = {}
    
    for i, key in enumerate(xml_dict):
        paper = hprd50_Paper()
        paper.id = key
        paper.xml_dict_value = xml_dict[key] 
        paper.known_relations_Tnumber = paper.get_known_relations_Tnumber()
        paper.entity_dict = paper.xml_dict_value[1][0]
        paper.known_relations_names = paper.get_known_relations_names()
        paper.abstract = paper.xml_dict_value[0].text
        paper.get_possible_relations()
        paper.abstract_replaced_names = paper.replace_names()
        paper.abstract_sentences = paper.abstract_replaced_names.split('\n')
        hprd50_paper_dict[paper.id] = paper

    return hprd50_paper_dict


def find_cooccurrence_sents(paper_obj, query):
    tokenizer = RegexpTokenizer("\W|_", gaps = True)
    ID_sentence_list = []
    for i, sentence in enumerate(paper_obj.abstract_sentences):
        sentence = tokenizer.tokenize(sentence)
        if (query.q1 in sentence) and (query.q2 in sentence):
            ID_sent_position_tuple = (paper_obj.id, paper_obj.abstract_sentences[i], i)
            ID_sentence_list.append(ID_sent_position_tuple)
    return ID_sentence_list

def get_all_predicted_relations_dict(hprd50_paper_dict, max_sentences, all_known_relations_dict):
    all_predicted_relations_dict = {}  
    count_key = 0
    for key in hprd50_paper_dict: 
        count_key += 1
        print count_key

        found_relation_score_tuples_list = [] 
        for relation in hprd50_paper_dict[key].possible_relations_Tnumber:
            q1 = relation[0]
            q2 = relation[1]
            found_relation_score_tuple = ((q1, q2), 0)
            if q1 == q2:
                raise error("q1 == q2!")
            query = queries.main(q1,q2)
            ID_sentence_list = []
            ID_sentence_list = find_cooccurrence_sents(hprd50_paper_dict[key], query)
            if not ID_sentence_list:
                found_relation_score_tuple = ((q1, q2), 0)
                found_relation_score_tuples_list.append(found_relation_score_tuple)
            if ID_sentence_list:
                sentences_with_score1 = Syntax_Scorer.main(ID_sentence_list, query, max_sentences)
                sentences_with_score2 = Semantics_Scorer.main(sentences_with_score1, query)
                sorted_sentences_with_score2 = list(sorted(sentences_with_score2, key=operator.attrgetter('score'), reverse=True))
                #------------------------------ if sorted_sentences_with_score2:
                    #----------------- if len(sorted_sentences_with_score2) > 1:
                        #-------------------------------------------------- pass
                    #----------------------------------------------------- else:
                        
                for sentence_object in sorted_sentences_with_score2:                  

#-------------------------TESTING----------------------------------------------------------------------------------------------                  

                    found_relation_score_tuple = ((q1, q2), sentence_object.score, sentence_object.order_in_abstract)
                    print ID_sentence_list[0][0], found_relation_score_tuple, sentence_object.sentence
                    
                    yes_no_list = []
                    for known_relation in all_known_relations_dict[key]:
                        known1 = known_relation[0]
                        known2 = known_relation[1]
                        if (known1 in found_relation_score_tuple[0]) and (known2 in found_relation_score_tuple[0]):
                            yes_no_list.append(1)
                        else:
                            yes_no_list.append(0)
                    
                    if 1 in yes_no_list:
#                        print 'GOLD = INTERACTS'
                        interaction_present = 1
                    else: 
#                        print 'GOLD = NO INTERACTION'
                        interaction_present = 0
                    with open(r'text_files\calibration_9214','a') as f:
                        if 1 in yes_no_list:
                            f.write('\n' + '1' + '\t' + str(sentence_object.score) +'\t' + str(sentence_object.method_scored) +'\t'+query.q1+'\t'+query.q2 + '\t' + sentence_object.sentence)
                        else:
                            f.write('\n' + '0' + '\t' + str(sentence_object.score) +'\t' + str(sentence_object.method_scored) +'\t'+query.q1+'\t'+query.q2 + '\t' + sentence_object.sentence)
                        f.close()
                    
#--------------------------------------------------------------------------------------------------------------------------------                 
                    
                    found_relation_score_tuple_final = ((q1, q2), sentence_object.score, sentence_object.order_in_abstract, sentence_object.method_scored, interaction_present)
                    found_relation_score_tuples_list.append(found_relation_score_tuple_final)
        all_predicted_relations_dict[key]=found_relation_score_tuples_list
    return all_predicted_relations_dict                      
                    
                                                
            
def get_all_possible_relations_dict(hprd50_paper_dict):
    all_possible_relations_dict = {}
    for key in hprd50_paper_dict:
        possible_relations_tuples = []
        for relation_tuple in hprd50_paper_dict[key].possible_relations_Tnumber:
            if relation_tuple[0] != relation_tuple[1]:
                possible_relations_tuples.append(relation_tuple)
        all_possible_relations_dict[key]=possible_relations_tuples
    return all_possible_relations_dict

def get_all_known_relations_dict(hprd50_paper_dict):
    all_known_relations_dict = {}
    for key in hprd50_paper_dict:
        known_relations_ind_paper = hprd50_paper_dict[key].known_relations_Tnumber
        all_known_relations_dict[key] = known_relations_ind_paper
    return all_known_relations_dict

def convert_data_for_testing(all_predicted_relations_dict, all_possible_relations_dict, all_known_relations_dict):

    gold_standard_dict = {}
    for key in sorted(all_known_relations_dict):
        if key not in all_possible_relations_dict:  #make sure both dictionaries have same keys
            raise IndexError
        else:
            gold_standard = []
            for possible_relation in all_possible_relations_dict[key]:
                known_relations_found = []
                for known_relation in all_known_relations_dict[key]:
                    if (known_relation[0] in possible_relation) and (known_relation[1] in possible_relation):
                        known_relations_found.append(1) 
                if len(known_relations_found) > 1:
                    raise error('multiple known relations found')
                if len(known_relations_found) == 1:
                    if isinstance(known_relations_found[0], int):
                        gold_standard.append(1)                                     
                else:
                    gold_standard.append(0)
            gold_standard_dict[key] = gold_standard

    test_predictions_dict = {}
    method_scored_dict = {}
    for key in sorted(all_predicted_relations_dict):
        if key not in all_possible_relations_dict:
            raise IndexError
        else:
            predict_list = []
            method_list = []
            for possible_relation in all_possible_relations_dict[key]:
                predicted_relations_found = []
                methods_found = []
                for predicted_relation in all_predicted_relations_dict[key]:
                    if (predicted_relation[0][0] in possible_relation) and (predicted_relation[0][1] in possible_relation):
#                         try:
#                             predicted_relations_found.append((predicted_relation[1],predicted_relation[3]))
#                         except Exception:
#                             predicted_relations_found.append((predicted_relation[1],[]))
                        predicted_relations_found.append(predicted_relation[1])
                        try:
                            methods_found.append(predicted_relation[3]) 
                        except Exception:
                            methods_found.append([])
                if len(predicted_relations_found) > 1:
                    raise error('multiple predicted relations found')
                elif predicted_relations_found:
                    predict_list.append(predicted_relations_found[0])
                    method_list.append(methods_found)
                else:
                    predict_list.append(0)
                    method_list.append([])
            test_predictions_dict[key] = predict_list 
            method_scored_dict[key] = method_list
 
    gold_standard = []
    test_predictions = []  
    method_scored_list = [] 
    
    for key in gold_standard_dict:
        if len(gold_standard_dict[key]) != len(test_predictions_dict[key]):
            raise Error('Length of gold standard different from length test predictions')
        if len(gold_standard_dict[key]) == len(test_predictions_dict[key]):
            gold_standard += gold_standard_dict[key]
            test_predictions += test_predictions_dict[key]
            method_scored_list += method_scored_dict[key]
            
    return gold_standard, test_predictions, method_scored_list

def compute_scores(gold_standard, test_predictions):
    
    
    expected = np.array(gold_standard)
    predicted = np.array(test_predictions)
    
    precision, recall, thresholds = metrics.precision_recall_curve(gold_standard, test_predictions)
    
    area = metrics.auc(recall, precision)  
    import math
   
    f1_scores = []
    for a in zip(precision,recall):
        p = a[0]
        r = a[1]
        f = 2 * (p * r) / (p + r)
        x = float(f)
        f1_scores.append(f)
    
#     print 'precision: ', precision
#     print 'recall: ', recall
#     print 'thresholds: ', thresholds
#     print 'f scores', f1_scores
#     print("Area Under Curve: %0.2f" % area)

    print '----------------------------------------------------'    
    '''pylab precision recall plot'''
    print 'pylab precision recall plot'
    pl.clf()
    pl.plot(recall, precision, label='Precision-Recall curve')
    pl.xlabel('Recall')
    pl.ylabel('Precision')
    pl.ylim([0.0, 1.05])
    pl.xlim([0.0, 1.0])
    pl.title('Precision-Recall example: AUC=%0.2f' % area)
    pl.legend(loc="lower left")
    pl.show()

    

def test_predictions_cooccur(test_predictions):
    
    co_occur_binary = []
    for a in test_predictions:
        if a >= 1:
            co_occur_binary.append(1)


def get_positive_class_vectors(gold_standard, test_predictions, method_scored_list):  
    
    print 'test predictions',len(test_predictions)
    print 'gold standard',len(gold_standard)
    print ''
    
    gold_standard_pos = []
    test_predictions_pos = []
    method_scored_pos = []
    test_predictions_neg = []
    gold_standard_neg = []
    method_scored_neg = []
               
            
    for num, score in enumerate(test_predictions):
        if (score != 0) or (gold_standard[num] == 1):
            test_predictions_pos.append(score)
            gold_standard_pos.append(gold_standard[num])
            method_scored_pos.append(method_scored_list[num])
        else:
            test_predictions_neg.append(score)
            gold_standard_neg.append(gold_standard[num])
            method_scored_neg.append(method_scored_list[num])


             
    return gold_standard_pos, test_predictions_pos, gold_standard_neg, test_predictions_neg
    
def SVM_parameter_search(X_train, y, clf):  #Grid search for best C param      
    
    Crange = np.logspace(-2, 2, 40)
    grid = GridSearchCV(clf, param_grid={'C': Crange},
                    scoring='precision', cv=5)
    parameters = {'kernel':('linear', 'rbf'), 'C': Crange}
    grid = GridSearchCV(clf, parameters)
    grid.fit(X_train, y)
    print "best parameter choice:", grid.best_params_
    scores = [g[1] for g in grid.grid_scores_]
    plt.semilogx(Crange, scores);
    plt.show()    

 
def find_feature_frequency(X_SVM_vectors, gold_standard):

    pos_feature_dict = {}   # total number of times each feature occurs in positive sentences
    neg_feature_dict = {}   # total number of times each feature occurs in negative sentences
    count_pos_sentences = 0
    count_neg_sentences = 0
#     pos_likelihood_dict = {}    # total number of positive sentences each feature represents
#     neg_likelihood_dict = {}    # total number of times each feature represents a negative sentence
    for num, vector in enumerate(X_SVM_vectors):
        if gold_standard[num] == 1:
            count_pos_sentences += 1
            for feature in vector:
                if feature > 0:
                    if feature in pos_feature_dict:
                        pos_feature_dict[feature] += 1
                    else:
                        pos_feature_dict[feature] = 1
        else: 
            if gold_standard[num] == 0:
                count_neg_sentences += 1
                for feature in vector:
                    if feature > 0:
                        if feature in neg_feature_dict:
                            neg_feature_dict[feature] += 1
                        else:
                            neg_feature_dict[feature] = 1
#         for feature in vector:
#             if gold_standard[num] == 1:
#                 if feature in pos_likelihood_dict:
#                     pos_likelihood_dict[feature] += 1
#                 else:
#                     pos_likelihood_dict[feature] = 1
#             if gold_standard[num] == 0:
#                 if feature in neg_likelihood_dict:
#                     neg_likelihood_dict[feature] += 1
#                 else:
#                     neg_likelihood_dict[feature] = 1

       
    score_dict = {1:20,2:20,3:0,4:0,5:10,6:20,7:10,8:20,9:10,10:20,11:1,12:3,13:-3,14:-6,15:5,16:0,17:15,18:0,19:0,20:15,21:15,22:15,23:15,24:5,25:25,26:25,27:25,28:35,29:10,30:20}

    combined_feature_dict = {}     #number of times each rule occurs total including pos and neg sentences
    for k, v in pos_feature_dict.items():
        combined_feature_dict[k] = float(v + neg_feature_dict.get(k, 0))
    
    pos_rule_frequency = {} # number of times rule occurs in positive sentences / number of times rule occurs overall
    for k, v in combined_feature_dict.items():
       pos_rule_frequency[k] = pos_feature_dict.get(k,0) / v

    neg_rule_frequency = {} # number of times rule occurs in negative sentences / number of times rule occurs overall
    for k, v in combined_feature_dict.items():
       neg_rule_frequency[k] = neg_feature_dict.get(k,0) / v
  
#    neg_rules = [(k,v) for k,v in neg_rule_frequency.items() if v>=.5]
#    for a in neg_rules:
#        print a
#    pos_rules = [(k,v) for k,v in pos_rule_frequency.items() if v>.5]
#    for a in pos_rules:
#        print a
    
    return pos_rule_frequency, neg_rule_frequency
  

def exitt():
    import sys
    sys.exit()  

#===============================================================================
# def convert_to_scores(X_feature_vectors):
#     print X_feature_vectors[:1]
#     score_dict = {1:20,2:20,3:0,4:0,5:10,6:20,7:10,8:20,9:10,10:20,11:1,12:3,13:-3,14:-6,15:5,16:0,17:15,18:0,19:0,20:15,21:15,22:15,23:15,24:5,25:25,26:25,27:25,28:35,29:10,30:20}
#     X_feature_vectors_scores = []
#     for vector in X_feature_vectors:
#         for num, i in enumerate(vector):
#             if i == 0:
#                 pass
#             else:
#                 i = score_dict[i]
#             X_feature_vectors.append(vector)
#     for vector in X_feature_vectors_scores:
#         print vector
#===============================================================================
           
def hprd_index():
    dir_entry = 'Data Sets\hprd50_bioc.xml'
    tree = ET.parse(dir_entry)
    root = tree.getroot()
    xml_dict = xml_to_dict(root)
    max_sentences = 10
    hprd50_paper_dict = make_hprd50_papers(xml_dict)   

      
#    def slicedict(d, s):                                                            
#       return {k:v for k,v in d.iteritems() if k in s}        #Only for testing, should be deleted
#    hprd50_paper_dict = slicedict(hprd50_paper_dict, ['HPRD50_d7', 'HPRD50_d26'])              #Only for testing, should be deleted

    all_predicted_relations_dict =  {'HPRD50_d33': [(('T4', 'T6'), 36, 1, [11, 1, 27, 28], 1), (('T5', 'T6'), 36, 1, [11, 1, 27, 28], 1), (('T2', 'T6'), 0), (('T3', 'T6'), 0), (('T1', 'T6'), 0), (('T5', 'T4'), 1, 1, [11], 0), (('T2', 'T4'), 0), (('T3', 'T4'), 0), (('T1', 'T4'), 0), (('T2', 'T5'), 0), (('T3', 'T5'), 0), (('T1', 'T5'), 0), (('T3', 'T2'), 4, 0, [11, 12, 14, 19, 26], 0)], 'HPRD50_d32': [(('T15', 'T14'), 0), (('T16', 'T14'), 0), (('T17', 'T14'), 0), (('T10', 'T14'), 0), (('T18', 'T14'), 0), (('T19', 'T14'), 0), (('T8', 'T14'), 0), (('T9', 'T14'), 0), (('T6', 'T14'), 0), (('T7', 'T14'), 0), (('T4', 'T14'), 0), (('T5', 'T14'), 0), (('T2', 'T14'), 0), (('T3', 'T14'), 0), (('T1', 'T14'), 0), (('T21', 'T14'), 0), (('T20', 'T14'), 0), (('T23', 'T14'), 0), (('T22', 'T14'), 0), (('T25', 'T14'), 0), (('T24', 'T14'), 0), (('T16', 'T15'), 31, 4, [11, 14, 18, 26, 27, 28], 0), (('T17', 'T15'), 0), (('T10', 'T15'), 0), (('T11', 'T15'), 0), (('T12', 'T15'), 0), (('T13', 'T15'), 0), (('T18', 'T15'), 0), (('T19', 'T15'), 0), (('T8', 'T15'), 0), (('T9', 'T15'), 0), (('T6', 'T15'), 0), (('T7', 'T15'), 0), (('T4', 'T15'), 0), (('T5', 'T15'), 0), (('T2', 'T15'), 0), (('T3', 'T15'), 0), (('T1', 'T15'), 0), (('T21', 'T15'), 0), (('T20', 'T15'), 0), (('T23', 'T15'), 0), (('T22', 'T15'), 0), (('T25', 'T15'), 0), (('T24', 'T15'), 0), (('T17', 'T16'), 0), (('T10', 'T16'), 0), (('T11', 'T16'), 0), (('T12', 'T16'), 0), (('T13', 'T16'), 0), (('T18', 'T16'), 0), (('T19', 'T16'), 0), (('T8', 'T16'), 0), (('T9', 'T16'), 0), (('T6', 'T16'), 0), (('T7', 'T16'), 0), (('T4', 'T16'), 0), (('T5', 'T16'), 0), (('T2', 'T16'), 0), (('T3', 'T16'), 0), (('T1', 'T16'), 0), (('T21', 'T16'), 0), (('T20', 'T16'), 0), (('T23', 'T16'), 0), (('T22', 'T16'), 0), (('T25', 'T16'), 0), (('T24', 'T16'), 0), (('T10', 'T17'), 0), (('T11', 'T17'), 0), (('T12', 'T17'), 0), (('T13', 'T17'), 0), (('T18', 'T17'), 51, 5, [2, 11, 14, 18, 26, 27, 28], 0), (('T19', 'T17'), 0), (('T8', 'T17'), 0), (('T9', 'T17'), 0), (('T6', 'T17'), 0), (('T7', 'T17'), 0), (('T4', 'T17'), 0), (('T5', 'T17'), 0), (('T2', 'T17'), 0), (('T3', 'T17'), 0), (('T1', 'T17'), 0), (('T21', 'T17'), 0), (('T20', 'T17'), 0), (('T23', 'T17'), 0), (('T22', 'T17'), 0), (('T25', 'T17'), 0), (('T24', 'T17'), 0), (('T11', 'T10'), 0), (('T12', 'T10'), 0), (('T13', 'T10'), 0), (('T18', 'T10'), 0), (('T19', 'T10'), 0), (('T8', 'T10'), 0), (('T9', 'T10'), 59, 2, [11, 12, 14, 19, 26, 25, 27, 28], 0), (('T6', 'T10'), 0), (('T7', 'T10'), 0), (('T4', 'T10'), 0), (('T5', 'T10'), 0), (('T2', 'T10'), 0), (('T3', 'T10'), 0), (('T1', 'T10'), 0), (('T21', 'T10'), 0), (('T20', 'T10'), 0), (('T23', 'T10'), 0), (('T22', 'T10'), 0), (('T25', 'T10'), 0), (('T24', 'T10'), 0), (('T12', 'T11'), 4, 3, [11, 12, 14, 18, 26], 0), (('T13', 'T11'), 34, 3, [11, 12, 14, 18, 26, 27, 28], 0), (('T18', 'T11'), 0), (('T19', 'T11'), 0), (('T8', 'T11'), 0), (('T9', 'T11'), 0), (('T6', 'T11'), 0), (('T7', 'T11'), 0), (('T4', 'T11'), 0), (('T5', 'T11'), 0), (('T2', 'T11'), 0), (('T3', 'T11'), 0), (('T1', 'T11'), 0), (('T21', 'T11'), 0), (('T20', 'T11'), 0), (('T23', 'T11'), 0), (('T22', 'T11'), 0), (('T25', 'T11'), 0), (('T24', 'T11'), 0), (('T13', 'T12'), 34, 3, [11, 12, 14, 18, 26, 27, 28], 0), (('T18', 'T12'), 0), (('T19', 'T12'), 0), (('T8', 'T12'), 0), (('T9', 'T12'), 0), (('T6', 'T12'), 0), (('T7', 'T12'), 0), (('T4', 'T12'), 0), (('T5', 'T12'), 0), (('T2', 'T12'), 0), (('T3', 'T12'), 0), (('T1', 'T12'), 0), (('T21', 'T12'), 0), (('T20', 'T12'), 0), (('T23', 'T12'), 0), (('T22', 'T12'), 0), (('T25', 'T12'), 0), (('T24', 'T12'), 0), (('T18', 'T13'), 0), (('T19', 'T13'), 0), (('T8', 'T13'), 0), (('T9', 'T13'), 0), (('T6', 'T13'), 0), (('T7', 'T13'), 0), (('T4', 'T13'), 0), (('T5', 'T13'), 0), (('T2', 'T13'), 0), (('T3', 'T13'), 0), (('T1', 'T13'), 0), (('T21', 'T13'), 0), (('T20', 'T13'), 0), (('T23', 'T13'), 0), (('T22', 'T13'), 0), (('T25', 'T13'), 0), (('T24', 'T13'), 0), (('T19', 'T18'), 0), (('T8', 'T18'), 0), (('T9', 'T18'), 0), (('T6', 'T18'), 0), (('T7', 'T18'), 0), (('T4', 'T18'), 0), (('T5', 'T18'), 0), (('T2', 'T18'), 0), (('T3', 'T18'), 0), (('T1', 'T18'), 0), (('T21', 'T18'), 0), (('T20', 'T18'), 0), (('T23', 'T18'), 0), (('T22', 'T18'), 0), (('T25', 'T18'), 0), (('T24', 'T18'), 0), (('T8', 'T19'), 0), (('T9', 'T19'), 0), (('T6', 'T19'), 0), (('T7', 'T19'), 0), (('T4', 'T19'), 0), (('T5', 'T19'), 0), (('T2', 'T19'), 0), (('T3', 'T19'), 0), (('T1', 'T19'), 0), (('T21', 'T19'), 11, 6, [11, 14, 18, 26, 27], 0), (('T23', 'T19'), 11, 6, [11, 14, 19, 26, 27], 0), (('T22', 'T19'), 11, 6, [11, 14, 19, 26, 27], 0), (('T25', 'T19'), 11, 6, [11, 14, 18, 26, 27], 0), (('T24', 'T19'), 11, 6, [11, 14, 19, 26, 27], 0), (('T9', 'T8'), 0), (('T4', 'T8'), 0), (('T2', 'T8'), 0), (('T3', 'T8'), 0), (('T1', 'T8'), 0), (('T21', 'T8'), 0), (('T20', 'T8'), 0), (('T23', 'T8'), 0), (('T22', 'T8'), 0), (('T25', 'T8'), 0), (('T24', 'T8'), 0), (('T6', 'T9'), 0), (('T7', 'T9'), 0), (('T4', 'T9'), 0), (('T5', 'T9'), 0), (('T2', 'T9'), 0), (('T3', 'T9'), 0), (('T1', 'T9'), 0), (('T21', 'T9'), 0), (('T20', 'T9'), 0), (('T23', 'T9'), 0), (('T22', 'T9'), 0), (('T25', 'T9'), 0), (('T24', 'T9'), 0), (('T4', 'T6'), 0), (('T2', 'T6'), 0), (('T3', 'T6'), 0), (('T1', 'T6'), 0), (('T21', 'T6'), 0), (('T20', 'T6'), 0), (('T23', 'T6'), 0), (('T22', 'T6'), 0), (('T25', 'T6'), 0), (('T24', 'T6'), 0), (('T4', 'T7'), 0), (('T2', 'T7'), 0), (('T3', 'T7'), 0), (('T1', 'T7'), 0), (('T21', 'T7'), 0), (('T20', 'T7'), 0), (('T23', 'T7'), 0), (('T22', 'T7'), 0), (('T25', 'T7'), 0), (('T24', 'T7'), 0), (('T5', 'T4'), 0), (('T2', 'T4'), 1, 0, [11, 19], 0), (('T3', 'T4'), 1, 0, [11], 0), (('T1', 'T4'), 1, 0, [11, 19], 0), (('T21', 'T4'), 0), (('T20', 'T4'), 0), (('T23', 'T4'), 0), (('T22', 'T4'), 0), (('T25', 'T4'), 0), (('T24', 'T4'), 0), (('T2', 'T5'), 0), (('T3', 'T5'), 0), (('T1', 'T5'), 0), (('T21', 'T5'), 0), (('T20', 'T5'), 0), (('T23', 'T5'), 0), (('T22', 'T5'), 0), (('T25', 'T5'), 0), (('T24', 'T5'), 0), (('T3', 'T2'), 1, 0, [11, 19], 0), (('T1', 'T2'), 1, 0, [11, 18], 0), (('T21', 'T2'), 0), (('T20', 'T2'), 0), (('T23', 'T2'), 0), (('T22', 'T2'), 0), (('T25', 'T2'), 0), (('T24', 'T2'), 0), (('T1', 'T3'), 1, 0, [11, 19], 0), (('T21', 'T3'), 0), (('T20', 'T3'), 0), (('T23', 'T3'), 0), (('T22', 'T3'), 0), (('T25', 'T3'), 0), (('T24', 'T3'), 0), (('T21', 'T1'), 0), (('T20', 'T1'), 0), (('T23', 'T1'), 0), (('T22', 'T1'), 0), (('T25', 'T1'), 0), (('T24', 'T1'), 0), (('T25', 'T21'), 11, 6, [11, 14, 18, 26, 27], 0), (('T24', 'T21'), 11, 6, [11, 14, 19, 26, 27], 0), (('T23', 'T20'), 11, 6, [11, 14, 19, 26, 27], 1), (('T22', 'T20'), 11, 6, [11, 14, 19, 26, 27], 1), (('T25', 'T20'), 11, 6, [11, 14, 18, 26, 27], 0), (('T24', 'T20'), 11, 6, [11, 14, 19, 26, 27], 0), (('T25', 'T23'), 11, 6, [11, 14, 19, 26, 27], 0), (('T24', 'T23'), 11, 6, [11, 14, 26, 27], 0), (('T25', 'T22'), 11, 6, [11, 14, 19, 26, 27], 0), (('T24', 'T22'), 11, 6, [11, 14, 26, 27], 0)], 'HPRD50_d31': [(('T2', 'T4'), 0), (('T3', 'T4'), 28, 1, [2, 12, 1], 1), (('T1', 'T4'), 0), (('T3', 'T2'), 0), (('T1', 'T2'), 1, 0, [11, 18], 1), (('T1', 'T3'), 0)], 'HPRD50_d30': [(('T3', 'T2'), 45, 0, [19, 26, 28], 0), (('T1', 'T2'), 35, 0, [18, 1, 27, 28], 1), (('T1', 'T3'), 35, 0, [19, 1, 27, 28], 1)], 'HPRD50_d37': [(('T2', 'T4'), 36, 0, [11, 19, 1, 27, 28], 0), (('T3', 'T4'), 1, 0, [11], 1), (('T1', 'T4'), 31, 0, [11, 19, 27, 28], 0), (('T3', 'T2'), 56, 0, [11, 19, 26, 27, 28], 0), (('T1', 'T2'), 11, 0, [11, 18, 27], 0), (('T1', 'T3'), 31, 0, [11, 19, 27, 28], 1)], 'HPRD50_d36': [(('T9', 'T8'), 0), (('T6', 'T8'), 25, 2, [1, 28], 0), (('T7', 'T8'), 25, 2, [1, 28], 1), (('T4', 'T8'), 0), (('T5', 'T8'), 0), (('T2', 'T8'), 0), (('T3', 'T8'), 0), (('T1', 'T8'), 0), (('T14', 'T8'), 0), (('T15', 'T8'), 0), (('T10', 'T8'), 0), (('T11', 'T8'), 0), (('T12', 'T8'), 0), (('T13', 'T8'), 0), (('T6', 'T9'), 0), (('T7', 'T9'), 0), (('T4', 'T9'), 0), (('T5', 'T9'), 0), (('T2', 'T9'), 0), (('T3', 'T9'), 0), (('T1', 'T9'), 0), (('T14', 'T9'), 0), (('T15', 'T9'), 0), (('T10', 'T9'), 0, 3, [19], 1), (('T11', 'T9'), 0), (('T12', 'T9'), 0), (('T13', 'T9'), 0), (('T7', 'T6'), 0, 2, [], 1), (('T4', 'T6'), 0), (('T5', 'T6'), 0), (('T2', 'T6'), 0), (('T3', 'T6'), 0), (('T1', 'T6'), 0), (('T14', 'T6'), 0), (('T15', 'T6'), 0), (('T10', 'T6'), 0), (('T11', 'T6'), 0), (('T12', 'T6'), 0), (('T13', 'T6'), 0), (('T4', 'T7'), 0), (('T5', 'T7'), 0), (('T2', 'T7'), 0), (('T3', 'T7'), 0), (('T1', 'T7'), 0), (('T14', 'T7'), 0), (('T15', 'T7'), 0), (('T10', 'T7'), 0), (('T11', 'T7'), 0), (('T12', 'T7'), 0), (('T13', 'T7'), 0), (('T5', 'T4'), 28, 1, [12, 26], 1), (('T2', 'T4'), 0), (('T3', 'T4'), 8, 1, [12, 1], 1), (('T1', 'T4'), 0), (('T14', 'T4'), 0), (('T15', 'T4'), 0), (('T10', 'T4'), 0), (('T11', 'T4'), 0), (('T12', 'T4'), 0), (('T13', 'T4'), 0), (('T2', 'T5'), 0), (('T3', 'T5'), 8, 1, [12, 1], 0), (('T1', 'T5'), 0), (('T14', 'T5'), 0), (('T15', 'T5'), 0), (('T10', 'T5'), 0), (('T11', 'T5'), 0), (('T12', 'T5'), 0), (('T13', 'T5'), 0), (('T3', 'T2'), 0), (('T1', 'T2'), 0, 0, [18], 1), (('T14', 'T2'), 0), (('T15', 'T2'), 0), (('T10', 'T2'), 0), (('T11', 'T2'), 0), (('T12', 'T2'), 0), (('T13', 'T2'), 0), (('T1', 'T3'), 0), (('T14', 'T3'), 0), (('T15', 'T3'), 0), (('T10', 'T3'), 0), (('T11', 'T3'), 0), (('T12', 'T3'), 0), (('T13', 'T3'), 0), (('T14', 'T1'), 0), (('T15', 'T1'), 0), (('T10', 'T1'), 0), (('T11', 'T1'), 0), (('T12', 'T1'), 0), (('T13', 'T1'), 0), (('T15', 'T14'), 45, 5, [18, 26, 28], 0), (('T10', 'T14'), 0), (('T11', 'T14'), 0), (('T12', 'T14'), 0), (('T13', 'T14'), 0), (('T10', 'T15'), 0), (('T11', 'T15'), 0), (('T12', 'T15'), 0), (('T13', 'T15'), 0), (('T11', 'T10'), 0), (('T12', 'T10'), 0), (('T13', 'T10'), 0), (('T12', 'T11'), 4, 4, [11, 12, 18], 0), (('T13', 'T11'), 49, 4, [11, 12, 18, 26, 28], 0), (('T13', 'T12'), 49, 4, [11, 12, 18, 26, 28], 1)], 'HPRD50_d35': [(('T2', 'T4'), 0), (('T3', 'T4'), 23, 1, [2, 12], 0), (('T1', 'T4'), 0), (('T3', 'T2'), 0), (('T1', 'T2'), 0, 0, [18], 0), (('T1', 'T3'), 0)], 'HPRD50_d34': [(('T5', 'T4'), 35, 1, [26, 27], 0), (('T2', 'T4'), 0), (('T3', 'T4'), 25, 1, [1, 28], 0), (('T1', 'T4'), 0), (('T2', 'T5'), 0), (('T3', 'T5'), 35, 1, [1, 27, 28], 0), (('T1', 'T5'), 0), (('T3', 'T2'), 0), (('T1', 'T2'), 38, 0, [12, 18, 1, 27, 28], 0), (('T1', 'T3'), 0)], 'HPRD50_d11': [(('T5', 'T4'), 38, 1, [2, 12, 15, 27], 1), (('T2', 'T4'), 0), (('T3', 'T4'), 0), (('T1', 'T4'), 0), (('T2', 'T5'), 0), (('T3', 'T5'), 0), (('T1', 'T5'), 0), (('T3', 'T2'), 26, 0, [11, 19, 26], 1), (('T1', 'T2'), 1, 0, [11, 18], 1), (('T1', 'T3'), 6, 0, [11, 19, 1], 1)], 'HPRD50_d10': [(('T9', 'T8'), 0), (('T6', 'T8'), 0), (('T7', 'T8'), 54, 2, [2, 11, 12, 27, 28], 1), (('T4', 'T8'), 0), (('T5', 'T8'), 0), (('T2', 'T8'), 0), (('T3', 'T8'), 0), (('T1', 'T8'), 0), (('T14', 'T8'), 0), (('T10', 'T8'), 0), (('T11', 'T8'), 0), (('T12', 'T8'), 0), (('T13', 'T8'), 0), (('T6', 'T9'), 0), (('T7', 'T9'), 0), (('T4', 'T9'), 0), (('T5', 'T9'), 0), (('T2', 'T9'), 0), (('T3', 'T9'), 0), (('T1', 'T9'), 0), (('T14', 'T9'), 0), (('T11', 'T9'), 33, 3, [12, 14, 19, 26, 27, 28], 0), (('T12', 'T9'), 0), (('T13', 'T9'), 0), (('T7', 'T6'), 0), (('T4', 'T6'), 1, 1, [11], 0), (('T5', 'T6'), 1, 1, [11], 0), (('T2', 'T6'), 0), (('T3', 'T6'), 0), (('T1', 'T6'), 0), (('T14', 'T6'), 0), (('T10', 'T6'), 0), (('T11', 'T6'), 0), (('T12', 'T6'), 0), (('T13', 'T6'), 0), (('T4', 'T7'), 0), (('T5', 'T7'), 0), (('T2', 'T7'), 0), (('T3', 'T7'), 0), (('T1', 'T7'), 0), (('T14', 'T7'), 0), (('T10', 'T7'), 0), (('T11', 'T7'), 0), (('T12', 'T7'), 0), (('T13', 'T7'), 0), (('T5', 'T4'), 1, 1, [11], 0), (('T2', 'T4'), 0), (('T3', 'T4'), 0), (('T1', 'T4'), 0), (('T14', 'T4'), 0), (('T10', 'T4'), 0), (('T11', 'T4'), 0), (('T12', 'T4'), 0), (('T13', 'T4'), 0), (('T2', 'T5'), 0), (('T3', 'T5'), 0), (('T1', 'T5'), 0), (('T14', 'T5'), 0), (('T10', 'T5'), 0), (('T11', 'T5'), 0), (('T12', 'T5'), 0), (('T13', 'T5'), 0), (('T3', 'T2'), 0, 0, [19], 0), (('T1', 'T2'), 0, 0, [18], 1), (('T14', 'T2'), 0), (('T10', 'T2'), 0), (('T11', 'T2'), 0), (('T12', 'T2'), 0), (('T13', 'T2'), 0), (('T1', 'T3'), 0, 0, [19], 1), (('T14', 'T3'), 0), (('T10', 'T3'), 0), (('T11', 'T3'), 0), (('T12', 'T3'), 0), (('T13', 'T3'), 0), (('T14', 'T1'), 0), (('T10', 'T1'), 0), (('T11', 'T1'), 0), (('T12', 'T1'), 0), (('T13', 'T1'), 0), (('T10', 'T14'), 0), (('T11', 'T14'), 0), (('T12', 'T14'), 39, 4, [11, 12, 18, 1, 27, 28], 0), (('T13', 'T14'), 39, 4, [11, 12, 18, 1, 27, 28], 0), (('T11', 'T10'), 33, 3, [12, 14, 18, 26, 27, 28], 0), (('T12', 'T10'), 0), (('T13', 'T10'), 0), (('T12', 'T11'), 0), (('T13', 'T11'), 0), (('T13', 'T12'), 4, 4, [11, 12, 18], 1)], 'HPRD50_d13': [(('T6', 'T8'), 0), (('T7', 'T8'), 18, 2, [12, 1, 27], 1), (('T4', 'T8'), 0), (('T5', 'T8'), 0), (('T2', 'T8'), 0), (('T3', 'T8'), 0), (('T1', 'T8'), 0), (('T7', 'T6'), 0), (('T4', 'T6'), 34, 1, [11, 12, 27, 28], 0), (('T5', 'T6'), 4, 1, [11, 12], 0), (('T2', 'T6'), 0), (('T3', 'T6'), 34, 1, [11, 12, 27, 28], 0), (('T1', 'T6'), 0), (('T4', 'T7'), 0), (('T5', 'T7'), 0), (('T2', 'T7'), 0), (('T3', 'T7'), 0), (('T1', 'T7'), 0), (('T5', 'T4'), 34, 1, [11, 12, 27, 28], 0), (('T2', 'T4'), 0), (('T3', 'T4'), 4, 1, [11, 12], 1), (('T1', 'T4'), 0), (('T2', 'T5'), 0), (('T3', 'T5'), 34, 1, [11, 12, 27, 28], 0), (('T1', 'T5'), 0), (('T3', 'T2'), 0), (('T1', 'T2'), 4, 0, [11, 12, 18], 1), (('T1', 'T3'), 0)], 'HPRD50_d12': [(('T5', 'T4'), 14, 1, [11, 12, 14, 26, 27], 1), (('T2', 'T4'), 0), (('T1', 'T4'), 0), (('T2', 'T5'), 0), (('T1', 'T5'), 0), (('T3', 'T2'), 0), (('T1', 'T2'), 1, 0, [11, 18], 1), (('T1', 'T3'), 0)], 'HPRD50_d15': [(('T9', 'T8'), 45, 3, [26, 28], 1), (('T6', 'T8'), 0), (('T7', 'T8'), 0), (('T4', 'T8'), 0), (('T5', 'T8'), 0), (('T2', 'T8'), 0), (('T3', 'T8'), 0), (('T1', 'T8'), 0), (('T10', 'T8'), 0), (('T11', 'T8'), 0), (('T6', 'T9'), 0), (('T7', 'T9'), 0), (('T4', 'T9'), 0), (('T5', 'T9'), 0), (('T2', 'T9'), 0), (('T3', 'T9'), 0), (('T1', 'T9'), 0), (('T10', 'T9'), 0), (('T11', 'T9'), 0), (('T7', 'T6'), 23, 2, [2, 12], 0), (('T4', 'T6'), 0), (('T5', 'T6'), 43, 2, [2, 12, 28], 1), (('T2', 'T6'), 0), (('T3', 'T6'), 0), (('T1', 'T6'), 0), (('T10', 'T6'), 0), (('T11', 'T6'), 0), (('T4', 'T7'), 0), (('T5', 'T7'), 43, 2, [2, 12, 28], 0), (('T2', 'T7'), 0), (('T3', 'T7'), 0), (('T1', 'T7'), 0), (('T10', 'T7'), 0), (('T11', 'T7'), 0), (('T5', 'T4'), 0), (('T2', 'T4'), 0), (('T3', 'T4'), 46, 1, [2, 11, 1, 28], 1), (('T1', 'T4'), 0), (('T10', 'T4'), 0), (('T11', 'T4'), 0), (('T2', 'T5'), 0), (('T3', 'T5'), 0), (('T1', 'T5'), 0), (('T10', 'T5'), 0), (('T11', 'T5'), 0), (('T3', 'T2'), 0), (('T1', 'T2'), 0, 0, [18], 1), (('T10', 'T2'), 0), (('T11', 'T2'), 0), (('T1', 'T3'), 0), (('T10', 'T3'), 0), (('T11', 'T3'), 0), (('T10', 'T1'), 0), (('T11', 'T1'), 0), (('T11', 'T10'), 48, 4, [12, 18, 26, 28], 1)], 'HPRD50_d14': [(('T2', 'T4'), 0), (('T3', 'T4'), 0, 1, [], 1), (('T1', 'T4'), 0), (('T3', 'T2'), 0), (('T1', 'T2'), 0, 0, [18], 0), (('T1', 'T3'), 0)], 'HPRD50_d17': [(('T1', 'T2'), 20, 0, [18, 28], 0)], 'HPRD50_d16': [(('T15', 'T14'), 0), (('T16', 'T14'), 0), (('T17', 'T14'), 0), (('T10', 'T14'), 0), (('T11', 'T14'), 0), (('T12', 'T14'), 3, 3, [12, 18], 1), (('T13', 'T14'), 3, 3, [12, 18], 1), (('T18', 'T14'), 0), (('T19', 'T14'), 0), (('T8', 'T14'), 0), (('T9', 'T14'), 0), (('T6', 'T14'), 0), (('T7', 'T14'), 0), (('T4', 'T14'), 0), (('T5', 'T14'), 0), (('T2', 'T14'), 0), (('T3', 'T14'), 0), (('T1', 'T14'), 0), (('T21', 'T14'), 0), (('T20', 'T14'), 0), (('T23', 'T14'), 0), (('T22', 'T14'), 0), (('T25', 'T14'), 0), (('T24', 'T14'), 0), (('T26', 'T14'), 0), (('T16', 'T15'), 3, 4, [12, 18], 0), (('T17', 'T15'), 53, 4, [12, 18, 26, 25], 1), (('T10', 'T15'), 0), (('T11', 'T15'), 0), (('T12', 'T15'), 0), (('T13', 'T15'), 0), (('T18', 'T15'), 0), (('T19', 'T15'), 0), (('T8', 'T15'), 0), (('T9', 'T15'), 0), (('T6', 'T15'), 0), (('T7', 'T15'), 0), (('T4', 'T15'), 0), (('T5', 'T15'), 0), (('T2', 'T15'), 0), (('T3', 'T15'), 0), (('T1', 'T15'), 0), (('T21', 'T15'), 0), (('T20', 'T15'), 0), (('T23', 'T15'), 0), (('T22', 'T15'), 0), (('T25', 'T15'), 0), (('T24', 'T15'), 0), (('T26', 'T15'), 0), (('T17', 'T16'), 53, 4, [12, 18, 26, 25], 1), (('T10', 'T16'), 0), (('T11', 'T16'), 0), (('T12', 'T16'), 0), (('T13', 'T16'), 0), (('T18', 'T16'), 0), (('T19', 'T16'), 0), (('T8', 'T16'), 0), (('T9', 'T16'), 0), (('T6', 'T16'), 0), (('T7', 'T16'), 0), (('T4', 'T16'), 0), (('T5', 'T16'), 0), (('T2', 'T16'), 0), (('T3', 'T16'), 0), (('T1', 'T16'), 0), (('T21', 'T16'), 0), (('T20', 'T16'), 0), (('T23', 'T16'), 0), (('T22', 'T16'), 0), (('T25', 'T16'), 0), (('T24', 'T16'), 0), (('T26', 'T16'), 0), (('T10', 'T17'), 0), (('T11', 'T17'), 0), (('T12', 'T17'), 0), (('T13', 'T17'), 0), (('T18', 'T17'), 0), (('T19', 'T17'), 0), (('T8', 'T17'), 0), (('T9', 'T17'), 0), (('T6', 'T17'), 0), (('T7', 'T17'), 0), (('T4', 'T17'), 0), (('T5', 'T17'), 0), (('T2', 'T17'), 0), (('T3', 'T17'), 0), (('T1', 'T17'), 0), (('T21', 'T17'), 0), (('T20', 'T17'), 0), (('T23', 'T17'), 0), (('T22', 'T17'), 0), (('T25', 'T17'), 0), (('T24', 'T17'), 0), (('T26', 'T17'), 0), (('T11', 'T10'), 3, 2, [12, 18], 1), (('T12', 'T10'), 0), (('T13', 'T10'), 0), (('T18', 'T10'), 0), (('T19', 'T10'), 0), (('T8', 'T10'), 0), (('T9', 'T10'), 0), (('T6', 'T10'), 0), (('T7', 'T10'), 0), (('T4', 'T10'), 0), (('T5', 'T10'), 0), (('T2', 'T10'), 0), (('T3', 'T10'), 0), (('T1', 'T10'), 0), (('T21', 'T10'), 0), (('T20', 'T10'), 0), (('T23', 'T10'), 0), (('T22', 'T10'), 0), (('T25', 'T10'), 0), (('T24', 'T10'), 0), (('T26', 'T10'), 0), (('T12', 'T11'), 0), (('T13', 'T11'), 0), (('T18', 'T11'), 0), (('T19', 'T11'), 0), (('T8', 'T11'), 0), (('T9', 'T11'), 0), (('T6', 'T11'), 0), (('T7', 'T11'), 0), (('T4', 'T11'), 0), (('T5', 'T11'), 0), (('T2', 'T11'), 0), (('T3', 'T11'), 0), (('T1', 'T11'), 0), (('T21', 'T11'), 0), (('T20', 'T11'), 0), (('T23', 'T11'), 0), (('T22', 'T11'), 0), (('T25', 'T11'), 0), (('T24', 'T11'), 0), (('T26', 'T11'), 0), (('T13', 'T12'), 3, 3, [12, 18], 0), (('T18', 'T12'), 0), (('T19', 'T12'), 0), (('T8', 'T12'), 0), (('T9', 'T12'), 0), (('T6', 'T12'), 0), (('T7', 'T12'), 0), (('T4', 'T12'), 0), (('T5', 'T12'), 0), (('T2', 'T12'), 0), (('T3', 'T12'), 0), (('T1', 'T12'), 0), (('T21', 'T12'), 0), (('T20', 'T12'), 0), (('T23', 'T12'), 0), (('T22', 'T12'), 0), (('T25', 'T12'), 0), (('T24', 'T12'), 0), (('T26', 'T12'), 0), (('T18', 'T13'), 0), (('T19', 'T13'), 0), (('T8', 'T13'), 0), (('T9', 'T13'), 0), (('T6', 'T13'), 0), (('T7', 'T13'), 0), (('T4', 'T13'), 0), (('T5', 'T13'), 0), (('T2', 'T13'), 0), (('T3', 'T13'), 0), (('T1', 'T13'), 0), (('T21', 'T13'), 0), (('T20', 'T13'), 0), (('T23', 'T13'), 0), (('T22', 'T13'), 0), (('T25', 'T13'), 0), (('T24', 'T13'), 0), (('T26', 'T13'), 0), (('T19', 'T18'), 1, 5, [11, 18], 0), (('T8', 'T18'), 0), (('T9', 'T18'), 0), (('T6', 'T18'), 0), (('T7', 'T18'), 0), (('T4', 'T18'), 0), (('T5', 'T18'), 0), (('T2', 'T18'), 0), (('T3', 'T18'), 0), (('T1', 'T18'), 0), (('T21', 'T18'), 1, 5, [11, 18], 1), (('T20', 'T18'), 1, 5, [11, 18], 0), (('T23', 'T18'), 0), (('T22', 'T18'), 1, 5, [11, 19], 1), (('T25', 'T18'), 0), (('T24', 'T18'), 0), (('T26', 'T18'), 0), (('T8', 'T19'), 0), (('T9', 'T19'), 0), (('T6', 'T19'), 0), (('T7', 'T19'), 0), (('T4', 'T19'), 0), (('T5', 'T19'), 0), (('T2', 'T19'), 0), (('T3', 'T19'), 0), (('T1', 'T19'), 0), (('T21', 'T19'), 1, 5, [11, 18], 1), (('T20', 'T19'), 1, 5, [11, 18], 0), (('T23', 'T19'), 0), (('T22', 'T19'), 1, 5, [11, 19], 1), (('T25', 'T19'), 0), (('T24', 'T19'), 0), (('T26', 'T19'), 0), (('T9', 'T8'), 3, 1, [12], 0), (('T6', 'T8'), 3, 1, [12], 0), (('T7', 'T8'), 3, 1, [12], 0), (('T4', 'T8'), 3, 1, [12], 0), (('T5', 'T8'), 3, 1, [12], 0), (('T2', 'T8'), 0), (('T3', 'T8'), 0), (('T1', 'T8'), 0), (('T21', 'T8'), 0), (('T20', 'T8'), 0), (('T23', 'T8'), 0), (('T22', 'T8'), 0), (('T25', 'T8'), 0), (('T24', 'T8'), 0), (('T26', 'T8'), 0), (('T6', 'T9'), 3, 1, [12], 0), (('T7', 'T9'), 3, 1, [12], 0), (('T4', 'T9'), 3, 1, [12], 0), (('T5', 'T9'), 3, 1, [12], 0), (('T2', 'T9'), 0), (('T3', 'T9'), 0), (('T1', 'T9'), 0), (('T21', 'T9'), 0), (('T20', 'T9'), 0), (('T23', 'T9'), 0), (('T22', 'T9'), 0), (('T25', 'T9'), 0), (('T24', 'T9'), 0), (('T26', 'T9'), 0), (('T7', 'T6'), 3, 1, [12], 0), (('T4', 'T6'), 3, 1, [12], 0), (('T5', 'T6'), 3, 1, [12], 0), (('T2', 'T6'), 0), (('T3', 'T6'), 0), (('T1', 'T6'), 0), (('T21', 'T6'), 0), (('T20', 'T6'), 0), (('T23', 'T6'), 0), (('T22', 'T6'), 0), (('T25', 'T6'), 0), (('T24', 'T6'), 0), (('T26', 'T6'), 0), (('T4', 'T7'), 3, 1, [12], 0), (('T5', 'T7'), 3, 1, [12], 0), (('T2', 'T7'), 0), (('T3', 'T7'), 0), (('T1', 'T7'), 0), (('T21', 'T7'), 0), (('T20', 'T7'), 0), (('T23', 'T7'), 0), (('T22', 'T7'), 0), (('T25', 'T7'), 0), (('T24', 'T7'), 0), (('T26', 'T7'), 0), (('T5', 'T4'), 3, 1, [12], 0), (('T2', 'T4'), 0), (('T3', 'T4'), 0), (('T1', 'T4'), 0), (('T21', 'T4'), 0), (('T20', 'T4'), 0), (('T23', 'T4'), 0), (('T22', 'T4'), 0), (('T25', 'T4'), 0), (('T24', 'T4'), 0), (('T26', 'T4'), 0), (('T2', 'T5'), 0), (('T3', 'T5'), 0), (('T1', 'T5'), 0), (('T21', 'T5'), 0), (('T20', 'T5'), 0), (('T23', 'T5'), 0), (('T22', 'T5'), 0), (('T25', 'T5'), 0), (('T24', 'T5'), 0), (('T26', 'T5'), 0), (('T3', 'T2'), 36, 0, [11, 19, 26, 27], 1), (('T1', 'T2'), 1, 0, [11, 18], 0), (('T21', 'T2'), 0), (('T20', 'T2'), 0), (('T23', 'T2'), 0), (('T22', 'T2'), 0), (('T25', 'T2'), 0), (('T24', 'T2'), 0), (('T26', 'T2'), 0), (('T1', 'T3'), 11, 0, [11, 19, 27], 1), (('T21', 'T3'), 0), (('T20', 'T3'), 0), (('T23', 'T3'), 0), (('T22', 'T3'), 0), (('T25', 'T3'), 0), (('T24', 'T3'), 0), (('T26', 'T3'), 0), (('T21', 'T1'), 0), (('T20', 'T1'), 0), (('T23', 'T1'), 0), (('T22', 'T1'), 0), (('T25', 'T1'), 0), (('T24', 'T1'), 0), (('T26', 'T1'), 0), (('T20', 'T21'), 1, 5, [11, 18], 1), (('T23', 'T21'), 0), (('T22', 'T21'), 1, 5, [11, 19], 0), (('T25', 'T21'), 0), (('T24', 'T21'), 0), (('T26', 'T21'), 0), (('T23', 'T20'), 0), (('T22', 'T20'), 1, 5, [11, 19], 1), (('T25', 'T20'), 0), (('T24', 'T20'), 0), (('T26', 'T20'), 0), (('T22', 'T23'), 0), (('T25', 'T23'), 4, 6, [11, 12, 19], 1), (('T24', 'T23'), 4, 6, [11, 12], 1), (('T26', 'T23'), 29, 6, [11, 12, 26], 0), (('T25', 'T22'), 0), (('T24', 'T22'), 0), (('T26', 'T22'), 0), (('T24', 'T25'), 4, 6, [11, 12, 19], 0), (('T26', 'T25'), 29, 6, [11, 12, 19, 26], 0), (('T26', 'T24'), 29, 6, [11, 12, 26], 0)], 'HPRD50_d9': [(('T5', 'T4'), 49, 1, [11, 12, 26, 28], 0), (('T2', 'T4'), 0), (('T3', 'T4'), 0), (('T1', 'T4'), 0), (('T2', 'T5'), 0), (('T3', 'T5'), 0), (('T1', 'T5'), 0), (('T3', 'T2'), 25, 0, [19, 26], 0), (('T1', 'T2'), 30, 0, [18, 27, 28], 0), (('T1', 'T3'), 35, 0, [19, 1, 27, 28], 0)], 'HPRD50_d8': [], 'HPRD50_d38': [(('T6', 'T8'), 0), (('T4', 'T8'), 0), (('T5', 'T8'), 0), (('T2', 'T8'), 0), (('T3', 'T8'), 0), (('T1', 'T8'), 0), (('T6', 'T9'), 0), (('T4', 'T9'), 0), (('T5', 'T9'), 0), (('T2', 'T9'), 0), (('T3', 'T9'), 0), (('T1', 'T9'), 0), (('T7', 'T6'), 0), (('T4', 'T6'), 0, 0, [2, 14, 1], 0), (('T5', 'T6'), 0, 0, [2, 14, 1], 0), (('T2', 'T6'), 0, 0, [2, 14, 19, 1], 0), (('T3', 'T6'), 0, 0, [2, 14, 1], 0), (('T1', 'T6'), 30, 0, [2, 14, 19, 1, 27, 28], 1), (('T4', 'T7'), 0), (('T5', 'T7'), 0), (('T2', 'T7'), 0), (('T3', 'T7'), 0), (('T1', 'T7'), 0), (('T5', 'T4'), 20, 0, [2, 14, 26], 0), (('T2', 'T4'), -5, 0, [2, 14, 19], 0), (('T3', 'T4'), -5, 0, [2, 14], 0), (('T1', 'T4'), 30, 0, [2, 14, 19, 1, 27, 28], 1), (('T2', 'T5'), 0, 0, [2, 14, 19, 1], 0), (('T3', 'T5'), 0, 0, [2, 14, 1], 0), (('T1', 'T5'), 30, 0, [2, 14, 19, 1, 27, 28], 1), (('T3', 'T2'), -5, 0, [2, 14, 19], 0), (('T1', 'T2'), 30, 0, [2, 14, 18, 1, 27, 28], 1), (('T1', 'T3'), 30, 0, [2, 14, 19, 1, 27, 28], 1)], 'HPRD50_d1': [(('T9', 'T8'), 20, 3, [2], 1), (('T6', 'T8'), 0), (('T7', 'T8'), 0), (('T4', 'T8'), 0), (('T5', 'T8'), 0), (('T2', 'T8'), 0), (('T3', 'T8'), 0), (('T1', 'T8'), 0), (('T6', 'T9'), 0), (('T7', 'T9'), 0), (('T4', 'T9'), 0), (('T5', 'T9'), 0), (('T2', 'T9'), 0), (('T3', 'T9'), 0), (('T1', 'T9'), 0), (('T7', 'T6'), 49, 2, [2, 11, 12, 26], 1), (('T4', 'T6'), 0), (('T5', 'T6'), 0), (('T2', 'T6'), 0), (('T3', 'T6'), 0), (('T1', 'T6'), 0), (('T4', 'T7'), 0), (('T5', 'T7'), 0), (('T2', 'T7'), 0), (('T3', 'T7'), 0), (('T1', 'T7'), 0), (('T5', 'T4'), 1, 1, [11], 0), (('T2', 'T4'), 0), (('T3', 'T4'), 6, 1, [11, 1], 0), (('T1', 'T4'), 0), (('T2', 'T5'), 0), (('T3', 'T5'), 6, 1, [11, 1], 0), (('T1', 'T5'), 0), (('T3', 'T2'), 0), (('T1', 'T2'), 10, 0, [18, 27], 1), (('T1', 'T3'), 0)], 'HPRD50_d0': [(('T3', 'T2'), 1, 0, [11, 19], 0), (('T1', 'T2'), 1, 0, [11, 18], 0), (('T1', 'T3'), 1, 0, [11, 19], 0)], 'HPRD50_d3': [(('T7', 'T6'), 20, 2, [2], 0), (('T4', 'T6'), 0), (('T5', 'T6'), 20, 2, [2], 1), (('T2', 'T6'), 0), (('T3', 'T6'), 0), (('T1', 'T6'), 0), (('T4', 'T7'), 0), (('T5', 'T7'), 20, 2, [2], 0), (('T2', 'T7'), 0), (('T3', 'T7'), 0), (('T1', 'T7'), 0), (('T5', 'T4'), 0), (('T2', 'T4'), 0), (('T3', 'T4'), 0, 1, [], 0), (('T1', 'T4'), 0), (('T2', 'T5'), 0), (('T3', 'T5'), 0), (('T1', 'T5'), 0), (('T3', 'T2'), 0), (('T1', 'T2'), 20, 0, [2, 18], 1), (('T1', 'T3'), 0)], 'HPRD50_d2': [(('T9', 'T8'), 3, 3, [12], 0), (('T6', 'T8'), 0), (('T7', 'T8'), 0), (('T4', 'T8'), 0), (('T5', 'T8'), 0), (('T2', 'T8'), 0), (('T3', 'T8'), 0), (('T1', 'T8'), 0), (('T10', 'T8'), 58, 3, [12, 19, 26, 27, 28], 0), (('T6', 'T9'), 0), (('T7', 'T9'), 0), (('T4', 'T9'), 0), (('T5', 'T9'), 0), (('T2', 'T9'), 0), (('T3', 'T9'), 0), (('T1', 'T9'), 0), (('T10', 'T9'), 58, 3, [12, 19, 26, 27, 28], 0), (('T7', 'T6'), -5, 2, [2, 14], 0), (('T4', 'T6'), 0), (('T5', 'T6'), 0), (('T2', 'T6'), 0), (('T3', 'T6'), 0), (('T1', 'T6'), 0), (('T10', 'T6'), 0), (('T4', 'T7'), 0), (('T5', 'T7'), 0), (('T2', 'T7'), 0), (('T3', 'T7'), 0), (('T1', 'T7'), 0), (('T10', 'T7'), 0), (('T2', 'T4'), 0), (('T3', 'T4'), 0), (('T1', 'T4'), 0), (('T10', 'T4'), 0), (('T2', 'T5'), 0), (('T3', 'T5'), 0), (('T1', 'T5'), 0), (('T10', 'T5'), 0), (('T3', 'T2'), 20, 0, [2, 19], 0), (('T1', 'T2'), 20, 0, [2, 18], 0), (('T10', 'T2'), 0), (('T1', 'T3'), 20, 0, [2, 19], 0), (('T10', 'T3'), 0), (('T10', 'T1'), 0)], 'HPRD50_d5': [(('T9', 'T8'), 0), (('T6', 'T8'), 0), (('T7', 'T8'), 24, 3, [11, 12, 28], 1), (('T4', 'T8'), 0), (('T5', 'T8'), 0), (('T2', 'T8'), 0), (('T3', 'T8'), 0), (('T1', 'T8'), 0), (('T10', 'T8'), 0), (('T11', 'T8'), 0), (('T12', 'T8'), 0), (('T6', 'T9'), 0), (('T7', 'T9'), 0), (('T4', 'T9'), 0), (('T5', 'T9'), 0), (('T2', 'T9'), 0), (('T3', 'T9'), 0), (('T1', 'T9'), 0), (('T10', 'T9'), 56, 4, [11, 19, 26, 27, 28], 1), (('T11', 'T9'), 0), (('T12', 'T9'), 0), (('T7', 'T6'), 0), (('T4', 'T6'), 0), (('T5', 'T6'), 1, 2, [11], 1), (('T2', 'T6'), 0), (('T3', 'T6'), 0), (('T1', 'T6'), 0), (('T10', 'T6'), 0), (('T11', 'T6'), 0), (('T12', 'T6'), 0), (('T4', 'T7'), 0), (('T5', 'T7'), 0), (('T2', 'T7'), 0), (('T3', 'T7'), 0), (('T1', 'T7'), 0), (('T10', 'T7'), 0), (('T11', 'T7'), 0), (('T12', 'T7'), 0), (('T5', 'T4'), 0), (('T2', 'T4'), 0), (('T3', 'T4'), 14, 1, [11, 12, 27], 1), (('T1', 'T4'), 0), (('T10', 'T4'), 0), (('T11', 'T4'), 0), (('T12', 'T4'), 0), (('T2', 'T5'), 0), (('T3', 'T5'), 0), (('T1', 'T5'), 0), (('T10', 'T5'), 0), (('T11', 'T5'), 0), (('T12', 'T5'), 0), (('T3', 'T2'), 0), (('T1', 'T2'), 21, 0, [2, 11, 18], 1), (('T10', 'T2'), 0), (('T11', 'T2'), 0), (('T12', 'T2'), 0), (('T1', 'T3'), 0), (('T10', 'T3'), 0), (('T11', 'T3'), 0), (('T12', 'T3'), 0), (('T10', 'T1'), 0), (('T11', 'T1'), 0), (('T12', 'T1'), 0), (('T11', 'T10'), 0), (('T12', 'T10'), 0), (('T12', 'T11'), 46, 5, [11, 18, 26, 28], 1)], 'HPRD50_d4': [(('T4', 'T6'), 0, 1, [], 0), (('T5', 'T6'), 0, 1, [], 0), (('T2', 'T6'), 0), (('T3', 'T6'), 50, 1, [1, 24, 28], 1), (('T1', 'T6'), 0), (('T5', 'T4'), 0, 1, [], 0), (('T2', 'T4'), 0), (('T3', 'T4'), 50, 1, [1, 24, 28], 1), (('T1', 'T4'), 0), (('T2', 'T5'), 0), (('T3', 'T5'), 50, 1, [1, 24, 28], 1), (('T1', 'T5'), 0), (('T3', 'T2'), 0), (('T1', 'T2'), 0, 0, [18], 0), (('T1', 'T3'), 0)], 'HPRD50_d7': [(('T9', 'T8'), 0), (('T4', 'T8'), 0), (('T2', 'T8'), 0), (('T3', 'T8'), 0), (('T1', 'T8'), 0), (('T10', 'T8'), 0), (('T6', 'T9'), 0), (('T7', 'T9'), 0), (('T4', 'T9'), 0), (('T5', 'T9'), 0), (('T2', 'T9'), 0), (('T3', 'T9'), 0), (('T1', 'T9'), 0), (('T10', 'T9'), 46, 3, [11, 19, 26, 28], 0), (('T7', 'T6'), 33, 2, [12, 14, 26, 27, 28], 0), (('T4', 'T6'), 0), (('T2', 'T6'), 0), (('T3', 'T6'), 0), (('T1', 'T6'), 0), (('T10', 'T6'), 0), (('T4', 'T7'), 0), (('T2', 'T7'), 0), (('T3', 'T7'), 0), (('T1', 'T7'), 0), (('T10', 'T7'), 0), (('T5', 'T4'), 0), (('T2', 'T4'), 0), (('T1', 'T4'), 0), (('T10', 'T4'), 0), (('T2', 'T5'), 0), (('T3', 'T5'), 0), (('T1', 'T5'), 0), (('T10', 'T5'), 0), (('T3', 'T2'), 0), (('T1', 'T2'), 1, 0, [11, 18], 1), (('T10', 'T2'), 0), (('T1', 'T3'), 0), (('T10', 'T3'), 0), (('T10', 'T1'), 0)], 'HPRD50_d6': [(('T9', 'T8'), 0, 3, [], 0), (('T6', 'T8'), 0), (('T7', 'T8'), 0), (('T4', 'T8'), 0), (('T5', 'T8'), 0), (('T2', 'T8'), 0), (('T3', 'T8'), 0), (('T1', 'T8'), 0), (('T10', 'T8'), 0, 3, [19], 0), (('T11', 'T8'), 0), (('T12', 'T8'), 0), (('T6', 'T9'), 0), (('T7', 'T9'), 0), (('T4', 'T9'), 0), (('T5', 'T9'), 0), (('T2', 'T9'), 0), (('T3', 'T9'), 0), (('T1', 'T9'), 0), (('T10', 'T9'), 0, 3, [19], 0), (('T11', 'T9'), 0), (('T12', 'T9'), 0), (('T7', 'T6'), 0, 2, [], 0), (('T4', 'T6'), 0), (('T5', 'T6'), 0, 2, [], 0), (('T2', 'T6'), 0), (('T3', 'T6'), 0), (('T1', 'T6'), 0), (('T10', 'T6'), 0), (('T11', 'T6'), 0), (('T12', 'T6'), 0), (('T4', 'T7'), 0), (('T5', 'T7'), 0, 2, [], 0), (('T2', 'T7'), 0), (('T3', 'T7'), 0), (('T1', 'T7'), 0), (('T10', 'T7'), 0), (('T11', 'T7'), 0), (('T12', 'T7'), 0), (('T5', 'T4'), 0), (('T2', 'T4'), 0), (('T3', 'T4'), 3, 1, [12], 0), (('T1', 'T4'), 0), (('T10', 'T4'), 0), (('T11', 'T4'), 0), (('T12', 'T4'), 0), (('T2', 'T5'), 0), (('T3', 'T5'), 0), (('T1', 'T5'), 0), (('T10', 'T5'), 0), (('T11', 'T5'), 0), (('T12', 'T5'), 0), (('T3', 'T2'), 0), (('T1', 'T2'), 0, 0, [18], 0), (('T10', 'T2'), 0), (('T11', 'T2'), 0), (('T12', 'T2'), 0), (('T1', 'T3'), 0), (('T10', 'T3'), 0), (('T11', 'T3'), 0), (('T12', 'T3'), 0), (('T10', 'T1'), 0), (('T11', 'T1'), 0), (('T12', 'T1'), 0), (('T11', 'T10'), 0), (('T12', 'T10'), 0), (('T12', 'T11'), 0, 4, [18], 1)], 'HPRD50_d28': [(('T9', 'T8'), 0), (('T4', 'T8'), 0), (('T5', 'T8'), 0), (('T2', 'T8'), 0), (('T3', 'T8'), 0), (('T1', 'T8'), 0), (('T10', 'T8'), 0), (('T11', 'T8'), 0), (('T6', 'T9'), 0), (('T7', 'T9'), 0), (('T4', 'T9'), 0), (('T5', 'T9'), 0), (('T2', 'T9'), 0), (('T3', 'T9'), 0), (('T1', 'T9'), 0), (('T11', 'T9'), 30, 3, [14, 19, 26, 27, 28], 0), (('T7', 'T6'), 33, 2, [12, 14, 26, 27, 28], 0), (('T4', 'T6'), 0), (('T5', 'T6'), 0), (('T2', 'T6'), 0), (('T3', 'T6'), 0), (('T1', 'T6'), 0), (('T10', 'T6'), 0), (('T11', 'T6'), 0), (('T4', 'T7'), 0), (('T5', 'T7'), 0), (('T2', 'T7'), 0), (('T3', 'T7'), 0), (('T1', 'T7'), 0), (('T10', 'T7'), 0), (('T11', 'T7'), 0), (('T5', 'T4'), 48, 1, [2, 12, 26], 1), (('T2', 'T4'), 0), (('T3', 'T4'), 0), (('T1', 'T4'), 0), (('T10', 'T4'), 0), (('T11', 'T4'), 0), (('T2', 'T5'), 0), (('T3', 'T5'), 0), (('T1', 'T5'), 0), (('T10', 'T5'), 0), (('T11', 'T5'), 0), (('T3', 'T2'), -5, 0, [2, 14, 19], 0), (('T1', 'T2'), 0, 0, [2, 14, 18, 1], 1), (('T10', 'T2'), 0), (('T11', 'T2'), 0), (('T1', 'T3'), 0, 0, [2, 14, 19, 1], 1), (('T10', 'T3'), 0), (('T11', 'T3'), 0), (('T10', 'T1'), 0), (('T11', 'T1'), 0), (('T11', 'T10'), 30, 3, [14, 18, 26, 27, 28], 0)], 'HPRD50_d29': [(('T9', 'T8'), 41, 2, [2, 11, 28], 1), (('T6', 'T8'), 0), (('T7', 'T8'), 0), (('T4', 'T8'), 0), (('T5', 'T8'), 0), (('T2', 'T8'), 0), (('T3', 'T8'), 0), (('T1', 'T8'), 0), (('T10', 'T8'), 41, 2, [2, 11, 19, 28], 0), (('T11', 'T8'), 0), (('T12', 'T8'), 0), (('T13', 'T8'), 0), (('T6', 'T9'), 0), (('T7', 'T9'), 0), (('T4', 'T9'), 0), (('T5', 'T9'), 0), (('T2', 'T9'), 0), (('T3', 'T9'), 0), (('T1', 'T9'), 0), (('T10', 'T9'), 21, 2, [2, 11, 19], 0), (('T11', 'T9'), 0), (('T12', 'T9'), 0), (('T13', 'T9'), 0), (('T7', 'T6'), 34, 1, [11, 12, 14, 26, 27, 28], 1), (('T2', 'T6'), 0), (('T1', 'T6'), 0), (('T10', 'T6'), 0), (('T11', 'T6'), 0), (('T12', 'T6'), 0), (('T13', 'T6'), 0), (('T2', 'T7'), 0), (('T1', 'T7'), 0), (('T10', 'T7'), 0), (('T11', 'T7'), 0), (('T12', 'T7'), 0), (('T13', 'T7'), 0), (('T2', 'T4'), 0), (('T1', 'T4'), 0), (('T10', 'T4'), 0), (('T11', 'T4'), 0), (('T12', 'T4'), 0), (('T13', 'T4'), 0), (('T2', 'T5'), 0), (('T1', 'T5'), 0), (('T10', 'T5'), 0), (('T11', 'T5'), 0), (('T12', 'T5'), 0), (('T13', 'T5'), 0), (('T3', 'T2'), 0), (('T1', 'T2'), 1, 0, [11, 18], 1), (('T10', 'T2'), 0), (('T11', 'T2'), 0), (('T12', 'T2'), 0), (('T13', 'T2'), 0), (('T1', 'T3'), 0), (('T10', 'T3'), 0), (('T11', 'T3'), 0), (('T12', 'T3'), 0), (('T13', 'T3'), 0), (('T10', 'T1'), 0), (('T11', 'T1'), 0), (('T12', 'T1'), 0), (('T13', 'T1'), 0), (('T11', 'T10'), 0), (('T12', 'T10'), 0), (('T13', 'T10'), 0), (('T12', 'T11'), 51, 3, [11, 18, 26, 25], 1), (('T13', 'T11'), 51, 3, [11, 18, 26, 25], 0), (('T13', 'T12'), 1, 3, [11, 18], 0)], 'HPRD50_d40': [(('T2', 'T4'), 0), (('T3', 'T4'), 9, 1, [11, 12, 1], 1), (('T1', 'T4'), 0), (('T3', 'T2'), 0), (('T1', 'T2'), 29, 0, [11, 12, 18, 1, 28], 1), (('T1', 'T3'), 0)], 'HPRD50_d41': [(('T1', 'T2'), 55, 0, [2, 14, 18, 1, 24, 27, 28], 1)], 'HPRD50_d24': [(('T4', 'T6'), 6, 1, [11, 1], 0), (('T5', 'T6'), 6, 1, [11, 1], 1), (('T2', 'T6'), 0), (('T3', 'T6'), 6, 1, [11, 1], 0), (('T1', 'T6'), 0), (('T5', 'T4'), 26, 1, [11, 26], 1), (('T2', 'T4'), 0), (('T3', 'T4'), 1, 1, [11], 0), (('T1', 'T4'), 0), (('T2', 'T5'), 0), (('T3', 'T5'), 6, 1, [11, 1], 1), (('T1', 'T5'), 0), (('T3', 'T2'), 0), (('T1', 'T2'), 1, 0, [11, 18], 1), (('T1', 'T3'), 0)], 'HPRD50_d25': [(('T9', 'T8'), 1, 2, [11], 0), (('T6', 'T8'), 0), (('T7', 'T8'), 26, 2, [11, 1, 28], 0), (('T4', 'T8'), 0), (('T5', 'T8'), 0), (('T2', 'T8'), 0), (('T3', 'T8'), 0), (('T1', 'T8'), 0), (('T6', 'T9'), 0), (('T7', 'T9'), 26, 2, [11, 1, 28], 0), (('T4', 'T9'), 0), (('T5', 'T9'), 0), (('T2', 'T9'), 0), (('T3', 'T9'), 0), (('T1', 'T9'), 0), (('T7', 'T6'), 0), (('T4', 'T6'), 3, 1, [12], 0), (('T5', 'T6'), 3, 1, [12], 0), (('T2', 'T6'), 0), (('T3', 'T6'), 0), (('T1', 'T6'), 0), (('T4', 'T7'), 0), (('T5', 'T7'), 0), (('T2', 'T7'), 0), (('T3', 'T7'), 0), (('T1', 'T7'), 0), (('T5', 'T4'), 3, 1, [12], 0), (('T2', 'T4'), 0), (('T3', 'T4'), 0), (('T1', 'T4'), 0), (('T2', 'T5'), 0), (('T3', 'T5'), 0), (('T1', 'T5'), 0), (('T3', 'T2'), 20, 0, [2, 19], 0), (('T1', 'T2'), 20, 0, [2, 18], 1), (('T1', 'T3'), 20, 0, [2, 19], 0)], 'HPRD50_d26': [(('T15', 'T14'), 0), (('T16', 'T14'), 0), (('T17', 'T14'), 0), (('T10', 'T14'), 0), (('T11', 'T14'), 0), (('T12', 'T14'), 0), (('T13', 'T14'), 18, 4, [12, 18, 1, 27], 1), (('T18', 'T14'), 0), (('T19', 'T14'), 0), (('T8', 'T14'), 0), (('T9', 'T14'), 0), (('T6', 'T14'), 0), (('T7', 'T14'), 0), (('T4', 'T14'), 0), (('T5', 'T14'), 0), (('T2', 'T14'), 0), (('T3', 'T14'), 0), (('T1', 'T14'), 0), (('T21', 'T14'), 0), (('T20', 'T14'), 0), (('T23', 'T14'), 0), (('T22', 'T14'), 0), (('T25', 'T14'), 0), (('T24', 'T14'), 0), (('T26', 'T14'), 0), (('T16', 'T15'), 21, 5, [11, 18, 28], 0), (('T17', 'T15'), 21, 5, [11, 18, 28], 1), (('T10', 'T15'), 0), (('T11', 'T15'), 0), (('T12', 'T15'), 0), (('T13', 'T15'), 0), (('T18', 'T15'), 0), (('T19', 'T15'), 0), (('T8', 'T15'), 0), (('T9', 'T15'), 0), (('T6', 'T15'), 0), (('T7', 'T15'), 0), (('T4', 'T15'), 0), (('T5', 'T15'), 0), (('T2', 'T15'), 0), (('T3', 'T15'), 0), (('T1', 'T15'), 0), (('T21', 'T15'), 0), (('T20', 'T15'), 0), (('T23', 'T15'), 0), (('T22', 'T15'), 0), (('T25', 'T15'), 0), (('T24', 'T15'), 0), (('T26', 'T15'), 0), (('T17', 'T16'), 1, 5, [11, 18], 1), (('T10', 'T16'), 0), (('T11', 'T16'), 0), (('T12', 'T16'), 0), (('T13', 'T16'), 0), (('T18', 'T16'), 0), (('T19', 'T16'), 0), (('T8', 'T16'), 0), (('T9', 'T16'), 0), (('T6', 'T16'), 0), (('T7', 'T16'), 0), (('T4', 'T16'), 0), (('T5', 'T16'), 0), (('T2', 'T16'), 0), (('T3', 'T16'), 0), (('T1', 'T16'), 0), (('T21', 'T16'), 0), (('T20', 'T16'), 0), (('T23', 'T16'), 0), (('T22', 'T16'), 0), (('T25', 'T16'), 0), (('T24', 'T16'), 0), (('T26', 'T16'), 0), (('T10', 'T17'), 0), (('T11', 'T17'), 0), (('T12', 'T17'), 0), (('T13', 'T17'), 0), (('T18', 'T17'), 0), (('T19', 'T17'), 0), (('T8', 'T17'), 0), (('T9', 'T17'), 0), (('T6', 'T17'), 0), (('T7', 'T17'), 0), (('T4', 'T17'), 0), (('T5', 'T17'), 0), (('T2', 'T17'), 0), (('T3', 'T17'), 0), (('T1', 'T17'), 0), (('T21', 'T17'), 0), (('T20', 'T17'), 0), (('T23', 'T17'), 0), (('T22', 'T17'), 0), (('T25', 'T17'), 0), (('T24', 'T17'), 0), (('T26', 'T17'), 0), (('T11', 'T10'), 28, 3, [12, 18, 26], 1), (('T12', 'T10'), 38, 3, [12, 18, 26, 27], 0), (('T13', 'T10'), 0), (('T18', 'T10'), 0), (('T19', 'T10'), 0), (('T8', 'T10'), 0), (('T9', 'T10'), 0), (('T6', 'T10'), 0), (('T7', 'T10'), 0), (('T4', 'T10'), 0), (('T5', 'T10'), 0), (('T2', 'T10'), 0), (('T3', 'T10'), 0), (('T1', 'T10'), 0), (('T21', 'T10'), 0), (('T20', 'T10'), 0), (('T23', 'T10'), 0), (('T22', 'T10'), 0), (('T25', 'T10'), 0), (('T24', 'T10'), 0), (('T26', 'T10'), 0), (('T12', 'T11'), 38, 3, [12, 18, 26, 27], 1), (('T13', 'T11'), 0), (('T18', 'T11'), 0), (('T19', 'T11'), 0), (('T8', 'T11'), 0), (('T9', 'T11'), 0), (('T6', 'T11'), 0), (('T7', 'T11'), 0), (('T4', 'T11'), 0), (('T5', 'T11'), 0), (('T2', 'T11'), 0), (('T3', 'T11'), 0), (('T1', 'T11'), 0), (('T21', 'T11'), 0), (('T20', 'T11'), 0), (('T23', 'T11'), 0), (('T22', 'T11'), 0), (('T25', 'T11'), 0), (('T24', 'T11'), 0), (('T26', 'T11'), 0), (('T13', 'T12'), 0), (('T18', 'T12'), 0), (('T19', 'T12'), 0), (('T8', 'T12'), 0), (('T9', 'T12'), 0), (('T6', 'T12'), 0), (('T7', 'T12'), 0), (('T4', 'T12'), 0), (('T5', 'T12'), 0), (('T2', 'T12'), 0), (('T3', 'T12'), 0), (('T1', 'T12'), 0), (('T21', 'T12'), 0), (('T20', 'T12'), 0), (('T23', 'T12'), 0), (('T22', 'T12'), 0), (('T25', 'T12'), 0), (('T24', 'T12'), 0), (('T26', 'T12'), 0), (('T18', 'T13'), 0), (('T19', 'T13'), 0), (('T8', 'T13'), 0), (('T9', 'T13'), 0), (('T6', 'T13'), 0), (('T7', 'T13'), 0), (('T4', 'T13'), 0), (('T5', 'T13'), 0), (('T2', 'T13'), 0), (('T3', 'T13'), 0), (('T1', 'T13'), 0), (('T21', 'T13'), 0), (('T20', 'T13'), 0), (('T23', 'T13'), 0), (('T22', 'T13'), 0), (('T25', 'T13'), 0), (('T24', 'T13'), 0), (('T26', 'T13'), 0), (('T19', 'T18'), 35, 6, [18, 26, 27], 1), (('T8', 'T18'), 0), (('T9', 'T18'), 0), (('T6', 'T18'), 0), (('T7', 'T18'), 0), (('T4', 'T18'), 0), (('T5', 'T18'), 0), (('T2', 'T18'), 0), (('T3', 'T18'), 0), (('T1', 'T18'), 0), (('T21', 'T18'), 55, 6, [18, 26, 27, 28], 0), (('T20', 'T18'), 35, 6, [18, 26, 27], 0), (('T23', 'T18'), 0), (('T22', 'T18'), 0), (('T25', 'T18'), 0), (('T24', 'T18'), 0), (('T26', 'T18'), 0), (('T8', 'T19'), 0), (('T9', 'T19'), 0), (('T6', 'T19'), 0), (('T7', 'T19'), 0), (('T4', 'T19'), 0), (('T5', 'T19'), 0), (('T2', 'T19'), 0), (('T3', 'T19'), 0), (('T1', 'T19'), 0), (('T21', 'T19'), 55, 6, [18, 26, 27, 28], 0), (('T20', 'T19'), 0, 6, [18], 0), (('T23', 'T19'), 0), (('T22', 'T19'), 0), (('T25', 'T19'), 0), (('T24', 'T19'), 0), (('T26', 'T19'), 0), (('T9', 'T8'), 0, 2, [], 0), (('T6', 'T8'), 0, 2, [], 1), (('T7', 'T8'), 0, 2, [], 0), (('T4', 'T8'), 0), (('T5', 'T8'), 0), (('T2', 'T8'), 0), (('T3', 'T8'), 0), (('T1', 'T8'), 0), (('T21', 'T8'), 0), (('T20', 'T8'), 0), (('T23', 'T8'), 0), (('T22', 'T8'), 0), (('T25', 'T8'), 0), (('T24', 'T8'), 0), (('T26', 'T8'), 0), (('T6', 'T9'), 0, 2, [], 1), (('T7', 'T9'), 0, 2, [], 0), (('T4', 'T9'), 0), (('T5', 'T9'), 0), (('T2', 'T9'), 0), (('T3', 'T9'), 0), (('T1', 'T9'), 0), (('T21', 'T9'), 0), (('T20', 'T9'), 0), (('T23', 'T9'), 0), (('T22', 'T9'), 0), (('T25', 'T9'), 0), (('T24', 'T9'), 0), (('T26', 'T9'), 0), (('T7', 'T6'), 0, 2, [], 1), (('T4', 'T6'), 0), (('T5', 'T6'), 0), (('T2', 'T6'), 0), (('T3', 'T6'), 0), (('T1', 'T6'), 0), (('T21', 'T6'), 0), (('T20', 'T6'), 0), (('T23', 'T6'), 0), (('T22', 'T6'), 0), (('T25', 'T6'), 0), (('T24', 'T6'), 0), (('T26', 'T6'), 0), (('T4', 'T7'), 0), (('T5', 'T7'), 0), (('T2', 'T7'), 0), (('T3', 'T7'), 0), (('T1', 'T7'), 0), (('T21', 'T7'), 0), (('T20', 'T7'), 0), (('T23', 'T7'), 0), (('T22', 'T7'), 0), (('T25', 'T7'), 0), (('T24', 'T7'), 0), (('T26', 'T7'), 0), (('T5', 'T4'), 24, 1, [2, 11, 12], 0), (('T2', 'T4'), 0), (('T3', 'T4'), 24, 1, [2, 11, 12], 1), (('T1', 'T4'), 0), (('T21', 'T4'), 0), (('T20', 'T4'), 0), (('T23', 'T4'), 0), (('T22', 'T4'), 0), (('T25', 'T4'), 0), (('T24', 'T4'), 0), (('T26', 'T4'), 0), (('T2', 'T5'), 0), (('T3', 'T5'), 24, 1, [2, 11, 12], 1), (('T1', 'T5'), 0), (('T21', 'T5'), 0), (('T20', 'T5'), 0), (('T23', 'T5'), 0), (('T22', 'T5'), 0), (('T25', 'T5'), 0), (('T24', 'T5'), 0), (('T26', 'T5'), 0), (('T3', 'T2'), 0), (('T1', 'T2'), 15, 0, [18, 1, 27], 1), (('T21', 'T2'), 0), (('T20', 'T2'), 0), (('T23', 'T2'), 0), (('T22', 'T2'), 0), (('T25', 'T2'), 0), (('T24', 'T2'), 0), (('T26', 'T2'), 0), (('T1', 'T3'), 0), (('T21', 'T3'), 0), (('T20', 'T3'), 0), (('T23', 'T3'), 0), (('T22', 'T3'), 0), (('T25', 'T3'), 0), (('T24', 'T3'), 0), (('T26', 'T3'), 0), (('T21', 'T1'), 0), (('T20', 'T1'), 0), (('T23', 'T1'), 0), (('T22', 'T1'), 0), (('T25', 'T1'), 0), (('T24', 'T1'), 0), (('T26', 'T1'), 0), (('T20', 'T21'), 35, 6, [18, 1, 27, 28], 0), (('T23', 'T21'), 0), (('T22', 'T21'), 0), (('T25', 'T21'), 0), (('T24', 'T21'), 0), (('T26', 'T21'), 0), (('T23', 'T20'), 0), (('T22', 'T20'), 0), (('T25', 'T20'), 0), (('T24', 'T20'), 0), (('T26', 'T20'), 0), (('T22', 'T23'), 64, 7, [11, 12, 1, 24, 27, 28], 1), (('T25', 'T23'), 49, 7, [11, 12, 19, 26, 28], 0), (('T24', 'T23'), 4, 7, [11, 12], 0), (('T26', 'T23'), 49, 7, [11, 12, 26, 28], 0), (('T25', 'T22'), 84, 7, [11, 12, 19, 26, 25, 27, 28], 0), (('T24', 'T22'), 84, 7, [11, 12, 26, 25, 27, 28], 1), (('T26', 'T22'), 84, 7, [11, 12, 26, 25, 27, 28], 0), (('T24', 'T25'), 29, 7, [11, 12, 19, 1, 28], 0), (('T26', 'T25'), 4, 7, [11, 12, 19], 1), (('T26', 'T24'), 49, 7, [11, 12, 26, 28], 0)], 'HPRD50_d27': [(('T4', 'T6'), 0), (('T5', 'T6'), 0, 1, [], 0), (('T2', 'T6'), 0), (('T3', 'T6'), 0), (('T1', 'T6'), 0), (('T5', 'T4'), 0), (('T2', 'T4'), 5, 0, [19, 1], 1), (('T3', 'T4'), 0, 0, [], 0), (('T1', 'T4'), 0, 0, [19], 1), (('T2', 'T5'), 0), (('T3', 'T5'), 0), (('T1', 'T5'), 0), (('T3', 'T2'), 25, 0, [19, 26], 1), (('T1', 'T2'), 0, 0, [18], 0), (('T1', 'T3'), 0, 0, [19], 1)], 'HPRD50_d20': [(('T9', 'T8'), 0), (('T4', 'T8'), 0), (('T5', 'T8'), 0), (('T2', 'T8'), 0), (('T3', 'T8'), 0), (('T1', 'T8'), 0), (('T10', 'T8'), 0), (('T11', 'T8'), 0), (('T6', 'T9'), 0), (('T7', 'T9'), 0), (('T4', 'T9'), 0), (('T5', 'T9'), 0), (('T2', 'T9'), 0), (('T3', 'T9'), 0), (('T1', 'T9'), 0), (('T10', 'T9'), 3, 3, [12, 19], 0), (('T11', 'T9'), 3, 3, [12, 19], 1), (('T4', 'T6'), 0), (('T5', 'T6'), 0), (('T2', 'T6'), 0), (('T3', 'T6'), 0), (('T1', 'T6'), 0), (('T10', 'T6'), 0), (('T11', 'T6'), 0), (('T4', 'T7'), 0), (('T5', 'T7'), 0), (('T2', 'T7'), 0), (('T3', 'T7'), 0), (('T1', 'T7'), 0), (('T10', 'T7'), 0), (('T11', 'T7'), 0), (('T5', 'T4'), 20, 1, [2], 0), (('T2', 'T4'), 0), (('T3', 'T4'), 0), (('T1', 'T4'), 0), (('T10', 'T4'), 0), (('T11', 'T4'), 0), (('T2', 'T5'), 0), (('T3', 'T5'), 0), (('T1', 'T5'), 0), (('T10', 'T5'), 0), (('T11', 'T5'), 0), (('T3', 'T2'), 35, 0, [19, 26, 27], 1), (('T1', 'T2'), 20, 0, [18, 28], 1), (('T10', 'T2'), 0), (('T11', 'T2'), 0), (('T1', 'T3'), 35, 0, [19, 1, 27, 28], 1), (('T10', 'T3'), 0), (('T11', 'T3'), 0), (('T10', 'T1'), 0), (('T11', 'T1'), 0), (('T11', 'T10'), 3, 3, [12, 18], 1)], 'HPRD50_d21': [(('T6', 'T8'), 0), (('T7', 'T8'), 30, 2, [27, 28], 0), (('T4', 'T8'), 0), (('T5', 'T8'), 0), (('T2', 'T8'), 0), (('T3', 'T8'), 0), (('T1', 'T8'), 0), (('T7', 'T6'), 0), (('T4', 'T6'), 28, 1, [12, 1, 28], 0), (('T5', 'T6'), 28, 1, [12, 1, 28], 0), (('T2', 'T6'), 0), (('T3', 'T6'), 0), (('T1', 'T6'), 0), (('T4', 'T7'), 0), (('T5', 'T7'), 0), (('T2', 'T7'), 0), (('T3', 'T7'), 0), (('T1', 'T7'), 0), (('T5', 'T4'), 3, 1, [12], 0), (('T2', 'T4'), 0), (('T3', 'T4'), 0), (('T1', 'T4'), 0), (('T2', 'T5'), 0), (('T3', 'T5'), 0), (('T1', 'T5'), 0), (('T3', 'T2'), 3, 0, [12, 19], 0), (('T1', 'T2'), 3, 0, [12, 18], 1), (('T1', 'T3'), 3, 0, [12, 19], 1)], 'HPRD50_d22': [(('T15', 'T14'), 94, 3, [2, 11, 12, 18, 26, 25, 28], 0), (('T16', 'T14'), 0), (('T17', 'T14'), 0), (('T10', 'T14'), 0), (('T11', 'T14'), 0), (('T12', 'T14'), 84, 3, [2, 11, 12, 18, 1, 24, 27, 28], 1), (('T13', 'T14'), 84, 3, [2, 11, 12, 18, 1, 24, 27, 28], 1), (('T18', 'T14'), 0), (('T19', 'T14'), 0), (('T8', 'T14'), 0), (('T9', 'T14'), 0), (('T6', 'T14'), 0), (('T7', 'T14'), 0), (('T4', 'T14'), 0), (('T5', 'T14'), 0), (('T2', 'T14'), 0), (('T3', 'T14'), 0), (('T1', 'T14'), 0), (('T21', 'T14'), 0), (('T20', 'T14'), 0), (('T23', 'T14'), 0), (('T22', 'T14'), 0), (('T25', 'T14'), 0), (('T24', 'T14'), 0), (('T26', 'T14'), 0), (('T16', 'T15'), 0), (('T17', 'T15'), 0), (('T10', 'T15'), 0), (('T11', 'T15'), 0), (('T12', 'T15'), 84, 3, [2, 11, 12, 18, 1, 24, 27, 28], 1), (('T13', 'T15'), 84, 3, [2, 11, 12, 18, 1, 24, 27, 28], 1), (('T18', 'T15'), 0), (('T19', 'T15'), 0), (('T8', 'T15'), 0), (('T9', 'T15'), 0), (('T6', 'T15'), 0), (('T7', 'T15'), 0), (('T4', 'T15'), 0), (('T5', 'T15'), 0), (('T2', 'T15'), 0), (('T3', 'T15'), 0), (('T1', 'T15'), 0), (('T21', 'T15'), 0), (('T20', 'T15'), 0), (('T23', 'T15'), 0), (('T22', 'T15'), 0), (('T25', 'T15'), 0), (('T24', 'T15'), 0), (('T26', 'T15'), 0), (('T17', 'T16'), 0, 4, [18], 0), (('T10', 'T16'), 0), (('T11', 'T16'), 0), (('T12', 'T16'), 0), (('T13', 'T16'), 0), (('T18', 'T16'), 0), (('T19', 'T16'), 0), (('T8', 'T16'), 0), (('T9', 'T16'), 0), (('T6', 'T16'), 0), (('T7', 'T16'), 0), (('T4', 'T16'), 0), (('T5', 'T16'), 0), (('T2', 'T16'), 0), (('T3', 'T16'), 0), (('T1', 'T16'), 0), (('T21', 'T16'), 0), (('T20', 'T16'), 0), (('T23', 'T16'), 0), (('T22', 'T16'), 0), (('T25', 'T16'), 0), (('T24', 'T16'), 0), (('T26', 'T16'), 0), (('T10', 'T17'), 0), (('T11', 'T17'), 0), (('T12', 'T17'), 0), (('T13', 'T17'), 0), (('T18', 'T17'), 0), (('T19', 'T17'), 0), (('T8', 'T17'), 0), (('T9', 'T17'), 0), (('T6', 'T17'), 0), (('T7', 'T17'), 0), (('T4', 'T17'), 0), (('T5', 'T17'), 0), (('T2', 'T17'), 0), (('T3', 'T17'), 0), (('T1', 'T17'), 0), (('T21', 'T17'), 0), (('T20', 'T17'), 0), (('T23', 'T17'), 0), (('T22', 'T17'), 0), (('T25', 'T17'), 0), (('T24', 'T17'), 0), (('T26', 'T17'), 0), (('T11', 'T10'), 55, 2, [18, 26, 27, 28], 1), (('T12', 'T10'), 0), (('T13', 'T10'), 0), (('T18', 'T10'), 0), (('T19', 'T10'), 0), (('T8', 'T10'), 0), (('T9', 'T10'), 0), (('T6', 'T10'), 0), (('T7', 'T10'), 0), (('T4', 'T10'), 0), (('T5', 'T10'), 0), (('T2', 'T10'), 0), (('T3', 'T10'), 0), (('T1', 'T10'), 0), (('T21', 'T10'), 0), (('T20', 'T10'), 0), (('T23', 'T10'), 0), (('T22', 'T10'), 0), (('T25', 'T10'), 0), (('T24', 'T10'), 0), (('T26', 'T10'), 0), (('T12', 'T11'), 0), (('T13', 'T11'), 0), (('T18', 'T11'), 0), (('T19', 'T11'), 0), (('T8', 'T11'), 0), (('T9', 'T11'), 0), (('T6', 'T11'), 0), (('T7', 'T11'), 0), (('T4', 'T11'), 0), (('T5', 'T11'), 0), (('T2', 'T11'), 0), (('T3', 'T11'), 0), (('T1', 'T11'), 0), (('T21', 'T11'), 0), (('T20', 'T11'), 0), (('T23', 'T11'), 0), (('T22', 'T11'), 0), (('T25', 'T11'), 0), (('T24', 'T11'), 0), (('T26', 'T11'), 0), (('T13', 'T12'), 24, 3, [2, 11, 12, 18], 0), (('T18', 'T12'), 0), (('T19', 'T12'), 0), (('T8', 'T12'), 0), (('T9', 'T12'), 0), (('T6', 'T12'), 0), (('T7', 'T12'), 0), (('T4', 'T12'), 0), (('T5', 'T12'), 0), (('T2', 'T12'), 0), (('T3', 'T12'), 0), (('T1', 'T12'), 0), (('T21', 'T12'), 0), (('T20', 'T12'), 0), (('T23', 'T12'), 0), (('T22', 'T12'), 0), (('T25', 'T12'), 0), (('T24', 'T12'), 0), (('T26', 'T12'), 0), (('T18', 'T13'), 0), (('T19', 'T13'), 0), (('T8', 'T13'), 0), (('T9', 'T13'), 0), (('T6', 'T13'), 0), (('T7', 'T13'), 0), (('T4', 'T13'), 0), (('T5', 'T13'), 0), (('T2', 'T13'), 0), (('T3', 'T13'), 0), (('T1', 'T13'), 0), (('T21', 'T13'), 0), (('T20', 'T13'), 0), (('T23', 'T13'), 0), (('T22', 'T13'), 0), (('T25', 'T13'), 0), (('T24', 'T13'), 0), (('T26', 'T13'), 0), (('T19', 'T18'), 29, 5, [11, 12, 18, 26], 0), (('T8', 'T18'), 0), (('T9', 'T18'), 0), (('T6', 'T18'), 0), (('T7', 'T18'), 0), (('T4', 'T18'), 0), (('T5', 'T18'), 0), (('T2', 'T18'), 0), (('T3', 'T18'), 0), (('T1', 'T18'), 0), (('T21', 'T18'), 0), (('T20', 'T18'), 29, 5, [11, 12, 18, 26], 0), (('T23', 'T18'), 0), (('T22', 'T18'), 0), (('T25', 'T18'), 0), (('T24', 'T18'), 0), (('T26', 'T18'), 0), (('T8', 'T19'), 0), (('T9', 'T19'), 0), (('T6', 'T19'), 0), (('T7', 'T19'), 0), (('T4', 'T19'), 0), (('T5', 'T19'), 0), (('T2', 'T19'), 0), (('T3', 'T19'), 0), (('T1', 'T19'), 0), (('T21', 'T19'), 0), (('T20', 'T19'), 4, 5, [11, 12, 18], 0), (('T23', 'T19'), 0), (('T22', 'T19'), 0), (('T25', 'T19'), 0), (('T24', 'T19'), 0), (('T26', 'T19'), 0), (('T9', 'T8'), 0, 1, [], 0), (('T6', 'T8'), 15, 1, [1, 27], 1), (('T7', 'T8'), 0, 1, [], 0), (('T4', 'T8'), 0), (('T5', 'T8'), 10, 1, [27], 1), (('T2', 'T8'), 0), (('T3', 'T8'), 0), (('T1', 'T8'), 0), (('T21', 'T8'), 0), (('T20', 'T8'), 0), (('T23', 'T8'), 0), (('T22', 'T8'), 0), (('T25', 'T8'), 0), (('T24', 'T8'), 0), (('T26', 'T8'), 0), (('T6', 'T9'), 15, 1, [1, 27], 1), (('T7', 'T9'), 0, 1, [], 0), (('T4', 'T9'), 0), (('T5', 'T9'), 10, 1, [27], 1), (('T2', 'T9'), 0), (('T3', 'T9'), 0), (('T1', 'T9'), 0), (('T21', 'T9'), 0), (('T20', 'T9'), 0), (('T23', 'T9'), 0), (('T22', 'T9'), 0), (('T25', 'T9'), 0), (('T24', 'T9'), 0), (('T26', 'T9'), 0), (('T7', 'T6'), 35, 1, [26, 27], 1), (('T4', 'T6'), 0), (('T5', 'T6'), 0, 1, [], 0), (('T2', 'T6'), 0), (('T3', 'T6'), 0), (('T1', 'T6'), 0), (('T21', 'T6'), 0), (('T20', 'T6'), 0), (('T23', 'T6'), 0), (('T22', 'T6'), 0), (('T25', 'T6'), 0), (('T24', 'T6'), 0), (('T26', 'T6'), 0), (('T4', 'T7'), 0), (('T5', 'T7'), 10, 1, [27], 1), (('T2', 'T7'), 0), (('T3', 'T7'), 0), (('T1', 'T7'), 0), (('T21', 'T7'), 0), (('T20', 'T7'), 0), (('T23', 'T7'), 0), (('T22', 'T7'), 0), (('T25', 'T7'), 0), (('T24', 'T7'), 0), (('T26', 'T7'), 0), (('T5', 'T4'), 0), (('T2', 'T4'), 10, 0, [19, 27], 1), (('T3', 'T4'), 0, 0, [], 0), (('T1', 'T4'), 10, 0, [19, 27], 1), (('T21', 'T4'), 0), (('T20', 'T4'), 0), (('T23', 'T4'), 0), (('T22', 'T4'), 0), (('T25', 'T4'), 0), (('T24', 'T4'), 0), (('T26', 'T4'), 0), (('T2', 'T5'), 0), (('T3', 'T5'), 0), (('T1', 'T5'), 0), (('T21', 'T5'), 0), (('T20', 'T5'), 0), (('T23', 'T5'), 0), (('T22', 'T5'), 0), (('T25', 'T5'), 0), (('T24', 'T5'), 0), (('T26', 'T5'), 0), (('T3', 'T2'), 10, 0, [19, 27], 1), (('T1', 'T2'), 0, 0, [18], 0), (('T21', 'T2'), 0), (('T20', 'T2'), 0), (('T23', 'T2'), 0), (('T22', 'T2'), 0), (('T25', 'T2'), 0), (('T24', 'T2'), 0), (('T26', 'T2'), 0), (('T1', 'T3'), 10, 0, [19, 27], 1), (('T21', 'T3'), 0), (('T20', 'T3'), 0), (('T23', 'T3'), 0), (('T22', 'T3'), 0), (('T25', 'T3'), 0), (('T24', 'T3'), 0), (('T26', 'T3'), 0), (('T21', 'T1'), 0), (('T20', 'T1'), 0), (('T23', 'T1'), 0), (('T22', 'T1'), 0), (('T25', 'T1'), 0), (('T24', 'T1'), 0), (('T26', 'T1'), 0), (('T20', 'T21'), 0), (('T23', 'T21'), 81, 6, [11, 19, 26, 25, 27, 28], 1), (('T22', 'T21'), 81, 6, [11, 19, 26, 25, 27, 28], 1), (('T25', 'T21'), 0), (('T24', 'T21'), 0), (('T26', 'T21'), 0), (('T23', 'T20'), 0), (('T22', 'T20'), 0), (('T25', 'T20'), 0), (('T24', 'T20'), 0), (('T26', 'T20'), 0), (('T22', 'T23'), 1, 6, [11], 0), (('T25', 'T23'), 0), (('T24', 'T23'), 0), (('T26', 'T23'), 0), (('T25', 'T22'), 0), (('T24', 'T22'), 0), (('T26', 'T22'), 0), (('T24', 'T25'), 25, 7, [19, 1, 28], 1), (('T26', 'T25'), 0, 7, [19], 0), (('T26', 'T24'), 45, 7, [26, 28], 1)], 'HPRD50_d23': [(('T9', 'T8'), 0), (('T2', 'T8'), 0), (('T1', 'T8'), 0), (('T14', 'T8'), 0), (('T10', 'T8'), 0), (('T11', 'T8'), 0), (('T12', 'T8'), 0), (('T13', 'T8'), 0), (('T6', 'T9'), 0), (('T7', 'T9'), 0), (('T4', 'T9'), 0), (('T5', 'T9'), 0), (('T2', 'T9'), 0), (('T3', 'T9'), 0), (('T1', 'T9'), 0), (('T14', 'T9'), 0), (('T10', 'T9'), 33, 2, [12, 14, 19, 26, 27, 28], 0), (('T11', 'T9'), 33, 2, [12, 14, 19, 26, 27, 28], 0), (('T12', 'T9'), 33, 2, [12, 14, 19, 26, 27, 28], 0), (('T13', 'T9'), 0), (('T7', 'T6'), 14, 1, [11, 12, 14, 26, 27], 0), (('T2', 'T6'), 0), (('T1', 'T6'), 0), (('T14', 'T6'), 0), (('T10', 'T6'), 0), (('T11', 'T6'), 0), (('T12', 'T6'), 0), (('T13', 'T6'), 0), (('T2', 'T7'), 0), (('T1', 'T7'), 0), (('T14', 'T7'), 0), (('T10', 'T7'), 0), (('T11', 'T7'), 0), (('T12', 'T7'), 0), (('T13', 'T7'), 0), (('T2', 'T4'), 0), (('T1', 'T4'), 0), (('T14', 'T4'), 0), (('T10', 'T4'), 0), (('T11', 'T4'), 0), (('T12', 'T4'), 0), (('T13', 'T4'), 0), (('T2', 'T5'), 0), (('T1', 'T5'), 0), (('T14', 'T5'), 0), (('T10', 'T5'), 0), (('T11', 'T5'), 0), (('T12', 'T5'), 0), (('T13', 'T5'), 0), (('T3', 'T2'), 0), (('T1', 'T2'), 0, 0, [18], 1), (('T14', 'T2'), 0), (('T10', 'T2'), 0), (('T11', 'T2'), 0), (('T12', 'T2'), 0), (('T13', 'T2'), 0), (('T1', 'T3'), 0), (('T14', 'T3'), 0), (('T10', 'T3'), 0), (('T11', 'T3'), 0), (('T12', 'T3'), 0), (('T13', 'T3'), 0), (('T14', 'T1'), 0), (('T10', 'T1'), 0), (('T11', 'T1'), 0), (('T12', 'T1'), 0), (('T13', 'T1'), 0), (('T10', 'T14'), 0), (('T11', 'T14'), 0), (('T12', 'T14'), 0), (('T13', 'T14'), 3, 3, [12, 18], 1), (('T13', 'T10'), 0), (('T13', 'T11'), 0), (('T13', 'T12'), 0)], 'HPRD50_d19': [(('T6', 'T8'), 0), (('T7', 'T8'), 0), (('T4', 'T8'), 0), (('T5', 'T8'), 0), (('T2', 'T8'), 0), (('T3', 'T8'), 0), (('T1', 'T8'), 0), (('T10', 'T8'), 24, 2, [11, 12, 14, 19, 26, 28], 0), (('T11', 'T8'), 24, 2, [11, 12, 14, 19, 26, 28], 0), (('T6', 'T9'), 0), (('T7', 'T9'), 0), (('T4', 'T9'), 0), (('T5', 'T9'), 0), (('T2', 'T9'), 0), (('T3', 'T9'), 0), (('T1', 'T9'), 0), (('T10', 'T9'), 24, 2, [11, 12, 14, 19, 26, 28], 0), (('T11', 'T9'), 24, 2, [11, 12, 14, 19, 26, 28], 0), (('T7', 'T6'), 1, 1, [11], 0), (('T4', 'T6'), 0), (('T5', 'T6'), 1, 1, [11], 0), (('T2', 'T6'), 0), (('T3', 'T6'), 0), (('T1', 'T6'), 0), (('T10', 'T6'), 0), (('T11', 'T6'), 0), (('T4', 'T7'), 0), (('T5', 'T7'), 1, 1, [11], 0), (('T2', 'T7'), 0), (('T3', 'T7'), 0), (('T1', 'T7'), 0), (('T10', 'T7'), 0), (('T11', 'T7'), 0), (('T5', 'T4'), 0), (('T2', 'T4'), 6, 0, [11, 19, 1], 1), (('T3', 'T4'), 1, 0, [11], 1), (('T1', 'T4'), 1, 0, [11, 19], 1), (('T10', 'T4'), 0), (('T11', 'T4'), 0), (('T2', 'T5'), 0), (('T3', 'T5'), 0), (('T1', 'T5'), 0), (('T10', 'T5'), 0), (('T11', 'T5'), 0), (('T3', 'T2'), 1, 0, [11, 19], 0), (('T1', 'T2'), 1, 0, [11, 18], 0), (('T10', 'T2'), 0), (('T11', 'T2'), 0), (('T1', 'T3'), 1, 0, [11, 19], 0), (('T10', 'T3'), 0), (('T11', 'T3'), 0), (('T10', 'T1'), 0), (('T11', 'T1'), 0)], 'HPRD50_d39': [(('T5', 'T4'), 30, 1, [2, 27], 0), (('T2', 'T4'), 0), (('T3', 'T4'), 0), (('T1', 'T4'), 0), (('T2', 'T5'), 0), (('T3', 'T5'), 0), (('T1', 'T5'), 0), (('T3', 'T2'), 25, 0, [19, 26], 1), (('T1', 'T2'), 0, 0, [18], 0), (('T1', 'T3'), 5, 0, [19, 1], 1)], 'HPRD50_d18': [(('T9', 'T8'), 0), (('T6', 'T8'), 0), (('T7', 'T8'), 20, 2, [28], 1), (('T4', 'T8'), 0), (('T5', 'T8'), 0), (('T2', 'T8'), 0), (('T3', 'T8'), 0), (('T1', 'T8'), 0), (('T10', 'T8'), 0), (('T11', 'T8'), 0), (('T12', 'T8'), 0), (('T13', 'T8'), 0), (('T6', 'T9'), 0), (('T7', 'T9'), 0), (('T4', 'T9'), 0), (('T5', 'T9'), 0), (('T2', 'T9'), 0), (('T3', 'T9'), 0), (('T1', 'T9'), 0), (('T10', 'T9'), 59, 3, [11, 12, 19, 26, 27, 28], 0), (('T11', 'T9'), 0), (('T12', 'T9'), 0), (('T13', 'T9'), 0), (('T7', 'T6'), 0), (('T4', 'T6'), 28, 1, [12, 1, 28], 0), (('T5', 'T6'), 28, 1, [12, 1, 28], 0), (('T2', 'T6'), 0), (('T3', 'T6'), 0), (('T1', 'T6'), 0), (('T10', 'T6'), 0), (('T11', 'T6'), 0), (('T12', 'T6'), 0), (('T13', 'T6'), 0), (('T4', 'T7'), 0), (('T5', 'T7'), 0), (('T2', 'T7'), 0), (('T3', 'T7'), 0), (('T1', 'T7'), 0), (('T10', 'T7'), 0), (('T11', 'T7'), 0), (('T12', 'T7'), 0), (('T13', 'T7'), 0), (('T5', 'T4'), 3, 1, [12], 0), (('T2', 'T4'), 0), (('T3', 'T4'), 0), (('T1', 'T4'), 0), (('T10', 'T4'), 0), (('T11', 'T4'), 0), (('T12', 'T4'), 0), (('T13', 'T4'), 0), (('T2', 'T5'), 0), (('T3', 'T5'), 0), (('T1', 'T5'), 0), (('T10', 'T5'), 0), (('T11', 'T5'), 0), (('T12', 'T5'), 0), (('T13', 'T5'), 0), (('T3', 'T2'), 25, 0, [19, 26], 1), (('T1', 'T2'), 0, 0, [18], 0), (('T10', 'T2'), 0), (('T11', 'T2'), 0), (('T12', 'T2'), 0), (('T13', 'T2'), 0), (('T1', 'T3'), 5, 0, [19, 1], 1), (('T10', 'T3'), 0), (('T11', 'T3'), 0), (('T12', 'T3'), 0), (('T13', 'T3'), 0), (('T10', 'T1'), 0), (('T11', 'T1'), 0), (('T12', 'T1'), 0), (('T13', 'T1'), 0), (('T11', 'T10'), 0), (('T12', 'T10'), 0), (('T13', 'T10'), 0), (('T13', 'T11'), 34, 4, [11, 12, 14, 18, 26, 27, 28], 0), (('T13', 'T12'), 34, 4, [11, 12, 14, 18, 26, 27, 28], 0)], 'HPRD50_d42': [(('T9', 'T8'), 0), (('T6', 'T8'), 0, 2, [], 0), (('T7', 'T8'), 0, 2, [], 0), (('T4', 'T8'), 0), (('T5', 'T8'), 20, 2, [28], 0), (('T2', 'T8'), 0), (('T3', 'T8'), 0), (('T1', 'T8'), 0), (('T14', 'T8'), 0), (('T10', 'T8'), 0), (('T11', 'T8'), 0), (('T12', 'T8'), 0), (('T13', 'T8'), 0), (('T6', 'T9'), 0), (('T7', 'T9'), 0), (('T4', 'T9'), 0), (('T5', 'T9'), 0), (('T2', 'T9'), 0), (('T3', 'T9'), 0), (('T1', 'T9'), 0), (('T14', 'T9'), 0), (('T10', 'T9'), 55, 3, [19, 26, 27, 28], 0), (('T11', 'T9'), 0), (('T12', 'T9'), 0), (('T13', 'T9'), 0), (('T7', 'T6'), 0, 2, [], 0), (('T4', 'T6'), 0), (('T5', 'T6'), 20, 2, [28], 1), (('T2', 'T6'), 0), (('T3', 'T6'), 0), (('T1', 'T6'), 0), (('T14', 'T6'), 0), (('T10', 'T6'), 0), (('T11', 'T6'), 0), (('T12', 'T6'), 0), (('T13', 'T6'), 0), (('T4', 'T7'), 0), (('T5', 'T7'), 20, 2, [28], 1), (('T2', 'T7'), 0), (('T3', 'T7'), 0), (('T1', 'T7'), 0), (('T14', 'T7'), 0), (('T10', 'T7'), 0), (('T11', 'T7'), 0), (('T12', 'T7'), 0), (('T13', 'T7'), 0), (('T5', 'T4'), 0), (('T2', 'T4'), 0), (('T3', 'T4'), 0, 1, [], 0), (('T1', 'T4'), 0), (('T14', 'T4'), 0), (('T10', 'T4'), 0), (('T11', 'T4'), 0), (('T12', 'T4'), 0), (('T13', 'T4'), 0), (('T2', 'T5'), 0), (('T3', 'T5'), 0), (('T1', 'T5'), 0), (('T14', 'T5'), 0), (('T10', 'T5'), 0), (('T11', 'T5'), 0), (('T12', 'T5'), 0), (('T13', 'T5'), 0), (('T3', 'T2'), 0), (('T1', 'T2'), 43, 0, [12, 15, 18, 1, 27, 28], 0), (('T14', 'T2'), 0), (('T10', 'T2'), 0), (('T11', 'T2'), 0), (('T12', 'T2'), 0), (('T13', 'T2'), 0), (('T1', 'T3'), 0), (('T14', 'T3'), 0), (('T10', 'T3'), 0), (('T11', 'T3'), 0), (('T12', 'T3'), 0), (('T13', 'T3'), 0), (('T14', 'T1'), 0), (('T10', 'T1'), 0), (('T11', 'T1'), 0), (('T12', 'T1'), 0), (('T13', 'T1'), 0), (('T10', 'T14'), 0), (('T11', 'T14'), 0), (('T12', 'T14'), 0), (('T13', 'T14'), 4, 5, [11, 12, 18], 0), (('T11', 'T10'), 0), (('T12', 'T10'), 0), (('T13', 'T10'), 0), (('T12', 'T11'), 56, 4, [11, 18, 26, 27, 28], 0), (('T13', 'T11'), 0), (('T13', 'T12'), 0)]}


    all_known_relations_dict = get_all_known_relations_dict(hprd50_paper_dict)      #    all_predicted_relations_dict = get_all_predicted_relations_dict(hprd50_paper_dict, max_sentences, all_known_relations_dict)
    all_possible_relations_dict = get_all_possible_relations_dict(hprd50_paper_dict)
#    all_predicted_relations_dict = get_all_predicted_relations_dict(hprd50_paper_dict, max_sentences, all_known_relations_dict)

    print 'all_predicted_relations_dict: ', all_predicted_relations_dict 
    print 'all_possible_relations_dict: ', all_possible_relations_dict
    print 'all_known_relations_dict: ', all_known_relations_dict

    print '-----------------------------------------------SVM---------------------------------------------------'
    sk_learn_testing.index(all_predicted_relations_dict, all_possible_relations_dict, all_known_relations_dict)
#    print '-----------------------------------------------heuristic---------------------------------------------'
    heuristic_scores.index(all_predicted_relations_dict, all_possible_relations_dict, all_known_relations_dict)
    
    
    
#    gold_standard, test_predictions, method_scored_list = convert_data_for_testing(all_predicted_relations_dict, all_possible_relations_dict, all_known_relations_dict)
#    gold_standard_pos, test_predictions_pos, gold_standard_neg, test_predictions_neg = get_positive_class_vectors(gold_standard, test_predictions, method_scored_list)
#   pos_rule_frequency, neg_rule_frequency = find_feature_frequency(X_feature_vectors, gold_standard)

    

if __name__=='__main__':
    hprd_index()
