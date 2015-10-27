'''
Created on Aug 6, 2014

@author: Adam
'''
import xml.etree.ElementTree as ET
import queries
import itertools
import Score1
import Score2
import operator
from collections import OrderedDict
import numpy as np
from sklearn import metrics
import pylab as pl
from pyroc import *


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
        print self.xml_dict_value
        print 'a'
        removal_dict = {}
        for key in self.xml_dict_value[1][2]:
            first_char = int(self.xml_dict_value[1][2][key][0][0])
            offset = int(self.xml_dict_value[1][2][key][0][1])
            last_char = first_char + offset
            removal_dict[key] = (first_char, last_char)
        sorted_removal_dict = OrderedDict(sorted(removal_dict.items(), key=lambda t: t[0], reverse=True))
        for key in sorted_removal_dict:
            first_char = sorted_removal_dict[key][0]
            last_char = sorted_removal_dict[key][1]
            abstract = abstract[:first_char] + key + abstract[last_char:]
        self.abstract_replaced_names = abstract
        
        import sys
        sys.exit()
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
    ID_sentence_list = []
    for i, sentence in enumerate(paper_obj.abstract_sentences):
        if (query.q1 in sentence) and (query.q2 in sentence):
            ID_sent_position_tuple = (paper_obj.id, paper_obj.abstract_sentences[i], i)
            ID_sentence_list.append(ID_sent_position_tuple)
    return ID_sentence_list

def get_all_predicted_relations_dict(hprd50_paper_dict, max_sentences):
    all_predicted_relations_dict = {}  
    for key in hprd50_paper_dict: 
        found_relation_score_tuples_list = [] 
        for relation in hprd50_paper_dict[key].possible_relations_Tnumber:
            q1 = relation[0]
            q2 = relation[1]
            found_relation_score_tuple = ((q1, q2), 0)
            if q1 == q2:
                continue
            query = queries.main(q1,q2)
            ID_sentence_list = []
            ID_sentence_list = find_cooccurrence_sents(hprd50_paper_dict[key], query)
            print 'ID_sentence_list: ',ID_sentence_list
            if not ID_sentence_list:
                found_relation_score_tuple = ((q1, q2), 0)
                print found_relation_score_tuple
                found_relation_score_tuples_list.append(found_relation_score_tuple)
            if ID_sentence_list:
                sentences_with_score1 = Score1.rank_sentences(ID_sentence_list, query, max_sentences)
                sentences_with_score2 = Score2.main(sentences_with_score1, query)
                sorted_sentences_with_score2 = list(sorted(sentences_with_score2, key=operator.attrgetter('score'), reverse=True))
                #------------------------------ if sorted_sentences_with_score2:
                    #----------------- if len(sorted_sentences_with_score2) > 1:
                        #-------------------------------------------------- pass
                    #----------------------------------------------------- else:
                for sentence_object in sorted_sentences_with_score2:
                    found_relation_score_tuple = ((q1, q2), sentence_object.score, sentence_object.order_in_abstract)
                    print found_relation_score_tuple
                    found_relation_score_tuples_list.append(found_relation_score_tuple)
        all_predicted_relations_dict[key]=found_relation_score_tuples_list
    return all_predicted_relations_dict                      
                    
                
                #===============================================================
                # for scored_sentence in sorted_sentences_with_score2:
                #     found_relation_score_tuple = ((q1, q2), scored_sentence.score)
                #     if not found_relation_score_tuples_list:
                #         found_relation_score_tuples_list.append(found_relation_score_tuple)
                #     else:
                #         for score_tuple in found_relation_score_tuples_list:
                #             if (q1 and q2) not in score_tuple[0]:
                #                 found_relation_score_tuples_list.append(found_relation_score_tuple)
                #             elif (q1 and q2) in score_tuple[0] and scored_sentence.score > score_tuple[1]:
                #                 found_relation_score_tuples_list.remove(score_tuple)
                #                 found_relation_score_tuples_list.append(found_relation_score_tuple)
                #===============================================================
                                                 
            
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
    for key in all_possible_relations_dict:
        if key not in all_known_relations_dict:
            raise IndexError
        else:
            gold_standard = []
            for possible_relation in all_possible_relations_dict[key]:
                for known_relation in all_known_relations_dict[key]:
                    if (known_relation[0] in possible_relation) and (known_relation[1] in possible_relation):
                        gold_standard.append(1)                                        ####POSSIBLE ERROR HERE! MIGHT NOT PRESERVE ORDER
                    else:
                        gold_standard.append(0)
            gold_standard_dict[key] = gold_standard                                        ####POSSIBLE ERROR HERE! MIGHT NOT PRESERVE ORDER
                

    test_predictions_dict = {}
    for key in all_predicted_relations_dict:
        predict_list = []
        for relation in all_predicted_relations_dict[key]:
            if relation[1] != 0 :
                predict_list.append(relation[1])
            if relation[1] <= 0:
                predict_list.append(0)
        test_predictions_dict[key] = predict_list                        
            
    gold_standard = []
    test_predictions = []        
    for key in gold_standard_dict:
        if len(gold_standard_dict[key]) != len(test_predictions_dict[key]):
            print 'NOT same length!:'
            print 'KEY ', key
            print 'gold_standard_dict[key]: ', gold_standard_dict[key]
            print 'test_predictions_dict[key]: ', test_predictions_dict[key]
        if len(gold_standard_dict[key]) == len(test_predictions_dict[key]):
            gold_standard += gold_standard_dict[key]
            test_predictions += test_predictions_dict[key]
    print 'gold_standard: ',gold_standard
    print 'test_predictions: ', test_predictions
    
    
###using scikit-learn###
#    if any(element for element in gold_standard_dict[key])
    y_true = np.array(gold_standard)
    scores = np.array(test_predictions)
    precision, recall, thresholds = metrics.precision_recall_curve(y_true, scores)
    print 'precision: ', precision
    print 'recall: ', recall
    print 'thresholds: ', thresholds
    
    area = metrics.auc(recall, precision)
    print("Area Under Curve: %0.2f" % area)
    
    pl.clf()
    pl.plot(recall, precision, label='Precision-Recall curve')
    pl.xlabel('Recall')
    pl.ylabel('Precision')
    pl.ylim([0.0, 1.05])
    pl.xlim([0.0, 1.0])
    pl.title('Precision-Recall example: AUC=%0.2f' % area)
    pl.legend(loc="lower left")
    pl.show()

    
    fpr, tpr, thresholds = metrics.roc_curve(y_true, scores, pos_label=1)
    print 'fpr: ', fpr
    print 'tpr: ', tpr
    print 'thresholds: ', thresholds
    
    

### Using pyroc ###   
    zipped = zip(gold_standard, test_predictions)
    roc = ROCData(zipped)
    print roc.auc()
    roc.plot(title='ROC Curve')
    print roc.plot(title='ROC Curve')
        
def hprd_index():
    dir_entry = 'Data Sets\hprd50_bioc.xml'
    tree = ET.parse(dir_entry)
    root = tree.getroot()
    xml_dict = xml_to_dict(root)
    max_sentences = 10
    cutoff_score = 26
    hprd50_paper_dict = make_hprd50_papers(xml_dict)    
    print hprd50_paper_dict
    
    def slicedict(d, s):                                                             #Only for testing, should be deleted
       return {k:v for k,v in d.iteritems() if k in s}        #Only for testing, should be deleted
    hprd50_paper_dict = slicedict(hprd50_paper_dict, ['HPRD50_d7', 'HPRD50_d26'])              #Only for testing, should be deleted
 #   def slicedict(d, s, p):                                                             #Only for testing, should be deleted
 #       return {k:v for k,v in d.iteritems() if (k.startswith(s) or k.startswith(p))}        #Only for testing, should be deleted
 #   hprd50_paper_dict = slicedict(hprd50_paper_dict, 'HPRD50_d3', 'HPRD50_d30')              #Only for testing, should be deleted   

    for key in hprd50_paper_dict:
        print key
        print hprd50_paper_dict[key].abstract
        print ''
        print hprd50_paper_dict[key].abstract_replaced_names
        print ''
        print hprd50_paper_dict[key].abstract_sentences
        print ''
        
    import sys
    sys.exit()
    
    all_predicted_relations_dict = get_all_predicted_relations_dict(hprd50_paper_dict, max_sentences)
    all_possible_relations_dict = get_all_possible_relations_dict(hprd50_paper_dict)
    all_known_relations_dict = get_all_known_relations_dict(hprd50_paper_dict)

#    all_predicted_relations_dict =  {'HPRD50_d33': [(('T6', 'T4'), 32), (('T6', 'T5'), 32), (('T6', 'T2'), 0), (('T6', 'T3'), 0), (('T6', 'T1'), 0), (('T4', 'T5'), 17), (('T4', 'T2'), 0), (('T4', 'T3'), 0), (('T4', 'T1'), 0), (('T5', 'T2'), 0), (('T5', 'T3'), 0), (('T5', 'T1'), 0), (('T2', 'T3'), 0), (('T2', 'T1'), 0), (('T3', 'T1'), 0)]}
#    all_possible_relations_dict =  {'HPRD50_d33': [('T6', 'T4'), ('T6', 'T5'), ('T6', 'T2'), ('T6', 'T3'), ('T6', 'T1'), ('T4', 'T5'), ('T4', 'T2'), ('T4', 'T3'), ('T4', 'T1'), ('T5', 'T2'), ('T5', 'T3'), ('T5', 'T1'), ('T2', 'T3'), ('T2', 'T1'), ('T3', 'T1')]}
#    all_known_relations_dict =  {'HPRD50_d33': [('T5', 'T6'), ('T1', 'T2'), ('T1', 'T3'), ('T4', 'T6')]}
    
    print 'all_predicted_relations_dict: ', all_predicted_relations_dict
    print 'all_possible_relations_dict: ', all_possible_relations_dict
    print 'all_known_relations_dict: ', all_known_relations_dict
    
    convert_data_for_testing(all_predicted_relations_dict, all_possible_relations_dict, all_known_relations_dict)


if __name__=='__main__':
    hprd_index()