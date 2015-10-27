'''
Created on Mar 2, 2014

@author: Adam
'''
from nltk.tokenize import RegexpTokenizer
from nltk.tag import pos_tag
import nltk
import re
from collections import OrderedDict
#import SentenceSplitter

def make_list_of_syns(text_file):
    word_list = []
    f = open(text_file)
    for line in f:
        line = line.strip()
        word_list.append(line)
    f.close()
    return word_list


def find_syns(first_word, query):
    for word in query.q1_syns_noNewline:
        if word == first_word:
            return True
    for word in query.q2_syns_noNewline:
        if word == first_word:
            return True
    return False  

def remove_inside_parenthesis(sentence, query):
    #remove parenthesis
    if ('(' in sentence) or (')' in sentence):
        regEx = re.compile(r'([^\(]*)\([^\)]*\) *(.*)')
        m = regEx.match(sentence)
        while m:
            possible_new_sentence = m.group(1) + m.group(2)
#             if any(syn in possible_new_sentence for syn in query.all_queries_syns):
#                 sentence = m.group(1) + m.group(2)
#                 break
            if query.blinded_q1 in possible_new_sentence: #or any(syn in possible_new_sentence for syn in query.q1_syns):
                if query.blinded_q2 in possible_new_sentence: #or any(syn in possible_new_sentence for syn in query.q2_syns): 
                    sentence = m.group(1) + m.group(2)
                    break
            else:
                break
            m = regEx.match(sentence)
    return sentence

def blind_sentence(sentence, query):
    entity_list = query.all_queries_syns
    sorted_entity_list = sorted(entity_list, key=len, reverse=True)
    if sorted_entity_list:
        
        protein_counter = 0
        for entity in sorted_entity_list:
            search_term = '(?='+entity+')'
            entity_positions = [m.start() for m in re.finditer(search_term, sentence)]
            removal_dict = {}
            for entity_position in entity_positions:
                protein_counter = protein_counter + 1
                first_char = entity_position
                offset = len(entity)
                last_char = first_char + offset
                removal_dict_key = 'Protein'+str(protein_counter)
                removal_dict[removal_dict_key] = (first_char, last_char)
            sorted_removal_dict = OrderedDict(sorted(removal_dict.items(), key=lambda t: t[1][0], reverse=True))
            for key in sorted_removal_dict:
                first_char = sorted_removal_dict[key][0]
                last_char = sorted_removal_dict[key][1]
                sentence = sentence[:first_char] + key + sentence[last_char:]
        return sentence


def preprocess(sentence, query):

    removed_parenthesis = remove_inside_parenthesis(sentence, query)
    blinded = blind_sentence(removed_parenthesis, query)
    b1 = blinded.replace('(',' ')
    b2 = b1.replace(')',' ')
    
    return b2

def index(ID_sentence_position_list, query, max_sents):
    stim_words = make_list_of_syns("text_files\stim_words.txt")
    conclusive_words = make_list_of_syns("text_files\conclusive_words.txt")
    PPI_tests = make_list_of_syns("text_files\PPI_tests.txt")    
    relation_keywords = make_list_of_syns(r"text_files\relation_keywords.txt")
    verb_tags = ["VB","VBD","VBN","VBP","VBZ"]
    bad_words = ["not", "lack", "fail", "without", "Although"]
    
    tokenizer = RegexpTokenizer("\s+", gaps = True)
    list_of_sentences_with_scores = []
    
    for ID_sentence_position in ID_sentence_position_list:
        score = [0, []]
        
        ID = ID_sentence_position[0]
        original_sentence = ID_sentence_position[1]
        sentence = ID_sentence_position[1]
        position = ID_sentence_position[2]   
        
        sentence = preprocess(sentence, query)
     
        tokenized_sentence = tokenizer.tokenize(sentence)
        pos_sentence = nltk.pos_tag(tokenized_sentence)
        relation_words_indices = []
        
        for i, word_tup in enumerate(pos_sentence):
            if word_tup[1] in relation_keywords:
                relation_words_indices.append(i)

         #if first word is a verb. Uses POS sentence, ex. [('Using', 'VBG'), ('the', 'DT'), ('real-time', 'JJ'), ('reverse', 'NN')]
        if pos_sentence[0][1] in verb_tags: 
             score[0] +=20
             score[1].append(1)
             
        #if second word is a verb
        if pos_sentence[1][1] in verb_tags:
            score[0] +=20
            score[1].append(2)     
 
        # If exact query term in sentence
        if query.q1 in sentence:
            score[0] += 1
            score[1].append(3)
   
        if query.q2 in sentence:
            score[0] += 1
            score[1].append(4)

        #if first word in sent is q1/q2 
        if pos_sentence[0][0] == query.q1 or pos_sentence[0][0] == query.q2:
            score[0] += 10
            score[1].append(5)
            if any(word in sentence for word in stim_words):
                score[0] += 20
                score[1].append(6)

        #first word is a q1 syn
        if query.q1_syns:
            for syn in query.q1_syns:
                if pos_sentence[0][0] == syn:
                    score[0] += 10
                    score[1].append(7)
                    if any(word in sentence for word in stim_words):
                        score[0] += 20
                        score[1].append(8)
        
        #first word is a q2 syn        
        if query.q2_syns:
            for syn in query.q2_syns:
                if pos_sentence[0][0] == syn:
                    score[0] += 10
                    score[1].append(9)
                    if any(word in sentence for word in stim_words):
                        score[0] += 20
                        score[1].append(10)
    
        
        #if stim word in sentence
        if any(word in sentence for word in stim_words):
                score[0] += 1
                score[1].append(11)
        
        # If conclusive words (suggest, found, show)
        if any(word in sentence for word in conclusive_words):
            score[0] += 3
            score[1].append(12)

        # Not, lack, fail, without
        for word in sentence:
            if word in bad_words:
                score[0] -=3
                score[1].append(13) 

        # TEST SENTENCE LENGTH
        if len(tokenized_sentence) > 30:
            score[0] -= 25
            score[1].append(14)
 
        #If PPI molecular test in sentence
        if any(word in sentence for word in PPI_tests):
            score[0] += 5
            score[1].append(15)
        
        #query1-query2 or query1/query2 (Signifies a two protein complex)          
        if query.q1+"-"+query.q2 in sentence:
            score[0] += 0
            score[1].append(16)
             
        if query.q1+"/"+query.q2 in sentence:
            score[0] += 15
            score[1].append(17)
        
        #doesnt change score, here for calibration purposes
        if query.q1_syns or query.q2_syns:
            if query.q1_syns and query.q2_syns:
                score[1].append(18)
            else:
                score[1].append(19)
        
        sentence_remove_dashes = sentence.replace('-',' ') 
        sentence_remove_slashes = sentence_remove_dashes.replace('/',' ')
        sentence_remove_commas = sentence_remove_slashes.replace(',','')
        tokenized_no_dashes = tokenizer.tokenize(sentence_remove_commas) 
        pos_no_dashes = nltk.pos_tag(tokenized_no_dashes)
        
        q1_index = []
        q2_index = []                
        if query.q1 in tokenized_sentence and query.q2 in tokenized_sentence:        
            q1_index = tokenized_sentence.index(query.q1)
            q2_index = tokenized_sentence.index(query.q2) 
            possible_TO_q1 = q1_index + 1
            possible_TO_q2 = q2_index - 1
            if len(pos_sentence) > possible_TO_q1 and len(pos_sentence) > possible_TO_q2:
                if pos_sentence[possible_TO_q1][1] == 'TO':
                    if pos_sentence[possible_TO_q2][1] == 'TO':
                        score[0] += 15
                        score[1].append(20)
            elif len(pos_sentence) < possible_TO_q1 and len(pos_sentence) < possible_TO_q2:
                if pos_sentence[possible_TO_q1][1] == 'TO':
                    if pos_sentence[possible_TO_q2][1] == 'TO':
                        score[0] += 15
                        score[1].append(21)          
        elif query.blinded_q1 in tokenized_no_dashes and query.blinded_q2 in tokenized_no_dashes:        
            q1_index = tokenized_no_dashes.index(query.blinded_q1)
            q2_index = tokenized_no_dashes.index(query.blinded_q2)
            possible_TO_q1 = q1_index + 1
            possible_TO_q2 = q2_index - 1
            pos_sentence = pos_no_dashes
            if len(pos_sentence) > possible_TO_q1 and len(pos_sentence) > possible_TO_q2:
                if pos_sentence[possible_TO_q1][1] == 'TO':
                    if pos_sentence[possible_TO_q2][1] == 'TO':
                        score[0] += 15
                        score[1].append(22)
            elif len(pos_sentence) < possible_TO_q1 and len(pos_sentence) < possible_TO_q2:
                if pos_sentence[possible_TO_q1][1] == 'TO':
                    if pos_sentence[possible_TO_q2][1] == 'TO':
                        score[0] += 15
                        score[1].append(23)
        else:
            print 'problem detected in finding blinded proteins, see syntax module'
#            print query.blinded_q1
#            print query.blinded_q2
#            print query.q1
#            print query.q2
#            print 'TOKENIZED SENTENCE', tokenized_sentence
#           print 'tokenized sentence no dash/comma', tokenized_no_dashes
   
     
   
   
   
   
        
        if q1_index and q2_index:
            for i, word_tup in enumerate(pos_sentence):
            #prot - REL - prot2
                if any(word_tup[1] == verb_tag for verb_tag in verb_tags):
                    if i > q1_index and i < q2_index:
                        score[0] += 5
                        score[1].append(1)
                        if word_tup[0] in relation_keywords:
                            score[0] += 25
                            score[1].append(24)
                        break
                    elif i < q1_index and i > q2_index:
                        score[0] += 25
                        score[1].append(26)
                        if any(word_tup[0] == relation for relation in relation_keywords):
                            score[0] += 25
                            score[1].append(25)
                        break 
                        
        #REL - Prot - PREP - Prot2 (ex. binding of P1 to P2)
        if q1_index and q2_index:
            for i, word_tup in enumerate(pos_sentence):
                if word_tup[1] == 'IN' or word_tup[1] == 'TO':
                    if word_tup[1] > q1_index and word_tup[1] < q2_index:
                        if any(relation < q1_index for relation in relation_keywords):
                            score[0] += 35
                            score[1].append(26)
                            break
                            
             
        scored_sentence_tuple = (score, ID_sentence_position[0], sentence, ID_sentence_position[2])
        if scored_sentence_tuple[0][0] > -10 and len(tokenized_sentence)<70:
#        if scored_sentence_tuple[0] > 0 and len(tokenized_sentence)<30:
            list_of_sentences_with_scores.append(scored_sentence_tuple)
    
    
    sorted_list_of_sentences_with_scores = sorted(list_of_sentences_with_scores, reverse=True)
    pruned_sorted_list_of_sentences_with_scores = sorted_list_of_sentences_with_scores[0:max_sents]
    return pruned_sorted_list_of_sentences_with_scores



def main(ID_sentence_position_list, query, max_sents):
    
#    print ''
#    ID_sentence_position_list = [('24987058', 'association between mrna expression of the (seorigjsoirgb) following genes and cf was analyzed: androgen receptor (ar) and its related genes (app, fox family, trim 36, oct1, and acsl 3), stem cell (sc)-like molecules (klf4, c-myc, oct 3/4, and sox2), estrogen receptor (er), her2, psa, and crp', 5), ('24987058', 'we identified 10 prognostic factors for cancer-specific survival (css): oct1, trim36, sox2, and c-myc expression in cancer cells; ar, klf4, and er expression in stromal cells; and psa, gleason score, and extent of disease', 7)]
    
    sentences_with_scores = index(ID_sentence_position_list, query, max_sents)
    return sentences_with_scores















