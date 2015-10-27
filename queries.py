'''
Created on Mar 1, 2014

@author: Adam
'''

class Query(object):

    def __init__(self):
        self.syn_dict = None
        
        self.q1 = None
        self.q1_syns = None        
        self.q1_syns_noNewline = None
        self.q1_search_string = None
        self.q1_syn_dict = None
        
        self.q2 = None
        self.q2_syns = None
        self.q2_syns_noNewline = None
        self.q2_search_string = None
        self.q2_syn_dict = None
        
        self.all_queries_syns = None
        self.blinded_q1 = 'Protein1'
        self.blinded_q2 = 'Protein2'


    def make_syn_dict(self, query_dot_1or2):
        syn_dict = r'text_files\lowercase_synonyms.txt'
        raw_dict = open(syn_dict)
        syn_dict = {}

        key_and_values = None
        for line in raw_dict:
            term_and_syns = line.split("|")
            key = term_and_syns[0]
            values = term_and_syns[1]
            values = values.strip("\n")     #strips newline from values in dict
            if values[-1] == ";":
                values = values[0:-1]
            value_list = values.split(";")
            value_list.append(key)
            query_dot_1or2_lower= query_dot_1or2.lower()
            query_dot_1or2_lower_stripped = query_dot_1or2_lower.replace("-", " ")
            if query_dot_1or2_lower in value_list:
                key_and_values = value_list
                index_in_list = key_and_values.index(query_dot_1or2_lower)
                break
            elif query_dot_1or2_lower_stripped in value_list:
                key_and_values = value_list
                index_in_list = key_and_values.index(query_dot_1or2_lower_stripped)
                break
            
            
        q = query_dot_1or2
        
        if key_and_values:
 #           index_in_list = key_and_values.index(query_dot_1or2)
            query_for_key = key_and_values.pop(index_in_list)
            values_without_query = key_and_values
            syn_dict[query_for_key] = values_without_query

        return syn_dict


    def get_syns(self, q):
        syn_list = None
        q_lower = q.lower()
        q_lower_replace = q_lower.replace("-", " ")
        if q in self.syn_dict:
            syn_list = self.syn_dict[q]
        elif q_lower in self.syn_dict:
            syn_list = self.syn_dict[q_lower]
        elif q_lower_replace in self.syn_dict:  
            syn_list = self.syn_dict[q_lower_replace]
        return syn_list    

    def make_search_strings(self,q, syn_list):
        string = '"%s"' % q
        if syn_list:
            string += "+OR+"
            for i, syn in enumerate(syn_list):
                syn = syn.replace(" ", "+")
                if i < len(syn_list) - 1:
                    string += '"%s"+OR+' % syn
                else:
                    string += '"%s"' % syn

        return string

    def make_combined_syn_dict(self, query_q1_syn_dict, query_q2_syn_dict):
        z = dict(query_q1_syn_dict.items() + query_q2_syn_dict.items())
        return z
        
    def make_all_queries_syns(self):
        combined = [self.q1] + [self.q2]
        if self.q1_syns:
            combined = combined + self.q1_syns
        if self.q2_syns:
            combined = combined + self.q2_syns
        return combined

def make_query_object(q1, q2):
    query = Query()

    query.q1 = q1
    query.q2 = q2
    
    query.q1_syn_dict = query.make_syn_dict(query.q1)
    query.q2_syn_dict = query.make_syn_dict(query.q2) 
    
    query.syn_dict = query.make_combined_syn_dict(query.q1_syn_dict, query.q2_syn_dict)
    
    query.q1_syns = query.get_syns(query.q1)  
    query.q2_syns = query.get_syns(query.q2)
            
    query.q1_search_string = query.make_search_strings(query.q1, query.q1_syns)
    query.q2_search_string = query.make_search_strings(query.q2, query.q2_syns)
    
    query.all_queries_syns = query.make_all_queries_syns()

    return query

def main(q1, q2):

    queries = make_query_object(q1, q2)

    return queries