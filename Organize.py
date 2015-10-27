'''
Created on Mar 19, 2014

@author: Adam
'''


def insert_syns(sent,q1,q1_syns,q2,q2_syns):        #Adds query in brackets [] if syn in sentence
 
    sentence2 = sent.replace("."," ").lower().split()
  
    if any(q1.lower() == word.lower() for word in sentence2):          # or any(q1.lower() == val for val in sentence2):
        new = "[" + q1 +"]"
        sent = sent.replace(q1, new)
    
    if any(q2.lower == word.lower for word in sentence2):          # or any(q2.lower() == val for val in sentence2):
        new = "[" + q2 +"]"
        sent = sent.replace(q2, new)
 
    if q1 not in sent:
        if q1_syns:
            for syn in q1_syns:
                if any(syn.lower() == word.lower() for word in sentence2):     #any(syn.lower() == word.lower() for word in sentence2)
                    syn_length = len(syn)
                    location_syn = sent.find(syn)
                    location_to_insert = location_syn + syn_length
                    sent = sent[:location_to_insert] + " [" + q1 + "]" + sent[location_to_insert:]
        
    if q2 not in sent:
        if q2_syns:
            for syn in q2_syns:
                if any(syn.lower() == word.lower() for word in sentence2):         # or any(syn.lower() == word.lower() for word in sentence2):
                    syn_length = len(syn)
                    location_syn = sent.find(syn)
                    location_to_insert = location_syn + syn_length
                
                    sent = sent[:location_to_insert] + " [" + q2 + "]" + sent[location_to_insert:]
                
            
            
            
#               sent = sent[:location_to_insert] + color.BOLD + " [" + q2 + "]" + color.END + sent[location_to_insert:]                
    

    return sent



