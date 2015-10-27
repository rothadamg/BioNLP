import nltk
import requests

FREEBASE_API_KEY = 'AIzaSyD5sRlLwyHPzv2B1CddJPhKam6k_sGBFPY'

class FindNames(object):

  def __init__(self, text, freebase_api_key):
    self.text = text
    self.key = freebase_api_key
    self.sentences = [sent.strip() for sent in nltk.tokenize.sent_tokenize(self.text)[:]]
    self.freebase_search_url = 'https://www.googleapis.com/freebase/v1/search'
    self.freebase_reconsile_url = 'https://www.googleapis.com/freebase/v1/reconcile'

  def get_named_entities(self):
    '''
    Returns a list of the named entities found in the text
    '''
    named_entities = []
    for sent in self.sentences:
      split = nltk.word_tokenize(sent)
      tokens = nltk.pos_tag(split)
      nes = nltk.chunk.ne_chunk(tokens, binary=True)
      for tok in nes:
        if isinstance(tok, nltk.tree.Tree): #this is named entity!
          if tok.node == 'NE':
            named_entities.append(' '.join(key for key,val in tok.leaves()))
          else:
            print 'Not sure what else...' + tok.node

    #kind of hacky n^2 alg to remove substrings so we don't double search
    setne = list(set(named_entities))
    print named_entities
    print setne
    final_ne = []
    for entity in setne:
      solid = True
      for entity2 in setne:
        if entity != entity2:
          if entity2.find(entity) >= 0:
            #keep if entity isn't a sub string
            solid = False
      if solid:
        final_ne.append(entity)
    return final_ne

  def get_freebase_possibilities(self):
    '''
    Gets the results returned from freebase call.
    '''
    named_entities = self.get_named_entities()
    for ne in named_entities:
      params = {}
      params['query'] = ne
      params['key'] = self.key
      params['filter'] = '(any domain:/american_football)'
      options = requests.get(self.freebase_search_url, params=params).json().get("result", "")
      if options and options[0].get('score') > 10:
        print ne + ': ' + options[0].get('name') + ' ' + str(options[0].get('score'))


def main():
  '''
  Run some tests
  '''
  fn = FindNames(text, FREEBASE_API_KEY)
  fn.get_freebase_possibilities()

if __name__ == '__main__':
  text = 'The androgen receptor (AR) plays roles in prostate development and cancer (PCa). In response to androgens, the AR binds to androgen-response elements (AREs) to modulate gene transcription. The responses of such genes are dependent on the cellular milieu and on sequences around the AREs, which attract other transcription factors. Previously, bioinformatic analysis of 62 AR-occupied regions (ARORs) in PCa cells revealed enrichment for both AREs and a TTGGCAAATA-like motif. We undertook the present study to investigate the significance of the TTGGCAAATA-like motif.'
  
  main()
