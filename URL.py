'''
Created on Mar 1, 2014

@author: Adam
'''
# from pyquery import PyQuery as pq
import generate_test_set
import PPI_cite_main
import urllib2
import xml.etree.ElementTree as ET
def get_xml(url):
    if url:
#        try:
        file = urllib2.urlopen(url)
        xml = file.read()
        file.close()
        print url
        return xml
    
#        except urllib2.URLError:
#            xml = None 
#            return xml

def make_search_url(base_URL, query, articles):
    max_papers = "&retmax=%d" % articles
    title_abstract_add = "[tiab]"
    search_url_add = "esearch.fcgi?db=pubmed&term=(%s)+AND+(%s)" % (query.q1_search_string, query.q2_search_string)
    url = base_URL + search_url_add + max_papers
#    print url
#    url = url.replace('"','')
#    print query.q1_search_string
#    print query.q2_search_string
#    print url
    return url

def get_ID_list(xml):
    try:
        root = ET.fromstring(xml)
        ID_List_ofElements = root.findall("./IdList/Id")
        ids = []
        for element in ID_List_ofElements:
            singleID_string = ET.tostring(element, method='text')
            singleID_string_stripped = singleID_string.replace("\n", "")
            ids.append(singleID_string_stripped)
    except AttributeError:
        ids = []
        print("No Papers with both queries were found on PubMed")
        PPI_cite_main.no_papers_with_queries("No Papers with both queries were found on PubMed", None, None, None, None, None)
        

    existing_papers = []  # Use this in the future to make database of existing IDs 
    papers_to_download = []
    for ind_id in ids:
        papers_to_download.append(ind_id)

    full_ID_List = {"existing_papers":existing_papers,
                                    "papers_to_download":papers_to_download}
    return full_ID_List

def make_fetch_url(base_URL, get_abstract_portion_URL, ids, articles):
    if ids["papers_to_download"]:
        max_papers = "&retmax=%d" % articles
        fetch_id_string = ",".join(ids["papers_to_download"])
        fetch_url_add = "efetch.fcgi?db=pubmed&id=%s" % fetch_id_string
        full_url = base_URL + fetch_url_add + get_abstract_portion_URL + max_papers
        return full_url
    else:
        max_papers = "&retmax=%d" % articles
        fetch_id_string = ",".join(ids["papers_to_download"])
        fetch_url_add = "efetch.fcgi?db=pubmed&id=%s" % fetch_id_string
        full_url = base_URL + fetch_url_add + get_abstract_portion_URL + max_papers
        return None


def get_info_from_docs_xml(xml, ids):
       
    root = ET.fromstring(xml)
    def findall(whattofind):  # closure function -- http://en.wikipedia.org/wiki/Closure_%28computer_programming%29
        listofelements = []
        for b in root.findall(whattofind):
            
            c = b.text
            if isinstance(c, unicode):
                c = c.encode('ascii', 'ignore')  # Note: ignores unicode, does not keep unicode letters
            listofelements.append(c)
        return listofelements
    
    id_list = findall(".//ArticleId[@IdType='pubmed']")
    if id_list > 0:
        print str(len(id_list)) + " papers with co-occurrence found"
    title_list = findall(".//ArticleTitle")
    abstract_list = findall(".//AbstractText")
    authors_list = []

    return_dict = {"fetched_id_list" : id_list, "title_list":title_list, "abstract_list":abstract_list, "authors_list": authors_list}
    return return_dict

def get_info_from_PubMed(query, articles):  # Creates URL to search PubMed
    base_URL = "http://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    get_abstract_portion_URL = "&rettype=abstract"
    search_url = make_search_url(base_URL, query, articles)
    if len(search_url) > 2000:
        return_dict = {}
        return return_dict
    
    id_xml_as_String = get_xml(search_url)
    full_ID_List = get_ID_list(id_xml_as_String)
    info_from_PubMed = {}
    if full_ID_List["papers_to_download"]:
        fetch_url = make_fetch_url(base_URL,
                                   get_abstract_portion_URL,
                                   full_ID_List,
                                   articles)
        docs_xml = get_xml(fetch_url)
        info_from_PubMed = get_info_from_docs_xml(docs_xml,
                                        full_ID_List)

    return info_from_PubMed
        
#    if info_from_PubMed:      
#        return_dict = {"info_from_PubMed":info_from_PubMed, 
#                       "existing_id_list": full_ID_List["existing_papers"]}
#        return return_dict
#    else:
#        return_dict = {"existing_id_list":full_ID_List["existing_papers"]}
#        return return_dict


def main(query, articles):
    info_from_PubMed = get_info_from_PubMed(query, articles)
    return info_from_PubMed
