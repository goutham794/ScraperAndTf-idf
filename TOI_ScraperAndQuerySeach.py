
# coding: utf-8

# # Using BeautifulSoup and Newspaper library to scrape news off TOI site and search a query, returning results ranked by TF-IDF

# In[86]:

import bs4 as bs 
import urllib.request
from newspaper import Article
import pickle
import re
import string
import math
import numpy as np


# ## Creating Beautiful soup object.

# In[87]:

url = "https://timesofindia.indiatimes.com/2018/2/26/archivelist/year-2018,month-2,starttime-43157.cms"
source = urllib.request.urlopen(url).read()
soup = bs.BeautifulSoup(source, 'lxml')
urllist = []


# ## News article links are found in Table2, storing all the links in a list.

# In[88]:

links = soup.findAll('table')[2].findAll('a')


# In[89]:

for link in links:
    urllist.append(link.get("href"))


# ## Article titles and body are stored in lists after scraping

# In[90]:

article_titles = []
article_body = []
try:

	for i in range(0, 50):
		newurl = 'https://timesofindia.indiatimes.com'+urllist[i]
		article = Article(newurl)
		article.download()
		article.parse()
		article_titles.append(article.title)
		article_body.append(article.text)	

except Exception as e:
	pass


# ## Text processing, converting to lower case and removing stopwords

# In[91]:

stopwords = set([
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'but', 'by',
    'for', 'if', 'in', 'into', 'is', 'it',
    'no', 'not', 'of', 'on', 'or', 's', 'such',
    't', 'that', 'the', 'their', 'then', 'there', 'these',
    'they', 'this', 'to', 'was', 'will', 'with'
])


processed_corpus = [[word for word in text.lower().split() if word.isalnum() and word not in stopwords]for text in article_body]


# ## Building a dictionary of all words in corpus

# In[92]:

def build_lexicon(docs):
    lexicon = set()
    for doc in docs:
        lexicon.update(doc)
    return(lexicon)


dictionary = build_lexicon(processed_corpus)


# ## Functions for calculating TF weight and IDF weight

# In[93]:

def tf(article, word):
    return article.count(word)

def numDocsContaining(word):
    doccount = 0
    for doc in processed_corpus:
        if doc.count(word) > 0:
            doccount +=1
    return doccount

def idf(word):
    n_samples = len(processed_corpus)
    df = numDocsContaining(word)
    return math.log(n_samples / 1+df)


# ## Creation of document matrix, each element is a td-idf weight vector of the article

# In[94]:

doc_term_matrix = []
for doc in processed_corpus:
    tfidf_vector = np.array([math.log(1+tf(doc,word))*idf(word) for word in dictionary])
    doc_term_matrix.append(tfidf_vector)


# ## Accepting query from user

# In[95]:

query = input("enter query :- ")
processed_query = [word for word in query.lower().split() if word.isalnum() and word not in stopwords]

query_tdidf = np.array([math.log(1+tf(processed_query,word))*idf(word) for word in dictionary])


# ## Calculation of cosine scores and printing articles that have the highest scores

# In[96]:

cosine_scores = np.array([np.dot(doc_tfidf,query_tdidf)/(np.linalg.norm(doc_tfidf)*np.linalg.norm(query_tdidf)) for doc_tfidf in doc_term_matrix])
ranked_indices = np.argsort(-1 * cosine_scores)
rank = list(ranked_indices)[0:5]

for n in rank:
    print (article_titles[n])
    print (article_body[n])
    print("\n")

