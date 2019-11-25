import scholarly
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
import requests
import io
import pdfminer
from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from pdfminer.layout import LTTextBoxHorizontal
import math
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.probability import FreqDist
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords    
from sklearn.feature_extraction.text import CountVectorizer
import re
import seaborn as sns
    
#Most frequently occuring words
def get_top_n_words(corpus, n=None):
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in      
                   vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], 
                       reverse=True)
    return words_freq[:n]

#Most frequently occuring Bi-grams
def get_top_n2_words(corpus, n=None):
    vec1 = CountVectorizer(ngram_range=(2,2),  
            max_features=2000).fit(corpus)
    bag_of_words = vec1.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in     
                  vec1.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], 
                reverse=True)
    return words_freq[:n]

#Most frequently occuring Tri-grams
def get_top_n3_words(corpus, n=None):
    vec1 = CountVectorizer(ngram_range=(3,3), 
           max_features=2000).fit(corpus)
    bag_of_words = vec1.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in     
                  vec1.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], 
                reverse=True)
    return words_freq[:n]
df = pd.read_csv('result3.csv')
# print (df['text'])

stop_words=set(stopwords.words("english"))
# print(stop_words)
res = []
# for i in range(5):
#     print(df.loc[i].iloc[3])
for i in tqdm(range(400)):
# for text in df['text']: #tqdm(df['text']):
    ngram = 5    
    text = df.loc[i].iloc[3]
    if pd.isna(text):
        text = "NO TEXT FOUND"
        df.loc[i].iloc[3]  = "NO TEXT FOUND"
    # print(df.loc[i].iloc[1])
    keywords = []
    if text == "NO TEXT FOUND":
        text = df.loc[i].iloc[2]
        ngram = 3
        if pd.isna(text):
            text = "NO TEXT FOUND"
            df.loc[i].iloc[2]  = "NO ABSTRACT FOUND"
            res.append('NO KEYWORDS FOUND')
            continue
    # print(text)
    # if len(text) < 8:
    #     res.append('No keywords found')
    #     continue
    tokenized_sent=word_tokenize(text)
    # print(tokenized_sent)

    filtered_sent=[]

    for w in tokenized_sent:
        if w not in stop_words:
            filtered_sent.append(w)            
    # print("Tokenized Sentence:",tokenized_sent)
    ps = PorterStemmer()

    lemmatized_words=[]
    # for w in filtered_sent:
    #     stemmed_words.append(ps.stem(w))
    lem = WordNetLemmatizer()
    for w in filtered_sent:
        lemmatized_words.append(lem.lemmatize(w))
    # print("Filtered Sentence:",filtered_sent)
    # print("Lemmatized Sentence:",lemmatized_words)
    # print(len(lemmatized_words))
    try:
        cv=CountVectorizer(max_df=0.8,stop_words=stop_words, max_features=10000, ngram_range=(1,3))
        X=cv.fit_transform(lemmatized_words)
    except:
        res.append('NO KEYWORDS FOUND')
        continue
    # print(len(list(cv.vocabulary_.keys())))
    # print(list(cv.vocabulary_.keys()))
    try:
        top_words = get_top_n_words(lemmatized_words, n=ngram)
        top_df = pd.DataFrame(top_words)
        top_df.columns=["Word", "Freq"]
    except:
        top_words = [('NO KEYWORDS FOUND', 1)]
    # print(top_df)
    
    # sns.set(rc={'figure.figsize':(13,8)})
    # g = sns.barplot(x="Word", y="Freq", data=top_df)
    # g.set_xticklabels(g.get_xticklabels(), rotation=30)

    try:
        top2_words = get_top_n2_words(lemmatized_words, n=ngram)
        top2_df = pd.DataFrame(top2_words)
        top2_df.columns=["Bi-gram", "Freq"]
    except:
        top2_words = [('', 1)]
    # print(top2_df)
    
    # sns.set(rc={'figure.figsize':(13,8)})
    # h=sns.barplot(x="Bi-gram", y="Freq", data=top2_df)
    # h.set_xticklabels(h.get_xticklabels(), rotation=45)

    # top3_words = get_top_n3_words(lemmatized_words, n=5)
    # top3_df = pd.DataFrame(top3_words)
    # top3_df.columns=["Tri-gram", "Freq"]
    # print(top3_df)
    
    # sns.set(rc={'figure.figsize':(13,8)})
    # j=sns.barplot(x="Tri-gram", y="Freq", data=top3_df)
    # j.set_xticklabels(j.get_xticklabels(), rotation=45)
    
    append_words = [x[0] for x in top_words]
    append2_words = [x[0] for x in top2_words]

    keywords.append(append_words)
    keywords.append(append2_words)
    # keywords.append(top2_df.iloc[0])
    # keywords.append(top3_df.iloc[1])
    flattened_keywords = [y for x in keywords for y in x]
    listToStr = ' '.join([str(elem) for elem in flattened_keywords])
    # print (flattened_keywords)
    res.append(listToStr)
    # print (res)
    # break

# tokenized_text=sent_tokenize(text)
print(len(res))

df.insert(3, 'keywords', res, True)
df.to_csv('final_result.csv', index = False)