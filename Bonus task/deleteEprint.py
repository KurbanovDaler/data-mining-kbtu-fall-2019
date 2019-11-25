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
    
df = pd.read_csv('final_result.csv')
df = df.drop(columns="eprint")
columnsTitles=['title', 'date', 'author', 'keywords', 'abstract', 'text', 'url']
df=df.reindex(columns=columnsTitles)
df.to_csv('final_result_I_swear.csv', index = False)