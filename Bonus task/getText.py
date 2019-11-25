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
df = pd.read_csv('result2.csv')
print (df['eprint'])

def get_text(url):
    # print (url)
    # print (url == )    
    try:
        r = requests.get(url)
    except:
        return 'NO TEXT FOUND'
    f = io.BytesIO(r.content)
    document = f
    rsrcmgr = PDFResourceManager()
    laparams = LAParams()
    device = PDFPageAggregator(rsrcmgr, laparams=laparams)
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    content = ''
    try:
        for page in PDFPage.get_pages(document):
            interpreter.process_page(page)
            layout = device.get_result()
            for element in layout:
                if isinstance(element, LTTextBoxHorizontal):
                    content = content + element.get_text().replace('\n', " ")
    except:
        return 'NO TEXT FOUND'
    return content



texts = []

for text in tqdm(df['eprint']):
    # print (" ")
    # print (text)
    # print (get_text(text))
    texts.append(get_text(text))    
print (len(texts))
df.insert(3, 'text', texts, True)
df.to_csv('result3.csv', index = False)