import scholarly
import pandas as pd
from tqdm import tqdm
import time
search = scholarly.search_pubs_query('Machine learning in logistics')
author = []
title = []
abstract = []
date = []
url = []
eprint = []
date = []
for i in tqdm(range(300)):
    publication = next(search)
    time.sleep(5)
    publication.fill()    
#     try:
#         s = BeautifulSoup(requests.get(publication.url_scholarbib).text, 'html.parser')
#         year = re.search(r'year={(\d*?\d*?\d{4})}', s.text)
#         date.append(int(year[1]))
#     except:
#         date.append(np.NaN)\
    date.append(publication.bib.get('year'))
    author.append(publication.bib.get('author'))
    title.append(publication.bib.get('title'))
    abstract.append(publication.bib.get('abstract'))
    url.append(publication.bib.get('url'))
    eprint.append(publication.bib.get('eprint'))
                    
kek = pd.DataFrame(
{
    'author': author,
    'date' : date,
    'title' : title,
    'abstract':abstract,
    'url':url,
    'eprint': eprint
})
kek.to_csv('result2.csv', index = False)