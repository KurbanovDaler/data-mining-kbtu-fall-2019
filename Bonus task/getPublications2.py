import scholarly
import pandas as pd
from tqdm import tqdm
search = scholarly.search_pubs_query('Machine learning in logistics')
author = []
title = []
abstract = []
date = []
url = []
eprint = []
for i in tqdm(range(500)):
    publication = next(search)
#     try:
#         s = BeautifulSoup(requests[publication.url_scholarbib).text, 'html.parser')
#         year = re.search(r'year={(\d*?\d*?\d{4})}', s.text)
#         date.append(int(year[1]))
#     except:
#         date.append(np.NaN)
    if 'eprint' in publication.bib:
        author.append(publication.bib['author'].rstrip("\n\r"))
        title.append(publication.bib['title'].rstrip("\n\r"))
        abstract.append(publication.bib['abstract'].rstrip("\n\r"))
        url.append(publication.bib['url'].rstrip("\n\r"))
        eprint.append(publication.bib['eprint'].rstrip("\n\r"))
                    
kek = pd.DataFrame(
{
    'author': author,
    'title' : title,
    'abstract':abstract,
    'url':url,
    'eprint': eprint
})
kek.to_csv('result2.csv', index = False)