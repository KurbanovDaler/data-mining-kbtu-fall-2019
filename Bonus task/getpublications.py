import scholarly
import pandas as pd
import time
# time.sleep(5)
search_query = scholarly.search_pubs_query('Machine learning in logistics')
# pub = next(search_query)

cnt = 0
results = []

df = pd.DataFrame()
df.insert(0, "Title", ['Example']) 
df.insert(1, "Date", [2012]) 
df.insert(2, "Authors", ['Example']) 
df.insert(3, "Keywords", ['More than one']) 
df.insert(4, "Abstract", ['It be like that']) 
df.insert(5, "Text", ['Once upon a time....']) 
df.insert(6, "Url", ['https://www.sciencedirect.com/science/article/pii/S0898122112001113']) 

print (df.head())
for i in range(20):
    print(i)
    if cnt == 200:
        break
    pub = next(search_query)    
    time.sleep(5)
    pub.fill()        
    df = df.append({'Title' : pub.bib['title'].rstrip("\n\r"), 'Date': pub.bib['year'], 'Authors': pub.bib['author'].rstrip("\n\r"), 'Abstract': pub.bib['abstract'].rstrip("\n\r"), 'Url': pub.bib['url']} , ignore_index=True)
    print(pub.fill())
    # if 'eprint' in pub.bib:
# df['Authors'].apply(lambda x: x.replace('\n', ' '))
# df['Keywords'].apply(lambda x: x.replace('\n', ' '))
# df['Abstract'].apply(lambda x: x.replace('\n', ' '))
# df['Text'].apply(lambda x: x.replace('\n', ' '))
print (df.head())

df.to_csv('result.csv', index = False)