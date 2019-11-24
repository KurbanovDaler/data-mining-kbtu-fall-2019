import scholarly
import pandas as pd

search_query = scholarly.search_pubs_query('Machine learning in logistics')
pub = next(search_query)
pub.fill()
print(pub)

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
# search_query = scholarly.search_pubs_query('Machine learning in logistics')
# pub = next(search_query)
print (pub.bib['abstract'])
# df.to_csv('test.csv', index = False)