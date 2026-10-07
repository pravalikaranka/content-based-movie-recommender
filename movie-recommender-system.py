#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv') 


# In[3]:


movies.head(2)


# In[4]:


credits.head(2)


# In[5]:


movies.shape


# In[6]:


credits.shape


# In[7]:


movies=movies.merge(credits,on='title')


# In[8]:


movies.head(1)


# In[9]:


movies.shape


# In[11]:


movies.head(2)
# budget
# homepage
# id
# original_language
# original_title
# popularity
# production_comapny
# production_countries
# release-date(not sure)


# In[13]:


movies['original_language'].value_counts()


# In[14]:


movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]


# In[15]:


movies.head()


# In[20]:


movies.isnull().sum()


# In[19]:


movies.dropna(inplace=True)


# In[22]:


movies.duplicated().sum()


# In[23]:


movies.iloc[0].genres


# In[32]:


#'[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]'
#['Action','Adventure','Fantasy','Science Fiction']
import ast


# In[35]:


def convert(obj):
    L=[]
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L


# In[37]:


movies['genres']=movies['genres'].apply(convert)


# In[38]:


movies.head()


# In[39]:


movies['keywords'].apply(convert)


# In[40]:


movies['keywords']=movies['keywords'].apply(convert)


# In[41]:


movies.head()


# In[42]:


def convert(obj):
    L=[]
    counter=0
    for i in ast.literal_eval(obj):
        if counter != 3:
            L.append(i['name'])
            counter+=1
        else:
            break
    return L


# In[52]:


movies['cast']=movies['cast'].apply(convert)


# In[44]:


def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
    return L


# In[45]:


movies['crew'].apply(fetch_director)


# In[46]:


movies['crew'] = movies['crew'].apply(fetch_director)


# In[47]:


movies.head()


# In[48]:


movies['overview'][0]


# In[49]:


movies['overview'].apply(lambda x:x.split())


# In[50]:


movies['overview'] = movies['overview'].apply(lambda x:x.split())


# In[53]:


movies.head()


# In[54]:


#Sam Worthington
#SamWorthington
movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])


# In[55]:


movies['genres']=movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords']=movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast']=movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew']=movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])


# In[56]:


movies.head()


# In[57]:



movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']


# In[58]:


movies.head()


# In[59]:


new_df=movies[['movie_id','title','tags']]


# In[60]:


new_df


# In[65]:


new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))
new_df.head()


# In[66]:


new_df['tags'][0]


# In[69]:


new_df['tags']=new_df['tags'].apply(lambda x:x.lower())


# In[70]:


new_df.head()


# In[97]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')


# In[98]:


cv.fit_transform(new_df['tags']).toarray()


# In[99]:


cv.fit_transform(new_df['tags']).toarray().shape


# In[100]:


vectors=cv.fit_transform(new_df['tags']).toarray()


# In[101]:


vectors


# In[78]:


vectors[0]


# In[102]:


cv.get_feature_names()


# In[80]:


len(cv.get_feature_names())


# In[81]:


['loved','loving','love']
['love','love','love'] #after stemming
import nltk


# In[82]:


from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()


# In[91]:


def stem(text):
    y=[]
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)
        


# In[92]:


ps.stem('danced')


# In[86]:


ps.stem('dancing')


# In[89]:


new_df['tags'][0]


# In[93]:


stem('in the 22nd century, a paraplegic marine is dispatched to the moon pandora on a unique mission, but becomes torn between following orders and protecting an alien civilization. action adventure fantasy sciencefiction cultureclash future spacewar spacecolony society spacetravel futuristic romance space alien tribe alienplanet cgi marine soldier battle loveaffair antiwar powerrelations mindandsoul 3d samworthington zoesaldana sigourneyweaver jamescameron')


# In[95]:


new_df['tags'].apply(stem)


# In[96]:


new_df['tags']=new_df['tags'].apply(stem)


# In[103]:


from sklearn.metrics.pairwise import cosine_similarity


# In[110]:


similarity = cosine_similarity(vectors)


# In[108]:


cosine_similarity(vectors).shape


# In[117]:


sorted(list(enumerate(similarity[0])),reverse=True,key=lambda x:x[1])[1:6]


# In[113]:


similarity[0].shape


# In[114]:


similarity[1]


# In[118]:


def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list=sorted(list(enumerate(distances)),reverse=True,key = lambda x: x[1])
    for i in movies_list[1:6]:
        print(new_df.iloc[i[0]].title)


print("Movie recommender started successfully!")

recommend('Avatar')
recommend('Batman Begins')
recommend('Titanic')

print("Movie recommender finished successfully!")






