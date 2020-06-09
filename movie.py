#!/usr/bin/env python
# coding: utf-8

# In[110]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')


# In[111]:


df1=pd.read_csv('Movie_Id_Titles')
df1.head(1)


# In[112]:


df2=pd.read_csv('u.data',delimiter='\t',names=['user_id','item_id','rating','timestamp'])
df2.head(1)


# In[113]:


df2=df2.drop(['timestamp'],axis=1)
df2.info()


# In[114]:


data_set=pd.merge(df1,df2,on='item_id')
data_set


# In[115]:


data_set.groupby('title')['rating'].describe()


# In[116]:


data_set_mean=data_set.groupby('title')['rating'].describe()['mean']
data_set_mean


# In[117]:


data_set_count=data_set.groupby('title')['rating'].describe()['count']
data_set_count


# In[118]:


data_set_mean_count=pd.concat([data_set_mean,data_set_count],axis=1)
data_set_mean_count


# In[119]:


data_set_mean_count.reset_index()


# In[120]:


data_set_mean_count['mean'].plot(bins=100,kind='hist',color='g')


# In[121]:


data_set_mean_count['count'].plot(bins=100,kind='hist',color='r')


# In[122]:


data_set_mean_count[data_set_mean_count['mean']==5]


# In[123]:


data_set_mean_count.sort_values('count',ascending=False).head(1)


# In[124]:


userid_title_matrix=data_set.pivot_table(index='user_id',columns='title',values='rating')
userid_title_matrix


# In[125]:


titanic=userid_title_matrix['Titanic (1997)']


# In[126]:


titanic_correlation=pd.DataFrame(userid_title_matrix.corrwith(titanic),columns=['Correlation'])
titanic_correlation=titanic_correlation.join(data_set_mean_count['count'])
titanic_correlation


# In[127]:


titanic_correlation.dropna(inplace=True)
titanic_correlation.sort_values('Correlation',ascending=False)
titanic_correlation[titanic_correlation['count']>80].sort_values('Correlation',ascending=False).head()


# In[128]:


userid_title_matrix


# In[129]:


movie_correlation=userid_title_matrix.corr(method='pearson',min_periods=80)


# In[130]:


myRatings=pd.read_csv('My_Ratings.csv')
myRatings


# In[131]:


similar_movies_list=pd.Series()
for i in range(0,2):
    similar_movie=movie_correlation[myRatings['Movie Name'][i]].dropna()
    similar_movie=similar_movie.map(lambda x:x*myRatings['Ratings'][i])
    similar_movies_list=similar_movies_list.append(similar_movie)


# In[132]:


similar_movies_list.sort_values(inplace = True, ascending = False)
print (similar_movies_list.head(10))


# In[ ]:




