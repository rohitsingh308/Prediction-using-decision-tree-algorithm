#!/usr/bin/env python
# coding: utf-8

# # Name: Rohit Kumar Singh
# 
# # Task 6: Prediction_using_Decision_Tree_Algorithm
# 
# # The Sparks Foundation
# 
# # Iot & Computer Vision Intern
# 
# # GRIPMAY23

# In[1]:


#importing all the necessary libraries.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sbn
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn import metrics


# In[9]:


dataset = pd.read_csv(r"E:\CS\python\New folder\Iris.csv") #Your folder location, please check from properties
dataset.head() #this function helps us to view the dataset



# In[10]:


dataset.describe() #to view the statistical data of the iris dataset


# In[11]:


dataset.info() #prints a concise summary of the dataset


# In[12]:


dataset.shape


# In[13]:


dataset.isna().sum()


# In[14]:


dataset.duplicated().sum()


# In[15]:


sbn.FacetGrid(dataset,hue="Species").map(plt.scatter,'SepalLengthCm','SepalWidthCm').add_legend()
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")
plt.show()


# In[16]:


sbn.FacetGrid(dataset,hue="Species").map(plt.scatter,'PetalLengthCm','PetalWidthCm').add_legend()
plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")
plt.show()


# In[17]:


dataset['Species'].value_counts().plot(kind = 'pie', autopct = '%1.1f%%', shadow = True, explode = [0.08,0.08,0.08])


# In[18]:


x = dataset.iloc[:, 1:-1].values #x is the matrix of features
y = dataset.iloc[:,-1].values #y is a vector of observed outcomes


# In[19]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state=0)
#we would have 30 observation in test set and 120 observations in the training set


# In[20]:


classifier = DecisionTreeClassifier(criterion = 'entropy')
classifier.fit(x_train,y_train)


# In[21]:


plt.figure(figsize=(15,10))
tree.plot_tree(classifier, filled = True)


# In[ ]:


# THANK YOU , PLEASE DO PROVIDE YOYR VALUABLE FEEDBACK!!


# In[ ]:




