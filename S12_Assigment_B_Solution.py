#!/usr/bin/env python
# coding: utf-8

# In[1]:


path="C:\\Users\\a793877\\OneDrive - Atos\\Desktop\\EVA\\s12\\S12_Assigment_B"
filename="S12_json.json"


# In[2]:


import os
os.chdir(path)


# In[3]:


from annotation_class import *


# In[4]:


# REad JASON file and create dataset
data=readJSON(path,filename)


# In[5]:


#Get dataset for imag details
data_img=addImagesize(path,filename)
data_img


# In[6]:


#MErging Img data and JSON Annotation data to get final dataset
final_df=finaldataframe(data_img,data)


# In[7]:


# Perform calculation on data to get the final dataset with scaled values
final_df_scaled=calculations(final_df)


# In[8]:


# Plot scatter plot for scaled and log scaled data
plotScatter(final_df_scaled)


# In[9]:


# Create Elbow method to understaND VALUE OF K IN SCALED HEIGHT AND WIDTH
elbowMethodscaled(final_df_scaled)
#nUMBER OF CLUSTERS FOR SCALED DATA IS 4


# In[10]:


# Create Elbow method to understaND VALUE OF K IN log HEIGHT AND WIDTH
elbowMethodlog(final_df_scaled)
#nUMBER OF CLUSTERS FOR SCALED DATA IS 6


# In[11]:


# Create K MEans clusters for scaled HEight and weight data
clusteringScalewd(final_df_scaled,4)


# In[13]:


# Create K MEans clusters for log HEight and weight data
clusteringLog(final_df_scaled,6)

