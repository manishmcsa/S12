#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os
import seaborn as sns
import math
from numpy import array
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt

# In[2]:


def readJSON(path,filename):
    os.chdir(path)
    df=pd.read_json(filename)
    df=df.transpose()
    data=pd.DataFrame()
    for i in range(0,len(df)):
        df1=pd.json_normalize(df['regions'][i])
        data=data.append(df1,ignore_index=True)
    data_new=data.rename(columns={"shape_attributes.x" : "x",
                             "shape_attributes.y" : "y",
                             "shape_attributes.width" : "w",
                             "shape_attributes.height" : "h",
                             "region_attributes.type" : "class"})
    data_img=data_new[["class","x","y","w","h"]]
    print(data_img)
    return data_img


# In[3]:


def addImagesize(path,filename):
    os.chdir(path)
    data=pd.read_json(filename)
    data=data.transpose()
    data_new=pd.DataFrame()
    data_new=data.drop(data.index[20:88])
    data_new=data_new.drop(data.index[15])
    data_new_img=pd.DataFrame()
    data_new_img["img"]=data_new["filename"]
    data_img=list(data_new_img["img"])
    list1=[]
    for i in range (0,len(data_new)):
        c=pd.json_normalize(data_new["regions"][i])
        list1.append(len(c))
    list1=list(filter(lambda num: num != 0, list1)) 
    list2=[]
    for i in range(0,len(list1)):
        c=[data_img[i],]*list1[i]
        list2.append(c)
    Image_name = [ item for elem in list2 for item in elem]
    df1=pd.DataFrame()
    df1["Img_Name"]=Image_name
    img_width=[283,259,273,275,220,275,276,277,300,266,291,299,276,259,450,275,184,275,276,265,177,259,203,299,300]
    img_height=[178,194,185,183,165,183,183,182,168,190,173,168,183,194,320,183,275,183,183,190,285,194,248,168,168]
    width=[]
    height=[]
    
    for i in range (len(Image_name)):
        if(Image_name[i]=="1.jpg"):
            width.append(img_width[0])
            height.append(img_height[0])
        elif(Image_name[i]=="2.jpg"):
            width.append(img_width[1])
            height.append(img_height[1])
        elif(Image_name[i]=="3.jpg"):
            width.append(img_width[2])
            height.append(img_height[2])
        elif(Image_name[i]=="4.jpg"):
            width.append(img_width[3])
            height.append(img_height[3])
        elif(Image_name[i]=="5.jpg"):
            width.append(img_width[4])
            height.append(img_height[4])
        elif(Image_name[i]=="6.jpg"):
            width.append(img_width[5])
            height.append(img_height[5])
        elif(Image_name[i]=="7.jpg"):
            width.append(img_width[6])
            height.append(img_height[6])
        elif(Image_name[i]=="8.jpg"):
            width.append(img_width[7])
            height.append(img_height[7])
        elif(Image_name[i]=="9.jpg"):
            width.append(img_width[8])
            height.append(img_height[8])
        elif(Image_name[i]=="10.jpg"):
            width.append(img_width[9])
            height.append(img_height[9])
        elif(Image_name[i]=="11.jpg"):
            width.append(img_width[10])
            height.append(img_height[10])
        elif(Image_name[i]=="12.jpg"):
            width.append(img_width[11])
            height.append(img_height[11])
        elif(Image_name[i]=="13.jpg"):
            width.append(img_width[12])
            height.append(img_height[12])
        elif(Image_name[i]=="14.jpg"):
            width.append(img_width[13])
            height.append(img_height[13])
        elif(Image_name[i]=="15.jpg"):
            width.append(img_width[14])
            height.append(img_height[14])
        elif(Image_name[i]=="17.jpg"):
            width.append(img_width[15])
            height.append(img_height[15])
        elif(Image_name[i]=="18.jpg"):
            width.append(img_width[16])
            height.append(img_height[16])
        elif(Image_name[i]=="19.jpg"):
            width.append(img_width[17])
            height.append(img_height[17])
        elif(Image_name[i]=="20.jpg"):
            width.append(img_width[18])
            height.append(img_height[18])
        elif(Image_name[i]=="91.jpg"):
            width.append(img_width[19])
            height.append(img_height[19])
        elif(Image_name[i]=="93.jpg"):
            width.append(img_width[20])
            height.append(img_height[20])
        elif(Image_name[i]=="95.jpg"):
            width.append(img_width[21])
            height.append(img_height[21])
        elif(Image_name[i]=="96.jpg"):
            width.append(img_width[22])
            height.append(img_height[22])
        elif(Image_name[i]=="98.jpg"):
            width.append(img_width[23])
            height.append(img_height[23])
        elif(Image_name[i]=="99.jpg"):
            width.append(img_width[24])
            height.append(img_height[24])   
    df1["Img_Height"] = height
    df1["Img_Width"] = width  
    return df1


# In[4]:


def finaldataframe(df1,df2):
    frames=[df1,df2]
    final_data=pd.concat(frames,axis=1)
    print(final_data)
    return final_data


# In[5]:


def calculations(df):
    hdf=[]
    wdf=[]
    scaled_height=[]
    scaled_weight=[]
    scaled_x=[]
    scaled_y=[]
    scaled_w=[]
    scaled_h=[]
    log_h=[]
    log_w=[]
    for i in range (len(df)):
        hdf.append(df["Img_Height"][i])
        wdf.append(df["Img_Width"][i])
    for i in range (len(df)):
        scaled_height.append(df["Img_Height"][i]/hdf[i])
        scaled_weight.append(df["Img_Width"][i]/wdf[i])
    for i in range (len(df)):
        scaled_x.append(df["x"][i]/wdf[i])
        scaled_y.append(df["y"][i]/hdf[i])
    for i in range (len(df)):
        scaled_w.append(df["w"][i]/wdf[i])
        scaled_h.append(df["h"][i]/hdf[i])
    for i in range (len(df)):
        log_h.append(math.log(scaled_h[i]))
        log_w.append(math.log(scaled_w[i]))
    df["wdf"]=wdf
    df["hdf"]=hdf
    df["scaled_height"]=scaled_height
    df["scaled_weight"]=scaled_weight
    df["scaled_x_img"]=scaled_x
    df["scaled_y_img"]=scaled_y
    df["scaled_w_img"]=scaled_w
    df["scaled_h_img"]=scaled_h
    df["log_h"]=log_h
    df["log_w"]=log_w
    print(df)
    return df

# In[6]:


def plotScatter(data):
    plt.scatter(list(data["scaled_w_img"]), list(data["scaled_h_img"]))
    plt.xlabel('scaled_w_img')
    plt.ylabel('scaled_h_img')
    plt.title('Scaled Height and Weight Scatter Plot')
    plt.show()
    plt.scatter(list(data["log_w"]), list(data["log_h"]))
    plt.xlabel('log_w')
    plt.ylabel('log_h')
    plt.title('Log Height and Weight Scatter Plot')
    plt.show()


# In[7]:


def clusteringScalewd(data,n):
    data_clust=pd.DataFrame()
    data_clust[["w","h"]]=data[["scaled_w_img","scaled_h_img"]]
    kmeans = KMeans(n_clusters=n)
    y = kmeans.fit_predict(data_clust)
    data_clust['Cluster'] = y
    results = pd.DataFrame(data_clust,columns=['w','h'])
    sns.scatterplot(x="w", y="h", hue=data_clust['Cluster'], data=results)
    plt.title('K-means Clustering with 2 dimensions')
    plt.show()


# In[8]:


def clusteringLog(data,n):
    data_clust=pd.DataFrame()
    data_clust[["w","h"]]=data[["log_w","log_h"]]
    kmeans = KMeans(n_clusters=n)
    y = kmeans.fit_predict(data_clust)
    data_clust['Cluster'] = y
    results = pd.DataFrame(data_clust,columns=['w','h'])
    sns.scatterplot(x="w", y="h", hue=data_clust['Cluster'], data=results)
    plt.title('K-means Clustering with 2 dimensions')
    plt.show()


# In[9]:


def elbowMethodlog(data):
    data_clust1=pd.DataFrame()
    data_clust1[["w","h"]]=data[["log_w","log_h"]]
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(data_clust1)
        wcss.append(kmeans.inertia_)
    plt.plot(range(1, 11), wcss)
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()

def elbowMethodscaled(data):
    data_clust1=pd.DataFrame()
    data_clust1[["w","h"]]=data[["scaled_w_img","scaled_h_img"]]
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(data_clust1)
        wcss.append(kmeans.inertia_)
    plt.plot(range(1, 11), wcss)
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()
