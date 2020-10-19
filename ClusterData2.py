# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 12:46:49 2017

@author: KiranChilla
"""
import glob,os
import string
import re
import nltk.corpus
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from string import digits
import numpy as np
import nltk
import pandas as pd
import numbers,operator
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import wordnet
english_vocab = set(w.lower() for w in nltk.corpus.words.words())
#calculating length
#basePath = r""
basePath = r""
list = os.listdir(basePath) # dir is your directory path
number_files = len(list)
print(number_files)

#reading names of multiple files
filePaths = glob.glob(os.path.join(basePath,'FlumeData.*'))
# Just open first ocurrence, if any

FinalS='a'
#reading data from file line by line from each file
a='"text":'
b=',"place"'
count=0
count1=0
user_ids=[]
processedword_ids=[]
all_words_to_id={}
TweetCount=0
filenumber=0


#write to function
def WriteToFile(cnum, filenumber, line1):
  Fname="ClusterKmeans"+str(cnum)+"-"+str(filenumber)+".txt"
  with open(Fname, "w") as myfile:
    myfile.write(line1)
    myfile.close()

#append data
def AppendToFunc(cnum,filenumber, line1):
  line1=line1+"\n"
  #print(line1)
  Fname="ClusterDTest"+str(cnum)+"-"+str(filenumber)+".txt"
  with open(Fname, "a") as myfile:
    myfile.write(line1)
    myfile.close()



#reading data from files and performming data cleaning
for i in range(number_files-2):   
    count1=0
    for line in open(filePaths[i],encoding="utf8").readlines():
        try:
            line=line.split(a)[1].split(b)[0]
        #line=line.lower()
        except Exception as inst:  
          continue
        
        #remove urls
        line = re.sub(r'^https?:\/\/.*[\r\n]*',' ', line)
        #removing years
        remove_digits = str.maketrans('', '', digits)
        line = line.translate(remove_digits)
        line = re.sub('[!@#$–‘’“”—]',',', line)
         #removingstopwords 
        stop_words = set(stopwords.words("english"))
        
                #removing punctuations
        for data in string.punctuation:
                line=line.replace(data,"  ")
        line=re.sub("\s+",",",line.strip())
        line=line.lower()
        strings=("fever","severe headache","muscle pain","weakness","fatigue","diarrhea","vomiting","abdominal","stomach pain","unexplained hemorrhage","zaire","vsv-ebov")
        check=0
        if any(s in line for s in strings):
            check=1
            line1=""
            tokenized = nltk.word_tokenize(line)
            is_noun = lambda pos: pos[:2] == 'NN'
            Line_words = [word for (word, pos) in nltk.pos_tag(tokenized) if is_noun(pos)]
            Unwanted_words=['null','photo','reply','name','fit','id','user']
            #processing each word from sorted data
            for index in range(len(Line_words)):
                word=Line_words[index]

                
                if word not in stopwords.words("english") and word not in Unwanted_words:
                        #Considering only English Words
                        if word in english_vocab and len(word)>3 :
                             line1=line1+"\t"+word
        
                             if word not in all_words_to_id:
                                 all_words_to_id[word]=len(all_words_to_id)
                             user_ids.append(count)
                             processedword_ids.append(all_words_to_id[word])
                             FinalS=FinalS+"\t"+word
            count+=1
            count1+=1





#implementation of Kmeans clusteing algorithm

#clusterData
from scipy.sparse import csr_matrix
rows=np.array(processedword_ids)
cols=np.array(user_ids)
data=np.ones((len(user_ids),))
num_rows=len(all_words_to_id)
num_cols=count
processeswords=[]
adj=csr_matrix( (data,(rows,cols)), shape=(num_rows, num_cols) )
print(adj.shape)
users_per_processedwords = adj.sum(axis=1).A1
processedwords=[0 for x in range(0,len(all_words_to_id))]
for word in all_words_to_id:
    processedwords[all_words_to_id[word]]=word
processedwords=np.array(processedwords)

 #SVD reducing the dimensions

from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
svd= TruncatedSVD(n_components=10)
embedded_coords = normalize(svd.fit_transform(adj))
        
from scipy.stats import rankdata
embedded_ranks = np.array([rankdata(c) for c in embedded_coords.T]).T

from sklearn.cluster import KMeans

no_clusters = [5,10,15,20,50]
#repeating the loop for different clusters
for ClusterTypes in range(5):
    print(ClusterTypes)
    Topic=" "
 #Clustering Algorithm
    n_clusters = no_clusters[ClusterTypes]
    km = KMeans(n_clusters)
    clusters = km.fit_predict(embedded_coords)        
                    
    clusters=np.array(clusters)
    #writing the data in to fules for each cluster 
    for num in range(n_clusters):
    
         ClusterIndice=np.where(clusters==num)
         for i in range(len(ClusterIndice[0])):
            
             clustername=" "
             if(i==0):
                 WriteToFile(n_clusters,num,clustername)
                 Word=processedwords[ClusterIndice[0][i]]
             else:
                 Word=Word+" "+processedwords[ClusterIndice[0][i]]
             if i%20==0 & i>0:
                 AppendToFunc(num, Word)
                 Word=processedwords[ClusterIndice[0][i]]
                  
         AppendToFunc(n_clusters,num, Word)           
            
          
            
            
#predicting topic from clusters
    
    Unique_Words=[]
    initial=0
    WordFound=0
    count_t=0
    Distance_Count=[0 for j1 in range(30000)]
    taxonomy=[0 for j2 in range(30000)]
    k=0
    
    #reading every cluster from selected cluster group
    for k in range(0,n_clusters):
        Unique_Words=[]
        initial=0
        WordFound=0
        count_t=0
        List1=[]
        Distance_Count=[0 for j1 in range(30000)]
        taxonomy=[0 for j2 in range(30000)]
        
        filename="ClusterKmeans"+str(n_clusters)+"-"+str(k)+".txt"
        with open(filename) as f:
                print(filename)
                for line in f:
                    is_noun = lambda pos: pos[:2] == 'NN'
                    # get the nouns
                    tokenized = nltk.word_tokenize(line)
                    Line_words = [word for (word, pos) in nltk.pos_tag(tokenized) if is_noun(pos)]

                    if initial==0:
                            Unique_Words.append(Line_words[0])
                            initial+=1                
                    for l in range(0,len(Line_words)):
                                               
                        Range=len(Unique_Words)
                        if Line_words[l] not in Unique_Words:# and Line_words[l] not in Words2:
                             List1.append(Line_words[l])
                             Unique_Words.append(Line_words[l])
                       
                        for j in range(0,Range):
                
                
                         try: 
                                                  Synset1 = wordnet.synsets(Line_words[l])
                                                  Synset2 = wordnet.synsets(Unique_Words[j])
                                                  #print(Synset1,"\t",Synset2,"\t",k)
                                                 #calculating similarity Index
                                                  taxonomy_distance = Synset1[0].wup_similarity(Synset2[0])
                                                  Distance_Count[j]=Distance_Count[j]+taxonomy_distance
                                            
                                                  taxonomy[count_t]= taxonomy_distance
                                                  count_t+=1 
                                                  WordFound=1
                     
                         except:
                                        #print("error")        
                                        continue    
    
                 #finding the topic with maximum similarity    
                MaxIndices, value = max(enumerate(Distance_Count), key=operator.itemgetter(1))

    
                PredictedTopics=Unique_Words[MaxIndices]
     
                Topic=Topic+"\t"+PredictedTopics
                temp=2000      
                WriteToFile(n_clusters,temp, Topic)
                
