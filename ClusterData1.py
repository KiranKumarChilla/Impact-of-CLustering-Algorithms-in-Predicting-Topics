# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 14:16:21 2017

@author: KiranChilla
"""



from time import time

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation

n_samples = 20000
n_features = 2000
n_components = [5,10,15,20,50]
n_top_words = 1000


def WriteToFile(Name,cnum, filenumber, line1):
  Fname="Cluster"+Name+str(cnum)+"-"+str(filenumber)+".txt"
  with open(Fname, "w") as myfile:
    myfile.write(line1)
    myfile.close()



def print_top_words(Name,model, feature_names, n_top_words,noClust):
    Countf=0
    Cno=noClust
    for topic_idx, topic in enumerate(model.components_):
       #message = "Topic #%d: " % topic_idx
        message = " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
        WriteToFile(Name,Cno,Countf,message)
        Countf+=1
    print()


print("Loading dataset...")
t0 = time()


import glob,os
import string
import re
import nltk.corpus
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from string import digits
import numpy as np
import nltk
import pandas as pd
import numbers,operator
from itertools import product
english_vocab = set(w.lower() for w in nltk.corpus.words.words())
import random

#give the folder path
basePath = r"C:/Users/Kiranchilla/Desktop/Fall 2017/extras/Independenntstudy/Ebola/EbolaVirusData"
list = os.listdir(basePath) # dir is your directory path
number_files =len(list)
print(number_files)

#reading names of multiple files
filePaths = glob.glob(os.path.join(basePath,'FlumeData.*'))


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
#f=open("data2.txt","w")

data_samples=[]

for i in range(number_files-2):   
    count1=0
    for line in open(filePaths[i],encoding="utf8").readlines():
        try:
          line=line.split(a)[1].split(b)[0]
        except Exception as inst:
          print(i,"file")
          continue
              
        #line=line.lower()
        
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
       # strings=("trump","donald trump", "anger","melania", "impeachment","us","usa","America","president","washington","muslim","racism","H1-b","terrorism","radical","radical islam","unitedstates","syria","army","frustrate")
        strings=("fever","severe headache","muscle pain","weakness","fatigue","diarrhea","vomiting","abdominal","stomach pain","unexplained hemorrhage","zaire","vsv-ebov")
        Unwanted_words=['null','photo','reply','name','fit','id','user','pic']
        check=0
        if any(s in line for s in strings):
            check=1 
            line1=""
            for word in line.rstrip().split(","):
               # word = word.lower() # in case they arenet all lower cased
                if word not in stopwords.words("english") and word not in Unwanted_words:
                        #Considering only English Words
                        if word in english_vocab and len(word)>3 :
                             line1=line1+"\t"+word
                
                TweetCount+=1 
                data_samples.append(line1)


print("done in %0.3fs." % (time() - t0))









for type in range(5):
    Topic=" "
    n_clusters=n_components[type]
    # Use tf-idf features for NMF.
    print("Extracting tf-idf features for NMF...")
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
                                       max_features=n_features,
                                       stop_words='english')
    t0 = time()
    tfidf = tfidf_vectorizer.fit_transform(data_samples)
    print("done in %0.3fs." % (time() - t0))
    
    # Use tf (raw term count) features for LDA.
    print("Extracting tf features for LDA...")
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                    max_features=n_features,
                                    stop_words='english')
    t0 = time()
    tf = tf_vectorizer.fit_transform(data_samples)
    print("done in %0.3fs." % (time() - t0))
    print()
    X=random.randint(0,12)
    
    
    # Fit the NMF modelFrobenius norm
    t0 = time()
    nmf = NMF(n_components=n_components[type], random_state=1,
              alpha=.1, l1_ratio=.5).fit(tfidf)
    print("done in %0.3fs." % (time() - t0))
    
    #print("\nTopics in NMF model (Frobenius norm):")
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()
    print_top_words("FN",nmf, tfidf_feature_names, n_top_words,n_clusters)
    
    
    
    
    
    # Fit the NMF model Kullback-Leibler divergence
    t0 = time()
    nmf = NMF(n_components=n_components[type], init=None, random_state=None,max_iter=1000, alpha=.1,l1_ratio=.5).fit(tfidf) 
    print("done in %0.3fs." % (time() - t0))
    
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()
    print_top_words("KL",nmf, tfidf_feature_names, n_top_words,n_clusters)
    
    
    
    
    
    
    #fit with LDA model
    
    lda = LatentDirichletAllocation(n_topics=n_components[type], max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(tf)
    
    t0 = time()
    lda.fit(tf)
    
    tf_feature_names = tf_vectorizer.get_feature_names()
    print_top_words("LDA",lda, tf_feature_names, n_top_words,n_clusters)
    print("done in %0.3fs." % (time() - t0))
    

    
    
    
    
    
    
#predicting topic from clusters
    
    ClusterTypes=["FN","KL","LDA"]
    for Type in range(0,3): 
        k=0
       
        #reading every cluster from selected cluster group
        for k in range(0,n_clusters):
            Unique_Words=[]
            initial=0
            WordFound=0
            count_t=0
            Distance_Count=[0 for j1 in range(30000)]
            taxonomy=[0 for j2 in range(30000)]
            List1=[]
            filename="Cluster"+ClusterTypes[Type]+str(n_clusters)+"-"+str(k)+".txt"
            with open(filename) as f:
                print(filename)
                for line in f:
                    is_noun = lambda pos: pos[:2] == 'NN'
                    # get the nouns
                    tokenized = nltk.word_tokenize(line)
                    Line_words = [word for (word, pos) in nltk.pos_tag(tokenized) if is_noun(pos)]
                    #Line_words = [word for (word, pos) in nltk.pos_tag(tokenized) if (is_noun(pos)or is_verb(pos)]
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
                                                 #calculating similarity
                                                  taxonomy_distance = Synset1[0].wup_similarity(Synset2[0])
                                                  Distance_Count[j]=Distance_Count[j]+taxonomy_distance
                                            
                                                  taxonomy[count_t]= taxonomy_distance
                                                  count_t+=1 
                                                  WordFound=1
                     
                         except:
      
                                        continue    
    
                     
                MaxIndices, value = max(enumerate(Distance_Count), key=operator.itemgetter(1))
                 #finding the topic with maximum similarity 
                PredictedTopics=Unique_Words[MaxIndices]
     
                Topic=Topic+"\t"+PredictedTopics
                temp=2000      
                WriteToFile(ClusterTypes[Type],n_clusters,temp,Topic)

