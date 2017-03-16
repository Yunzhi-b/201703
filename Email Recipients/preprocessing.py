import pandas as pd
import re
import numpy as np
import operator
import math
from collections import Counter 
from itertools import islice,izip
from itertools import chain
from nltk.corpus import wordnet as wn
import nltk
import enchant
import difflib
#nltk.download()

# get the train data and the test data from files
def getdatas(train_fname= 'train.csv',test_fname= 'test.csv'):
    X = []
    y = []
    with open(train_fname) as f:
        for line in f:
            y.append(int(line[0]))
            X.append(line[5:-6])
    y = np.array(y)
    
    X_test = []
    with open(test_fname) as f1:
        for line in f1:
            X_test.append(line[3:-6])

    return X, y, X_test


#this fonction will split text to words in a 2-dimension list, and transform uppercase letter to lowercase letter 
def reformWithSplitOfWords(X):
    X2=[]
    for line in X:
        s=re.sub('[,;\.!"\?\(\)\']','',line).lower()
        X2.append(s.split())
    return X2

#this function delete meaningless words or symbols 
def deleteRubbish(X):
    for i in np.arange(len(X)):
        for j in np.arange(len(X[i])):
            X[i][j]=re.sub('^((\w*)[^(\w|\s)](\w*))+$','',X[i][j])
    X1=[]
    for line in X:
        X1.append(filter(lambda x: len(x)>0, line)) 
    return X1

#this function delete the words of high frequence 
def removeWordsWithHF(X,nbToRemove):
    iList = nbToRemove
    count={}
    for line in X:
        for word in line:
            if count.has_key(word):
                count[word] = count[word] + 1
            else:
                count[word] = 1
    WordsToRemove=[i[0] for i in sorted(count.iteritems(), key=operator.itemgetter(1), reverse=True)[0:iList]]
    
    X1=[]
    for line in X:
        X1.append(filter(lambda x: x not in WordsToRemove, line))
        
    return WordsToRemove,X1


# this can remove empty lines in X after removing words. (useless)
def removeTheNullLine(X):
    return filter(lambda x: len(x)>0,X)


# return a list of words apparing in training set
def listOfWords(X):
    words=[]
    for line in X:
        for word in line:
            if word not in words:
                words.append(word)
    return words, len(words)

# return a dictonary of words in form of {word: number of documents containing this word + 1}
def IDF(X,words):
    IDFs={}
    n=float(len(X))
    for word in words:
        IDFs[word]=1
        
    for line in X:
        for word in words:
            if word in line:
                IDFs[word]=IDFs[word]+1
    
    for word in words:
        IDFs[word]=math.log(n/IDFs[word])    
    return IDFs

# return a line of Tf-idf. It's of shape (1,m) with m the number of words we use.
def TFIDF_oneLine(line,words,IDFs):
    tfidfs=[]
    length=float(len(line))+1.
    for word in words:
        tfidfs.append(line.count(word)/length*IDFs[word])
    return tfidfs

# return a matrix of Tf-idfs of shape (n,m). with n number of sample, m the number of words we use.
def matrixOfTfidf(X,words):
    idfs=IDF(X,words)
    result=[]
    for line in X:
        result.append(TFIDF_oneLine(line,words,idfs))
    return result


# a version intermediate
# this function is a combinaison of above functions. So just use this function to finish the preprocessing.
# this function will return a matrix of Tf-idfs of shape (n,m). with n number of sample, m the number of words we use.
def preprocessing(X,nbToRemove=50):
    X1=reformWithSplitOfWords(X)
    X2=deleteRubbish(X1)
    wordsToRemove,X3=removeWordsWithHF(X2,nbToRemove)
    result=matrixOfTfidf(X3,listOfWords(X3)[0])
    return wordsToRemove,result

# Correct the spell. "performe" will be corrected to "perform"
def correctOneWord(word):
    d = enchant.Dict("en_US")
    if len(word)==0:
        return word
    if not d.check(word):
        dict1,max1 = {},0
        a = set(d.suggest(word))
        if len(a)==0:
            return word
        for b in a:
            tmp = difflib.SequenceMatcher(None, word, b).ratio();
            dict1[tmp] = b
            if tmp > max1:
                max1 = tmp
        return dict1[max1]
    else:
        return word

# for exemple: for "started", "starting", we we transform them to "start". It means delete the conjugaisons.
def MorphyOneWord(x):
    output = wn.morphy(x)
    if output is None:
        return x
    else:
        return str(output)

# do correct spell for all the data
def correctSpelling(X):
    X1=[]
    i=0
    for line in X:
        if i==592 or i==592+4415:
            X1.append("")
            continue
        elif len(line)==0:
            X1.append("")
        else:
            temp=[]
            for word in line:
                temp.append(correctOneWord(word))
            X1.append(temp)
        i=i+1
#        print i
    return X1

# do morphy for all the data
def morphy(X):
    X1=[]
    i=0
    for line in X:
        if len(line)==0:
            X1.append("")
        else:     
            temp=[]
            for word in line:
                temp.append(MorphyOneWord(word))
            X1.append(temp)
        i=i+1
#        print i
    return X1

# We have a list of usually used words, like 'you', 'and', we delete them 
def deleteStopWords(X):
    stopwords = nltk.corpus.stopwords.words('english')
    X1=[]
    for line in X:
        X1.append(filter(lambda x: x not in stopwords, line))
    return X1

# remove the words with bass frequence
def removeWordsWithBF(X,seuil):
    count={}
    for line in X:
        for word in line:
            if count.has_key(word):
                count[word] = count[word] + 1
            else:
                count[word] = 1
    
    WordsToRemove=[]
    for word in count.keys():
        if count[word] <= seuil:
            WordsToRemove.append(word)
    
    X1=[]
    for line in X:
        X1.append(filter(lambda x: x not in WordsToRemove, line))
        
    return WordsToRemove,X1

def bigrams_total(text,n_bigrams=100):
    if  np.array(text).dtype !='O':  
        words = text
    else:
        words = list(chain.from_iterable(text))
        #words = list(text[i] for i in range(np.size(text)))
    list_bigram = Counter(izip(words,islice(words,1,None))).most_common(n_bigrams)
    
    all_biwords = []
    all_freq = []
    for u,v in list_bigram:
        all_biwords.append(u)
        all_freq.append(v)
    return all_biwords, all_freq 

def TFIDF_bigram(X,words):
    IDFs={}
    count={}
    length = []
    result = np.zeros((np.size(X),np.shape(words)[0]))
    #print np.size(X),np.shape(words)[0]

    n=float(len(X))
    for bi_word in words:
        IDFs[bi_word]=1
  
    for line in X:
        length.append(float(len(line)+1))
        for bi_word in words:
            count.setdefault(bi_word, [])
            if  (bi_word[0] not in line) or(bi_word[1] not in line): # none of the 'w1','w2' in bi_word appears in the current line
                count[bi_word].append(0)
                continue
            else:
                # if both w1 and w2 appear in the current line i, then check whether the bigram('w1','w2') exists in li
                line_all_biwords, line_all_freq = bigrams_total(line) 
                if bi_word in line_all_biwords:
                    count[bi_word].append(line_all_biwords.count(bi_word))
                    IDFs[bi_word]=IDFs[bi_word] + 1
                else:
                    count[bi_word].append(0)
    j =0
    for bi_word in words:
        IDFs[bi_word]=math.log(n/IDFs[bi_word])   
        for i in range(np.size(X)):
            result[i,j] = float(count[bi_word][i]/length[i]*IDFs[bi_word])
        j = j+1 #next bi_word
         
    return result


"""
Input: 
    X_train, n_bigrams 
    where **** wordsToRemove, X_train = preprocessing(X)***
    
Output: 
    result: 
    matrix tfidf des bigrams = (n_commentaire, n_bigrams)  
"""
def matrix_bigram(X_train,X_test,n_bigrams=500):
    all_biwords, all_freq =  bigrams_total(np.array(X_train),n_bigrams)
    matrix_train = TFIDF_bigram(X_train,all_biwords)
    matrix_test = TFIDF_bigram(X_test,all_biwords)
    return  matrix_train,matrix_test

# little change, useful in preprocessing_final_bis
def matrix_bigram_bis(X_train,n_bigrams=500):
    all_biwords, all_freq =  bigrams_total(np.array(X_train),n_bigrams)
    matrix_train = TFIDF_bigram(X_train,all_biwords)
    return  matrix_train


# this function is a combinaison of above functions. So just use this function to finish the preprocessing.
# this function will return a matrix of Tf-idfs of shape (n,m). with n number of sample, m the number of words we use.
import time
def preprocessing_final(train,test,nbToRemove=50,seuil=2,delStopWords=True,corSpell=True,morphy_=True,bigram=True,n_bigrams=500):
    train1=reformWithSplitOfWords(train)
    train2=deleteRubbish(train1)
    train3=removeWordsWithBF(train2,seuil)[1]


    test1=reformWithSplitOfWords(test)
    test2=deleteRubbish(test1)
    test3=removeWordsWithBF(test2,seuil)[1]

    if corSpell:
        train4=correctSpelling(train3)
        test4=correctSpelling(test3)
    if morphy_:
        if corSpell:
            train5=morphy(train4)
            test5=morphy(test4)
        else:
            train5=morphy(train3)
            test5=morphy(test3)
    else:
        if corSpell:
            train5=train4
            test5=test4
        else:
            train5=train3
            test5=test3

    wordsRemoved,train6=removeWordsWithHF(train5,nbToRemove)
    if delStopWords:
        train7=deleteStopWords(train6)
    else:
        train7=train6
    
    train_new1=matrixOfTfidf(train7,listOfWords(train7)[0])
    test_new1=matrixOfTfidf(test5,listOfWords(train7)[0])
    
    if bigram==True:
        matrix_train,matrix_test=matrix_bigram(train7,test5,n_bigrams=n_bigrams)
        train_new=np.concatenate((train_new1,matrix_train),axis=1)
        test_new=np.concatenate((test_new1,matrix_test),axis=1)
    else:
        train_new=train_new1
        test_new=test_new1
    
    return wordsRemoved,train_new,test_new


# this function is a combinaison of above functions. So just use this function to finish the preprocessing.
# this function will return a matrix of Tf-idfs of shape (n,m). with n number of sample, m the number of words we use.
# Different from preprocessing_final, in this method, we do the traitements on the ensemble of test and train datas. For exemple, when we count the frequence of a word, we consider both of test and train datas.
def preprocessing_final_bis(train,test,nbToRemove=50,seuil=2,delStopWords=True,corSpell=True,morphy_=True,bigram=True,n_bigrams=500):
    
    donnees=train+test
    train1=reformWithSplitOfWords(donnees)
    train3=deleteRubbish(train1)
    
    if corSpell:
        train4=correctSpelling(train3)
    if morphy_:
        if corSpell:
            train5=morphy(train4)
        else:
            train5=morphy(train3)
    else:
        if corSpell:
            train5=train4
        else:
            train5=train3

    print("3333333333")
    
    train5=removeWordsWithBF(train5,seuil)[1]
    wordsRemoved,train6=removeWordsWithHF(train5,nbToRemove)
    if delStopWords:
        train7=deleteStopWords(train6)
    else:
        train7=train6
    
    train_new1=matrixOfTfidf(train7,listOfWords(train7)[0])
    
    if bigram==True:
        matrix_train=matrix_bigram_bis(train7,n_bigrams=n_bigrams)
        train_new=np.concatenate((train_new1,matrix_train),axis=1)
    else:
        train_new=train_new1
    
    return wordsRemoved,train_new[:4415],train_new[4415:]
    
