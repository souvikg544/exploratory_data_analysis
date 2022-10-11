import string 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import pickle
from sqlalchemy import column

df = pd.read_csv("spam.csv", encoding= "ISO-8859-1")   # read data set 

df=df[['v1','v2']]
df.rename(columns={'v1':'spam','v2':'text'},inplace=True)

df.spam=df.spam.apply(lambda s: True if s=='spam' else False)

df.text= df.text.apply(lambda t: t.lower().translate(str.maketrans('','',string.punctuation)))    # remove punctuation  (but not understood by me yet)

df = df.sample(frac=1)


# print(df)
dftrain= df.iloc[:int(len(df)*0.8)]
dftest= df.iloc[int(len(df)*0.8):]

ratio_spam= dftrain.spam.mean()
# print(ratio_spam)

trainspamword= ' '.join(dftrain[dftrain.spam==True].text).split(' ')

trainnonspamword= ' '.join(dftrain[dftrain.spam==False].text).split(' ')

commonword= set(trainspamword).intersection(set(trainnonspamword))



trainspambag= dict()
for w in commonword:
    trainspambag[w]= trainspamword.count(w)/len(trainspamword)
    
trainnonspambag= dict()
for w in commonword:
    trainnonspambag[w]= trainnonspamword.count(w)/len(trainnonspamword)
    
    
# print(trainspambag)

def predict(t,verbose=False):
    req= [w for w in t.split(' ') if w in trainspambag]
    
    spamprob= [trainspambag[w] for w in req]
    nonspamprob= [trainnonspambag[w] for w in req]
    
    spamscore=sum([np.log(p) for p in spamprob])+np.log(ratio_spam)
    nonspamscore=sum([np.log(p) for p in nonspamprob])+np.log(1-ratio_spam)
    
    return (spamscore>=nonspamscore)
    




