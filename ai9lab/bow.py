# Python3 code for preprocessing text 
import nltk 
import re 
import numpy as np 

# execute the text here as : 
text = """ my name is mehreen""" 
dataset = nltk.sent_tokenize(text) 
for i in range(len(dataset)): 
	dataset[i] = dataset[i].lower() 
	dataset[i] = re.sub(r'\W', ' ', dataset[i]) 
	dataset[i] = re.sub(r'\s+', ' ', dataset[i]) 
