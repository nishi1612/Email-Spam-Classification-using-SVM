import os
import nltk
import time
import string
import operator
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

def text_cleanup(text):
    text_without_punctuation = [c for c in text if c not in string.punctuation]
    text_without_punctuation = ''.join(text_without_punctuation)
    text_without_stopwords = [word for word in text_without_punctuation.split() if word.lower() not in stopwords.words('english')]
    text_without_stopwords = ' '.join(text_without_stopwords)
    cleaned_text = [word.lower() for word in text_without_stopwords.split()]
    return cleaned_text

start_time = time.time()

lmtzr = WordNetLemmatizer()
k=0
count = {}

directory_in_str = "emails/"
directory = os.fsencode(directory_in_str)

for file in os.listdir(directory):
    file = file.decode("utf-8")
    file_name = str(os.getcwd()) + '/emails/'
    file_name = file_name + file
    file_reading = open(file_name,"r",encoding='utf-8', errors='ignore')
    words = text_cleanup(file_reading.read())
    for word in words:
        if (word.isdigit()==False and len(word)>2):
            word = lmtzr.lemmatize(word)
            if word in count:
                count[word] += 1
            else:
                count[word] = 1
    k+=1
    if(k%100==0):
        print("Done " + str(k))

sorted_count = sorted(count.items(),key=operator.itemgetter(1),reverse=True)
sorted_count = dict(sorted_count)

f= open("wordslist.csv","w+")
f.write('word,count')
f.write('\n')
for word , times in sorted_count.items():
    if times < 100:
        break
    f.write(str(word) + ',' + str(times))
    f.write('\n')
f.close()

print('Time (in seconds) to pre process the emails ' + str(round(time.time() - start_time,2)))
