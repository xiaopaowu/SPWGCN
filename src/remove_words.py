'''
Author: Wuyifan
Date: 2023-10-14 12:45:22
LastEditors: Wuyifan
LastEditTime: 2024-02-27 19:44:00
'''
from pydoc import doc
from nltk.corpus import stopwords
import nltk
from nltk.wsd import lesk
from nltk.corpus import wordnet as wn
import sys
import jieba
import jieba.posseg as pseg
import warnings
warnings.filterwarnings("ignore")
 
if len(sys.argv) != 3:
    sys.exit("Use: python build_graph.py <dataset>")

pro = sys.argv[1]
ratio = sys.argv[2]
    
dataset =pro + "_my_method_"+str(ratio)

doc_content_list = []
f = open('data/corpus/' + dataset + '.txt', encoding='utf-8')
for line in f.readlines():
    doc_content_list.append(line.strip())
f.close()

f_stopwords = open('stopwords1893.txt', encoding='utf-8')
stopwords = f_stopwords.read()
stopwords = stopwords.split('\n')

word_freq = {}  # to remove rare words
clean_docs = []
for doc_content in doc_content_list:
    doc_content = doc_content.strip()
    doc_content = doc_content.replace('\n', '')
    words =  jieba.lcut(doc_content, cut_all=False)
    doc_words = []
    for word in words:
        if word not in stopwords and word !=' ':
            doc_words.append(word)
    doc_str = ' '.join(doc_words).strip()
    clean_docs.append(doc_str)
clean_corpus_str = '\n'.join(clean_docs)

f = open('data/corpus/' + dataset + '.clean.txt', 'w')
f.write(clean_corpus_str)
f.close()
min_len = 10000
aver_len = 0
max_len = 0 
f = open('data/corpus/' + dataset + '.clean.txt', 'r')
lines = f.readlines()
for line in lines:
    line = line.strip()
    temp = line.split()
    aver_len = aver_len + len(temp)
    if len(temp) < min_len:
        min_len = len(temp)
    if len(temp) > max_len:
        max_len = len(temp)
f.close()
aver_len = 1.0 * aver_len / len(lines)
print('min_len : ' + str(min_len))
print('max_len : ' + str(max_len))
print('average_len : ' + str(aver_len))
