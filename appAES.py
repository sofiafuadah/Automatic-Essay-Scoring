import streamlit as st
import pandas as pd
import numpy as np
import json
import nltk
import re
import string
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

jawaban_siswa_df2 = pd.read_excel('data_Biologi.xlsx', sheet_name='jawaban') 
kunci_jawaban_df2 = pd.read_excel('data_Biologi.xlsx', sheet_name='soal')
kunci_jawaban_df3 = pd.read_excel('data_Biologi.xlsx', sheet_name='soal')

def preprocess(text):
    if not isinstance(text, str): #memastikan bahwa input yang diberikan berupa string
        return ''
    
    #case folding
    text = text.lower() #mengubah ke bentuk huruf kecil
    text = text.translate(str.maketrans('', '', string.punctuation)) #menghapus tanda baca
    text = re.sub(r'\d+', '', text) #menghapus angka
    
    #tokenization
    tokens = word_tokenize(text)
    
    #stopword removal 
    stopword_remover = StopWordRemoverFactory().get_stop_words()
    tokens = [word for word in tokens if word not in stopword_remover]
    
    #stemming
    stemmer = StemmerFactory().create_stemmer()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return ' '.join(stemmed_tokens)

for col in range(1, 21):
    jawaban_siswa_df2[f'prePro_J{col}'] = jawaban_siswa_df2[f'J{col}'].apply(preprocess)
kunci_jawaban_df2['prePro'] = kunci_jawaban_df2['kunci jawaban'].apply(preprocess)

def load(filename):
    with open(filename) as data_file:
        data = json.load(data_file)
    return data

mydict = load('dictIPA.json')

def get_synonyms(word):
    if word in mydict.keys():
        return mydict[word]['sinonim']
    else:
        return []

def query_expansion(kunci_jawaban, jawaban_siswa):
    expanded_kunci_jawaban = []
    kunci_jawaban = nltk.word_tokenize(kunci_jawaban)
    jawaban_siswa = nltk.word_tokenize(jawaban_siswa)
    for kunci in kunci_jawaban:
        sinonim = get_synonyms(kunci)
        found = False
        for sin in sinonim:
            if sin in jawaban_siswa:
                expanded_kunci_jawaban.append(sin)       
                found = True
                break
        if not found:
            expanded_kunci_jawaban.append(kunci)
    return expanded_kunci_jawaban

dataNew = pd.DataFrame(columns=['No', 'Siswa', 'KunciJawaban','PreJawabanSiswa','expended'])

kunci_jawaban_qe2 = kunci_jawaban_df2['prePro']
kunci_jawaban_qe3 = kunci_jawaban_df3['kunci jawaban']
nama_siswa=jawaban_siswa_df2['NAMA SISWA']
i=0
j=0
for col in range(0,36):
  for row in range(0,20):
    jawaban_siswa_qe2 = jawaban_siswa_df2[f'prePro_J{row+1}']
    expended =' '.join(query_expansion(kunci_jawaban_qe2[row], jawaban_siswa_qe2[col]))
    dataNew.loc[j]=[j+1,nama_siswa[col],kunci_jawaban_qe3[row],jawaban_siswa_qe2[col],expended]
    j+=1
dataNew

def tfidf(corpus):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    return X

tfidf_matrix_kunci = tfidf(dataNew['expended'])
tfidf_matrix_siswa = tfidf(dataNew['PreJawabanSiswa'])

#menghitung nilai similarity
def dice_similarity(str1, str2):
    # tokenization
    str1_tokens = set(word_tokenize(str1))
    str2_tokens = set(word_tokenize(str2))
    
    # similarity calculation
    intersection = len(str1_tokens.intersection(str2_tokens))
    dice_sim = 2 * intersection / (len(str1_tokens) + len(str2_tokens))
    
    return dice_sim

similarity_scores = []
for i, jawaban_siswa in enumerate(dataNew['PreJawabanSiswa']):
    dice_scores = []
    for j, kunci_jawaban in enumerate(dataNew['expended']):
        dice_scores.append(dice_similarity(jawaban_siswa, kunci_jawaban))
    similarity_scores.append(max(dice_scores))    

# Create a DataFrame from the numpy array and transpose it
dataNew['similarity_score'] = pd.DataFrame(similarity_scores)

dataNew2 = pd.DataFrame(columns=['No', 'Siswa', 'Scorer'])
pd.set_option('mode.chained_assignment', None)

data_similarity=dataNew['similarity_score']
nama_siswa=dataNew['Siswa']
j=0
k=1
tempData=0
for i in range (len(dataNew)):
  tempData+=data_similarity[i]
  if(k%20==0):
    dataNew2.loc[j]=[j+1,nama_siswa[k-2],tempData/20]
    tempData=0
    j+=1
  k+=1
dataNew2

def calculate_numeric_grade(x):
    if x >= 0 and x <= 0.05:
      return 5
    elif x >= 0.05 and x <= 0.1:
      return 10
    elif x >= 0.1 and x <= 0.15:
      return 15
    elif x >= 0.15 and x <= 0.2:
      return 20
    elif x >= 0.2 and x <= 0.25:
      return 25
    elif x >= 0.25 and x <= 0.3:
      return 30
    elif x >= 0.3 and x <= 0.35:
      return 35
    elif x >= 0.35 and x <= 0.4:
      return 40
    elif x >= 0.4 and x <= 0.45:
      return 45
    elif x >= 0.45 and x <= 0.5:
      return 50
    elif x >= 0.5 and x <= 0.55:
      return 55
    elif x >= 0.55 and x <= 0.6:
      return 60
    elif x >= 0.6 and x <= 0.65:
      return 65
    elif x >= 0.65 and x <= 0.7:
      return 70
    elif x >= 0.7 and x <= 0.75:
      return 75
    elif x >= 0.75 and x <= 0.8:
      return 80
    elif x >= 0.8 and x <= 0.85:
      return 85
    elif x >= 0.85 and x <= 0.9:
      return 90
    elif x >= 0.9 and x <= 0.95:
      return 95
    elif x >= 0.95 and x <= 1:
      return 100
    
dataSim=dataNew2['Scorer']
dataNew2['hasil_siswa']=0
data_total=dataNew2['hasil_siswa']
for col in range(len(dataNew2)):
  data_total[col]=calculate_numeric_grade(dataSim[col])

dataNew2

