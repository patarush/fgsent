# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 23:15:48 2022

@author: Rushil
"""
import pickle
from pathlib import Path
import streamlit_authenticator as stauth
from bs4 import BeautifulSoup
import requests
import pandas as pd
import streamlit as st
import numpy as np
import time
from PIL import Image
import json
from streamlit_lottie import st_lottie
def load_lottieurl(url:str):
     r=requests.get(url)
     if r.status_code !=200:
         return None
     return r.json()

hello=load_lottieurl('https://assets3.lottiefiles.com/packages/lf20_dhlmjljy.json')
#https://assets6.lottiefiles.com/packages/lf20_fi2zcy9b.json
#https://assets5.lottiefiles.com/packages/lf20_OANz0r.json

st.header('Google Sentimental analysis')
#st.write('')
#st.write('')

col1,col2 = st.columns(2)

with col2:    
    st.write('')
    st.text('Search for your interested topic ') 
    st.text('We will analyse the Sentiment of the text')
    st.text('based on the top 10 google search results')
with col1:
    
    st.write('')
    st_lottie(hello,key='hi')

st.write('')
st.write('')


# Web Scrapping

## Scrapping Url

#image=Image.open('background.jpg')
#st.image(image)

#st.write('we will do the sentimental analysis of the top 10 Google Search results')

x = st.text_input('Google Search')
xx=str(x)
print(xx)
url = "https://www.google.com/search?q="+ xx  
print(url)
req = requests.get(url)
soup = BeautifulSoup(req.text, 'html.parser')
l = []
for link in soup.find_all('a', href=True):
    l.append(link['href'])
print(l)

## Url Cleaning


# Slicing Urls into useful part
final_list = []
for a in l:
    if '/url?q=' in a:
        final_list.append(a.replace('/url?q=', ''))

a = []
for i in final_list:
    # print(i.index('&sa'))
    a.append(i[:i.index('&sa')])

# Removing Duplicates
a = list(dict.fromkeys(a))

# Removing not useful website
google_domains = ('https://www.google.',
                  'https://google.',
                  'https://webcache.googleusercontent.',
                  'http://webcache.googleusercontent.',
                  'https://policies.google.',
                  'https://support.google.',
                  'https://maps.google.',
                  'https://www.tiktok',
                  'https://twitter.com',
                  'https://www.instagram.com',
                  'https://www.youtube.com',
                  'https://accounts.google.com',
                  'https://m.youtube.com'
                  )
for url in a[:]:
    if url.startswith(google_domains):
        a.remove(url)

# Getting Text from Website
j = []
for i in a[:11]:
    # print(i)

    try:
        req_result = requests.get(i)
        soup = BeautifulSoup(req_result.text, 'html.parser')
        j.append(soup.text)
    except:
        j.append('NA')

my_dict = {'link': a[:11], 'Text': j}
df = pd.DataFrame(my_dict)

# df1.to_csv('te.csv')
#df

#df = df[df["Text"].str.contains("Denied") == False]

#df['link'][10]

# Text Cleaning

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from textblob import TextBlob
import matplotlib.pyplot as plt
from wordcloud import WordCloud

final_text = list(df['Text'])
#final_text

# 1 Lowering the list
clean_text = []


def to_lower(data):
    for word in final_text:
        clean_text.append(str.lower(word))


to_lower(final_text)

# 2 Word tokenize

clean_text_2 = [word_tokenize(i) for i in clean_text]
#clean_text_2

import re

clean_text_3 = []
for words in clean_text_2:
    clean = []
    for w in words:
        res = re.sub(r'[^\w\s]', "", w)
        if res != '':
            clean.append(res)
    clean_text_3.append(clean)
#clean_text_3

# 4 remove stop words
from nltk.corpus import stopwords

clean_text_4 = []
for words in clean_text_3:
    w = []
    for word in words:
        if not word in stopwords.words('english'):
            w.append(word)
    clean_text_4.append(w)
#clean_text_4

# lemmatization -> form of stemming but makes sure output makes sense
from nltk.stem.wordnet import WordNetLemmatizer

wnet = WordNetLemmatizer()
clean_text_5 = []
for words in clean_text_4:
    w = []
    for word in words:
        w.append(wnet.lemmatize(word))
    clean_text_5.append(w)
#clean_text_5

dff = pd.DataFrame(clean_text_5)
dff['ColumnA'] = dff[dff.columns[0:]].apply(
    lambda x: ' '.join(x.dropna().astype(str)),
    axis=1
)
#dff

dff['ColumnA'].replace(',', ' ', inplace=True)


# Sentiment Analysis

def get_subjectivity(text):
    return TextBlob(text).sentiment.subjectivity


def get_polarity(text):
    return TextBlob(text).sentiment.polarity


dff['Subjectivity'] = dff['ColumnA'].apply(get_subjectivity)
dff['Polarity'] = dff['ColumnA'].apply(get_polarity)

dff = dff.iloc[:, -3:]
#dff




def get_analysis(score):
    if score > 0.05:
        return 'Positive'
    elif score == 0.0:
        return 'Neutral'
    else:
        return 'Negative'


dff['Analysis'] = dff['Polarity'].apply(get_analysis)
#dff

plt.style.use('ggplot')

import matplotlib.pyplot as plt

# Subjectivity VS polarity
plt.figure(figsize=(8, 6))
for i in range(0, dff['Polarity'].shape[0]):
    plt.scatter(dff['Polarity'][i], dff['Subjectivity'][i], color='blue')
plt.title('Subjectivity Vs Polarity')
plt.xlabel('Polarity')
plt.ylabel('Subjectivity')
plt.show()


###################################
st.write('')
st.write('')
st.subheader('SENTIMENTAL ANALYSIS')
chart=load_lottieurl('https://assets5.lottiefiles.com/private_files/lf30_ps1145pz.json')
st_lottie(chart,key='chart')
st.write('')
st.write('Based on your search we got folllowing result')

####################################################
# percent of posiive
pos = dff[dff['Analysis'] == 'Positive']
#print(round(pos.shape[0] / dff.shape[0] * 100, 2))
st.caption("Positive Percent:")
st.write(round(pos.shape[0] / dff.shape[0] * 100, 2))

# percent of Negative
neg = dff[dff['Analysis'] == 'Negative']
#print(round(neg.shape[0] / dff.shape[0] * 100, 2))
st.caption("Negative Percent:")
st.write(round(neg.shape[0] / dff.shape[0] * 100, 2))

nut = dff[dff['Analysis'] == 'Neutral']
#print(round(nut.shape[0] / dff.shape[0] * 100, 2))
st.caption("Neutral Percent:")
st.write(round(nut.shape[0] / dff.shape[0] * 100, 2))

########## Word Cloud ################

plt.figure(figsize=(8, 6))
allwords = ' '.join([col for col in dff['ColumnA']])
wordcloud = WordCloud(width=500, height=300, random_state=21, max_font_size=120).generate(allwords)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

st.write('')
st.subheader('WORD CLOUD')
st.write('')
fig, ax = plt.subplots(figsize = (12, 8))
ax.imshow(wordcloud)
plt.axis("off")
st.pyplot(fig)



################ BAR CHART ######################


plt.figure(figsize=(12,8))
plt.title('Sentimental Analysis')
plt.xlabel('Sentiment')
plt.ylabel('Count')
bar=dff['Analysis'].value_counts().plot(kind='bar', color='pink')
plt.show()

st.write('')
st.subheader('BAR CHART')
st.write('')
st.bar_chart(dff['Analysis'].value_counts())




dff['Link'] = df['link']
#dff.columns

dff.rename(columns={'ColumnA': 'Text'}, inplace=True)
dff['Text']=j


#Streamlit

st.write('')
st.subheader('DATA SCRAPPED')
st.write(dff)
st.write('')


##################
st.write('')
st.write('')
st.subheader('WORKING OF THE PROJECT')
st.write('')
st.write('')
st.subheader('WEB SCRAPPING')
st.write('')
st.write('Web scraping, web harvesting, or web data extraction is data scraping used for extracting data from websites. The web scraping software may directly access the World Wide Web using the Hypertext Transfer Protocol or a web browser. While web scraping can be done manually by a software user, the term typically refers to automated processes implemented using a bot or web crawler. It is a form of copying in which specific data is gathered and copied from the web, typically into a central local database or spreadsheet, for later retrieval or analysis.')
st.write('')
st.write('We scrape the Google search result of the user using Beautiful Soup , which is a python library for Web Scrapping.')
st.write('')
st.subheader('DATA CLEANING')
st.write('')
st.write('Firstly we Remove urls where access is denied (eg. Youtube,Instagram and others)')
st.write('Once we get valid Urls , we scrape the text inside those urls and we perform :')
st.write('')
st.write('1 Lowering the text')
st.write('2 Tokenization (Word or Sentence)')
st.write('3 Removing  [^\w\s]')
st.write('4 Removing Stop Words')
st.write('5 Stemming / Lemmatization')
st.write('')
st.subheader('SENTIMENTAL ANALYSIS')
st.write('')
st.write('Sentiment analysis (also known as opinion mining or emotion AI) is the use of natural language processing, text analysis, computational linguistics, and biometrics to systematically identify, extract, quantify, and study affective states and subjective information. Sentiment analysis is widely applied to voice of the customer materials such as reviews and survey responses, online and social media, and healthcare materials for applications that range from marketing to customer service to clinical medicine.')
st.write('')
st.write('We find Subjectivity and Polarity of our search results and we analyse our result based on polarity')
st.write('')
st.write('We have taken the polarity >0.05(Positive) , polarity<0.05(Negative) , polarity =0.0(Neutral(for access denied))')
st.write('After that we plot the desired analysis.')
st.write('')
panda=load_lottieurl('https://assets4.lottiefiles.com/packages/lf20_0fh7phym.json')
#https://assets8.lottiefiles.com/packages/lf20_kd5rzej5.json
st_lottie(panda,key='bye')
st.write('')
st.write('Hope you liked this project !!!')
#print(dff)
