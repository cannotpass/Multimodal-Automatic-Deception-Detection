# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 16:16:08 2019

@author: Lukasz Kozarski
"""

# LOADING MODULES

from os import listdir
from os.path import isfile, join

import numpy as np
import pandas as pd
import nltk
import matplotlib.pyplot as plt
from nltk.tokenize import wordpunct_tokenize
from scipy.stats import shapiro
from scipy.stats import ttest_ind as ttest
from scipy.stats import mannwhitneyu as mwtest

'''
    Functions:
'''

def text_preprocess(text = '',
                    special_chars = '!\'\"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n'):
  '''
  Function takes two arguments:
      text:
          text to be preprocessed, in a form of one string
      special_chars:
          set of special characters to be removed from the text,
          as default a typical set of special characters is set
          
  Function returns:
      text:
          a string of initial text in lowercase, without the special characters
  '''  
  text = text.lower()
  text = ''.join(char for char in text if (
      char not in special_chars or char == ' '))
  return(text)

def num_of_words(list = [], part_of_speech = 'NOUN'):
  '''
  Function takes two arguments:
      list:
          list of words for the function to tag part of speech in
      part_of_speech:
          a name of part of speech to be tagged in the list of words
          
  Function returns:
      (part_of_speech, num):
          (a tuple of name of the part of speech that was tagged,
           number of cases of the part of speech in the initial list)
  '''  
  part_of_speech = part_of_speech.upper()
  num = 0
  for element in list:
    if element[0] == part_of_speech:
      num = element[1]
  return((part_of_speech, num))

def ave_len(tab = []):
  '''
  Function takes one argument:
      tab:
        list of iterable elements
  
  Function returns:
      x:
        average length of elements in the input
  '''
  x = 0
  for item in tab:
    x += len(item)
  x = x/len(tab)
  return(x)

'''
    Loading texts from files. Starting with deceptive and following
    with truthful.
'''

path = 'drive/My Drive/data/transcription/'

# LOADING DECEPTIVE

file_names = [i for i in listdir(
    join(path,'deceptive/')) if isfile(join(path, 'deceptive/', i))]

dec_texts = []

for name in file_names:
    tmp = join(path, 'deceptive/', name)
    with open(tmp, 'r',  encoding="utf8") as f:
        dec_texts.append(f.read())        

# LOADING TRUTHFUL
        
file_names = [i for i in listdir(
    join(path,'truthful/')) if isfile(join(path,'truthful/', i))]

tru_texts = []

for name in file_names:
    tmp = join(path, 'truthful/', name)
    with open(tmp, 'r',  encoding="utf8") as f
        tru_texts.append(f.read())
        
'''
    Creating one dataframe to keep all data used in analysis.
'''

data = pd.DataFrame(tru_texts, columns = ['original_text'])
data['deceptive'] = 0
data['truthful'] = 1

data_f = pd.DataFrame(dec_texts, columns = ['original_text'])
data_f['deceptive'] = 1
data_f['truthful'] = 0
data = data.append(data_f)
data = data.reset_index(drop = True)
del(data_f)

# COUNTING TYPES OF SPEECH

data['edited_text'] = data['original_text'].apply(
    lambda x : text_preprocess(x))

data['tokens'] = data['edited_text'].apply(
    lambda x : wordpunct_tokenize(x))

# Code adapted from nltk official examples:
# https://www.nltk.org/book/ch05.html

data['parts_of_speech'] = data['edited_text'].apply(
    lambda x : (nltk.FreqDist(
        tag for (word, tag) in nltk.pos_tag(
            wordpunct_tokenize(x), tagset="universal"))))

data['parts_of_speech'] = data['parts_of_speech'].apply(
    lambda x : x.most_common(12))
data['sum_of_tokens'] = data['tokens'].apply(lambda x : len(x))

# verb, noun, adjective, adverb, determiner/article, pronoun
# adposition/preposition or postposition/, conjunction
# punctuation, unknown/other, numerical, particle
parts_of_speech = ['verb', 'noun', 'adj', 'adv', 'det',
                   'pron', 'adp', 'conj', 'x', 'prt']

for part in parts_of_speech:
  data[part] = data['parts_of_speech'].apply(
      lambda x : num_of_words(x, part.upper())[1])
  
  data[part] = data[part]/data['sum_of_tokens']
  
# TESTING FOR NORMALITY TO COMAPRE MEANS

CHOSEN_COLUMNS = ['verb', 'noun', 'adj', 'adv', 'det',
                   'pron', 'adp', 'conj', 'prt']

normally_distributed = []
non_normally_distributed = []

# setting up plots
NUM_OF_HISTS = len(CHOSEN_COLUMNS)
NUM_OF_ROWS = round(NUM_OF_HISTS/2, 0)
NUM_OF_COLUMNS = 3

fig = plt.figure(figsize=(14,14))

for i in range(0, 9):
  name = CHOSEN_COLUMNS[i]
  
  shapiro_test = shapiro(data[name])
  shapiro_test_t = shapiro(data[data['truthful']==1][name])
  shapiro_test_d = shapiro(data[data['deceptive']==1][name])
  
  # fragment below prints histograms, now turned off for space reasons
  
  fig = plt.subplot(NUM_OF_ROWS, NUM_OF_COLUMNS, i+1)
  plt.title(name)
  plt.hist(data[data['truthful'] == 1][name], color = 'green', alpha = 0.5)
  plt.hist(data[data['deceptive'] == 1][name], color = 'red', alpha = 0.5)
  plt.subplots_adjust(bottom=1, top=2)
  
  if (shapiro_test_f[1] < 0.05) or (shapiro_test_d[1] < 0.05):
    non_normally_distributed.append(name)
  else:
    normally_distributed.append(name)
  print
  print('{:6s}overall: {:5.3f}  truthful: {:5.3f}  deceptive: {:.3f}'.format(
      name.upper(), shapiro_test[1], shapiro_test_t[1], shapiro_test_d[1]))  

print()
print('Normally distributed (combined):     ', normally_distributed)
print('Non-normally distributed (combined): ', non_normally_distributed)

# TOKENS LENGHT

data['ave_token_len'] = data['tokens'].apply(lambda x : ave_len(x))

# t-test for length

plt.title('Average token length')
plt.hist(data[data['truthful'] == 1]['ave_token_len'],
         color = 'green', alpha = 0.5)
plt.hist(data[data['deceptive'] == 1]['ave_token_len'],
         color = 'red', alpha = 0.5)
plt.hist(data['ave_token_len'], color = 'black', alpha = 0.2)
plt.show()

print('Test for normal distribution:\n')
for i in [0,1]:
  print('For deceptive =', i, '  p-value = {:.3f}'.format(
      shapiro(data[data['deceptive'] == i]['ave_token_len'])[1]))
print('Combined            p-value = {:.3f}'.format(
    shapiro(data['ave_token_len'])[1]))
    
# mann whitney u-test for tokens length
print('\nTesting for equal means of the deceptive and truthful tokens length:\n\nMW-test p-value: {:.3f}\n'.format(mwtest(data[data['truthful'] == 1]['ave_token_len'], data[data['truthful'] == 0]['ave_token_len'])[1]))

print('t-tests for normally distributed parts of speech:\n')

# t-test for parts of speech
for part in normally_distributed:
  print('{:<5}'.format(part),
        'p-value:',
        " {:.03f}".format(ttest(data[data['truthful'] == 1][part],
                                data[data['truthful'] == 0][part],
                                equal_var = False,
                                nan_policy = 'omit'
                               )[1]
                         )
       )

# null-hypothesis for the MW test is that the distributions are the same,
# so if p<0.05 reject the hypothesis

print('\nMann-Whitney U-test for non-normally distributed parts of speech:\n')

for part in non_normally_distributed:
  print('{:<5}'.format(part),
        'p-value:',
        " {:.03f}".format(mwtest(data[data['truthful'] == 1][part],
                                 data[data['truthful'] == 0][part]
                                )[1]
                         )
       )
