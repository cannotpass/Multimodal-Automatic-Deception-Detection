# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 16:16:08 2019
@author: Lukasz Kozarski
"""

# MODULES and DOWNLOADS

from os import listdir
from os.path import isfile, join

from random import shuffle

import numpy as np
from numpy.random import seed
import pandas as pd

import nltk
from nltk import FreqDist
from nltk.tokenize import wordpunct_tokenize
from nltk.corpus import stopwords
import re

from sklearn.model_selection import GroupKFold
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras import Sequential
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Dropout
from keras.models import Model
from keras.initializers import Constant, RandomNormal
from keras.callbacks import ModelCheckpoint

from scipy.sparse import hstack
from scipy.sparse import csr_matrix

nltk.download('stopwords')

'''
    ADDITIONAL FUNCTIONS:
'''

def remove_words(_input, wordset):
  return ' '.join([w for w in _input.split() if not w in wordset])

def optimize_model(X, y, groups, MODEL, n_splits, parameters):
  param_grid = ParameterGrid(parameters)
  best_score = 0
  best_params = dict()
  i = 1
  for param_set in param_grid:
    fold_score = 0
    for train, test in GroupKFold(n_splits = n_splits).split(
            X, y, groups = groups):
      # SHUFFLING INDICES
      shuffle(train)
      shuffle(test)
      # referring to inputs and outputs: X(train), y(train), X(test), y(test)
      mod = MODEL(**param_set).fit(X[train], y[train])
      predicted = mod.predict(X[test])
      fold_score += accuracy_score(y[test], predicted)
    i+=1
    if fold_score/n_splits > best_score:
      best_score = fold_score/n_splits
      best_params = param_set
  return(best_score, best_params)

def chose_best_input(X, y, groups, MODEL, n_splits, parameters):
  best_score = 0
  best_params = dict()
  ind, j = 0, 0
  for _input in X:
    set_score, set_params = optimize_model(
            _input, y, groups, MODEL, n_splits, parameters)
    if set_score > best_score:
      best_score, best_params = set_score, set_params
    else:
      ind = j
    j += 1
  return(j, best_score, best_params)

def show_all_sets(X, y, groups, MODEL, n_splits, parameters):
  scores = []
  params = []
  sets = []
  num_of_sets = len(X)
  for i in range(len(X)):
    set_score, set_params = optimize_model(X[i], y, groups,
                                           MODEL, n_splits, parameters)
    scores.append(set_score)
    params.append(set_params)
    sets.append(i+1)
    num_of_sets = i+1
  return num_of_sets, sets, scores, params
  
def optimize_auc(X, y, groups, MODEL, n_splits, parameters):
  param_grid = ParameterGrid(parameters)
  best_score = 0
  best_params = dict()
  i = 1
  for param_set in param_grid:
    fold_score = 0
    for train, test in GroupKFold(n_splits = n_splits).split(X, y,
                                 groups = groups):
      # SHUFFLING INDICES
      shuffle(train)
      shuffle(test)
      # referring to inputs and outputs: X(train), y(train), X(test), y(test)
      mod = MODEL(**param_set).fit(X[train], y[train])
      
      preds = mod.predict_proba(X[test])
      pred = [i[1] for i in preds]
      
      fold_score += metrics.roc_auc_score(y[test], pred)
      
    i+=1
    if fold_score/n_splits > best_score:
      best_score = fold_score/n_splits
      best_params = param_set
  return(best_score, best_params)

def chose_best_auc(X, y, groups, MODEL, n_splits, parameters):
  best_score = 0
  best_params = dict()
  ind, j = 0, 0
  for _input in X:
    set_score, set_params = optimize_auc(_input, y, groups,
                                         MODEL, n_splits, parameters)
    if set_score > best_score:
      best_score, best_params = set_score, set_params
    else:
      ind = j
    j += 1
  return(j, best_score, best_params)

def show_all_auc(X, y, groups, MODEL, n_splits, parameters):
  scores = []
  params = []
  sets = []
  num_of_sets = len(X)
  for i in range(len(X)):
    set_score, set_params = optimize_auc(X[i], y, groups, MODEL,
                                         n_splits, parameters)
    scores.append(set_score)
    params.append(set_params)
    sets.append(i+1)
    num_of_sets = i+1
  return num_of_sets, sets, scores, params

def preprocess(text):
    text = str(text)
    text = re.sub(r"(\nhttps?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]\
                     [a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9]\
                     [a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|\
                     (?!www))[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9]\.\
                     [^\s]{2,})", "url", text)
    text = re.sub(r'([^\s\w]|_)+', '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'U. S.', 'U.S.', text)
    text = re.sub(r'[0-9]+', '', text)
    text = text.lower()
    text = ' '.join(text.split())
    return text
    

'''
    DATASET IMPORT:
    
    Loading texts from files. Starting with deceptive and following
    with truthful.
'''

path = './court_trials_dataset/transcription/'

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
    with open(tmp, 'r',  encoding="utf8") as f:
        tru_texts.append(f.read())
        
trial = pd.DataFrame(tru_texts, columns = ['original_text'])
trial['deceptive'] = 0

trial_f = pd.DataFrame(dec_texts, columns = ['original_text'])
trial_f['deceptive'] = 1

# IMPORTING SUBJECT CODE (GROUPS FOR CROSS-VALIDATION)

tmp = join(path, 'tru_groups.csv')
tru_groups = pd.read_csv(tmp)
trial['group'] = tru_groups['group']

tmp = join(path, 'dec_groups.csv')
dec_groups = pd.read_csv(tmp)
trial_f['group'] = dec_groups['group']

# MERGING TRUTHFUL AND DECEPTIVE DATAFRAMES

trial = trial.append(trial_f, sort = False)
trial = trial.reset_index(drop = True)

del(trial_f, dec_groups, tru_groups)
     
'''
    OPERATIONS ON THE DATASET:
'''
# PREPROCESSING

trial['input'] = trial['original_text'].apply(preprocess)

trial['len'] = trial['input'].apply(lambda x : len(x.split()))
trial = trial[trial['len']<140]
trial.reset_index(inplace = True, drop = True)

# REMOVE STOPWORDS
# stop_words = list of words to be removed
stop_words = set(stopwords.words('english'))

trial['no_stopwords'] = trial['input'].apply(
        lambda x : remove_words(x, stop_words))

# REMOVE THE MOST COMMON WORDS - NOISE
# NUMB = number of most freq words to be removed
NUMB = 10

texts = ' '.join(trial['no_stopwords'])
wordset = dict(FreqDist(texts.split()).most_common(NUMB)).keys()

trial['less_noise'] = trial['no_stopwords'].apply(
        lambda x : remove_words(x, wordset))


'''
    scikit MLPerceptron
'''
# MultiLayer Perceptron Parameters

from sklearn.neural_network import MLPClassifier
modelsX = [('Multi-Layer Perceptron',
           MLPClassifier,
           {'hidden_layer_sizes': [(150, 512)],
            'activation': ['relu'],
            'solver': ['adam'],
            'alpha': [1e-4],
            'batch_size': [32],
            'random_state': [940321],
           }
           )
          ]

tfidf = TfidfVectorizer()
X_1 = tfidf.fit_transform(trial['input'])
X_2 = tfidf.fit_transform(trial['no_stopwords'])
X_3 = tfidf.fit_transform(trial['less_noise'])
X = [X_1, X_2, X_3]
y = trial['deceptive']
groups = trial['group']
n_splits = 10

num_of_sets = len(X)

for MODEL in modelsX:
  AUC_num_of_sets, AUC_sets, AUC_scores, AUC_params = show_all_auc(
          X, y, groups, MODEL[1], n_splits, MODEL[2])
  ACC_num_of_sets, ACC_sets, ACC_scores, ACC_params = show_all_sets(
          X, y, groups, MODEL[1], n_splits, MODEL[2])
  print(MODEL[0])
  for i in range(num_of_sets):
    print('    set: {}\n\tAUC: {:.2}\n\t  parameters: {}'.format(AUC_sets[i], 
          AUC_scores[i], AUC_params[i]))
    print('        accuracy: {:.2}\n\t  parameters: {}'.format(ACC_scores[i], 
          ACC_params[i]))
  print()

# WORD VECTORIZATION MODELS

# SETTING UP EMBEDDINGS

# Function below addapted from stacoverflow post by murauer,
# posted on the 12th of March 2019, 12:04
# Available at: https://stackoverflow.com/a/38230349

def load_model(file_path):
    print("Loading Glove Model...")
    model = {}
    with open(file_path, 'r', encoding = "utf8") as f:
      for line in f:
          splitted_line = line.split()
          word = splitted_line[0]
          embedding = np.array([float(val) for val in splitted_line[1:]])
          model[word] = embedding

    print(len(model), " words loaded.")
    return model

  
  
EMBEDDING_PATH = './court_trials_dataset/glove.6B.50d.txt'
EMBEDDING_DIM = 50
NUM_WORDS = 20000
PADDING = 'post'

word2vec_model = load_model(EMBEDDING_PATH)

# choosing testimonies / eliminating outliers

TESTIM = trial

tokenizer = Tokenizer(num_words = NUM_WORDS)
tokenizer.fit_on_texts(TESTIM['input'])

sequences = tokenizer.texts_to_sequences(TESTIM['input'])
trial_sequences = sequences
word_index = tokenizer.word_index

vocab_size = min(len(word_index) + 1, NUM_WORDS)
embedding_matrix = np.zeros((vocab_size, EMBEDDING_DIM))

for word, i in word_index.items():
    if i >= NUM_WORDS:
        continue
    try:
        embedding_vector = word2vec_model[word]
        embedding_matrix[i] = embedding_vector
    except KeyError:
        embedding_matrix[i] = np.random.normal(scale = 0.1,size = (
                EMBEDDING_DIM, ))
        
# SETTING UP DATA FOR TRAINING AND TESTING NN        
        
max_len = max([len(x) for x in sequences])

X = pad_sequences(sequences, maxlen = max_len, padding = PADDING)
y = TESTIM['deceptive']
groups = TESTIM['group']

from keras.layers import Conv2D


'''
    SERIAL CONVOLUTIONS
'''

seq_len = max_len
batch_size = 10
epochs = 50
verbose = 0
n_splits = 10

ave_acc = 0
ave_auc = 0

for train, test in GroupKFold(n_splits = n_splits).split(
        X, y, groups = groups):
  # referring to inputs and outputs: X(train), y(train), X(test), y(test)
  
  mcnn = Sequential()
  mcnn.add(Embedding(vocab_size, EMBEDDING_DIM, input_length = seq_len, 
                     embeddings_initializer = Constant(embedding_matrix)))
  mcnn.add(Conv1D(filters = 20, kernel_size = (8), 
                  kernel_initializer = RandomNormal(mean=0.0, stddev=0.05)))
  mcnn.add(MaxPooling1D(pool_size = 2))
  mcnn.add(Conv1D(filters = 20, kernel_size = (5), 
                  kernel_initializer = RandomNormal(mean=0.0, stddev=0.05)))
  mcnn.add(MaxPooling1D(pool_size = 2))
  mcnn.add(Conv1D(filters = 20, kernel_size = (2), 
                  kernel_initializer = RandomNormal(mean=0.0, stddev=0.05)))
  mcnn.add(MaxPooling1D(pool_size = 2))
  mcnn.add(Flatten())
  mcnn.add(Dense(300, activation = 'relu'))
  mcnn.add(Dropout(0.5))
  mcnn.add(Dense(1, activation = 'sigmoid'))
  
  mcnn.compile(loss = 'binary_crossentropy', optimizer = 'sgd', 
               metrics = ['accuracy'])
  mcnn.fit(X[train], y[train], batch_size = batch_size, epochs = epochs, 
           verbose = verbose, validation_data = (X[test], y[test]))
  _, t_acc = mcnn.evaluate(X[train], y[train], verbose = 0)
  _, acc = mcnn.evaluate(X[test], y[test], verbose = 0)
  
  AUC =  metrics.roc_auc_score(y[test], mcnn.predict(X[test]))
  
  print('Training accuracy: {:.2}'.format(t_acc))
  print('Validation accuracy: {:.2}'.format(acc))
  print('AUC: {:.2}'.format(AUC))
  ave_acc += acc
  ave_auc += AUC

print('Average accuracy: {:.3}'.format(ave_acc/n_splits*100))
print('Average AUC: {:.3}'.format(ave_auc/n_splits))

seq_len = max_len
batch_size = 8
epochs = 26
verbose = 0
n_splits = 10

ave_acc = 0
ave_auc = 0

'''
    PARALLEL CONVOLUTIONS
'''

from keras.initializers import Constant
from keras.layers import BatchNormalization, Reshape, MaxPooling2D, concatenate

# function adapted from stackoverflow post of durjoy
# available at: https://stackoverflow.com/a/45304179

def build_model(Xt, yt, embeddings, verbose = verbose, 
                epochs = epochs, batch_size =  batch_size):

  seq_len = Xt.shape[1]
  filter_size = [(3), (5), (8)]
  feature_maps = 20
  dropout = 0.5
  
  mod = None
  
  input = Input(shape=(seq_len,), dtype = 'int32')
  
  embedding = Embedding(vocab_size, EMBEDDING_DIM, 
                        embeddings_initializer = Constant
                        (embedding_matrix))(input)
  
  reshape = Reshape((seq_len, EMBEDDING_DIM, 1))(embedding)
  
  conv0 = Conv2D(feature_maps, (filter_size[0], EMBEDDING_DIM), 
                 kernel_initializer='random_normal')(reshape)
  maxp0 = MaxPooling2D((seq_len - filter_size[0] + 1, 1), strides = (2, 1), 
                       padding = 'valid')(conv0)
  
  conv1 = Conv2D(feature_maps, (filter_size[1], EMBEDDING_DIM), 
                 kernel_initializer='random_normal')(reshape)
  maxp1 = MaxPooling2D((seq_len - filter_size[1] + 1, 1), strides = (3, 1), 
                       padding = 'valid')(conv1)
  
  conv2 = Conv2D(feature_maps, (filter_size[2], EMBEDDING_DIM), 
                 kernel_initializer='random_normal')(reshape)
  maxp2 = MaxPooling2D((seq_len - filter_size[2] + 1, 1), strides = (4, 1), 
                       padding = 'valid')(conv2)
  
  concat = concatenate(([maxp0, maxp1, maxp2]), axis = 1)

  flat = Flatten()(concat)
  
  den1 = Dense(300, activation = 'relu', 
               kernel_initializer = 'random_normal')(flat)
  
  drop = Dropout(dropout)(den1)
  
  den2 = Dense(1000, activation = 'relu', 
               kernel_initializer = 'random_normal')(flat)
  
  drop2 = Dropout(dropout)(den2)
  
  output = Dense(1, activation = 'sigmoid')(drop2)

  mod = Model(input, output)
  
  mod.compile(loss = 'binary_crossentropy', optimizer = 'sgd', 
              metrics = ['binary_accuracy'])
  
  mod.fit(Xt, yt, batch_size = batch_size, epochs = epochs, verbose = verbose)
  
  return mod

for train, test in GroupKFold(n_splits = n_splits).split(
        X, y, groups = groups):
  # referring to inputs and outputs: X(train), y(train), X(test), y(test)
  
  mcnn = build_model(X[train], y[train], 
                     embeddings = embedding_matrix, 
                     verbose = verbose, epochs = epochs, 
                     batch_size = batch_size)
  
  _, t_acc = mcnn.evaluate(X[train], y[train], verbose = verbose)
  _, acc = mcnn.evaluate(X[test], y[test], verbose = verbose)
  
  AUC =  metrics.roc_auc_score(y[test], mcnn.predict(X[test]))
  
  print('Training accuracy: {:.2}'.format(t_acc))
  print('Validation accuracy: {:.2}'.format(acc))
  print('AUC: {:.2}'.format(AUC))
  ave_acc += acc
  ave_auc += AUC

print('Average accuracy: {:.3}'.format(ave_acc/n_splits*100))
print('Average AUC: {:.3}'.format(ave_auc/n_splits))

'''
    BidirectionalLSTM
'''

from keras.layers import Bidirectional, LSTM, SpatialDropout1D

seq_len = max_len
batch_size = 10
epochs = 50
verbose = 0
n_splits = 10

ave_acc = 0
ave_auc = 0

for train, test in GroupKFold(n_splits = n_splits).split(
        X, y, groups = groups):
  # referring to inputs and outputs: X(train), y(train), X(test), y(test)
  
  rnn = Sequential()
  rnn.add(Embedding(vocab_size, output_dim = EMBEDDING_DIM, 
                    input_length = seq_len, 
                    embeddings_initializer = Constant(embedding_matrix)))
  rnn.add(Bidirectional(LSTM(64)))
  rnn.add(Dense(128, activation = 'relu'))
  rnn.add(Dropout(0.2))
  rnn.add(Dense(1, activation = 'sigmoid'))
  
  rnn.compile(loss = 'binary_crossentropy', optimizer = 'sgd',
              metrics = ['accuracy'])
  rnn.fit(X[train], y[train], batch_size = batch_size, epochs = epochs, 
          verbose = verbose, validation_data = (X[test], y[test]))
  _, t_acc = rnn.evaluate(X[train], y[train], verbose = 0)
  _, acc = rnn.evaluate(X[test], y[test], verbose = 0)
  
  AUC =  metrics.roc_auc_score(y[test], rnn.predict(X[test]))
  
  print('Training accuracy: {:.2}'.format(t_acc))
  print('Validation accuracy: {:.2}'.format(acc))
  print('AUC: {:.2}'.format(AUC))
  ave_acc += acc
  ave_auc += AUC

print('Average accuracy: {:.3}'.format(ave_acc/n_splits*100))
print('Average AUC: {:.3}'.format(ave_auc/n_splits))

'''
    FAKE NEWS TRAINING
'''
path = join('./fake_news_dataset/train.csv')

fake_set = pd.read_csv(path)

#print(fake_set.head(3))

fake_set = fake_set[['text', 'label']]

fake_set['preprocessed'] = fake_set['text'].apply(preprocess)

#print(fake_set.head(3))

# REMOVE STOPWORDS
# stop_words = list of words to be removed
stop_words = set(stopwords.words('english'))

fake_set['no_stopwords'] = fake_set['preprocessed'].apply(
        lambda x : remove_words(x, stop_words))


# REMOVE THE MOST COMMON WORDS - NOISE
# NUMB = number of most freq words to be removed
NUMB = 10

fake_texts = ' '.join(trial['no_stopwords'])
wordset = dict(FreqDist(fake_texts.split()).most_common(NUMB)).keys()

fake_set['less_noise'] = fake_set['no_stopwords'].apply(
        lambda x : remove_words(x, wordset))
n_splits = 10

#fake_set.describe()

# SERIAL CONV
seq_len = max_len
batch_size = 250
epochs = 25
verbose = 0
n_splits = 10

ave_acc = 0
ave_auc = 0

for train, test in StratifiedKFold(n_splits = n_splits).split(X_fake, y_fake):
  # referring to inputs and outputs: X(train), y(train), X(test), y(test)
  
  scnn = Sequential()
  scnn.add(Embedding(vocab_size, EMBEDDING_DIM, input_length = seq_len, 
                     embeddings_initializer = Constant(embedding_matrix)))
  scnn.add(Conv1D(filters = 20, kernel_size = 8, 
                  kernel_initializer = RandomNormal(mean=0.0,
                                                    stddev=0.05, 
                                                    seed=940321)))
  scnn.add(MaxPooling1D(pool_size = 2))
  scnn.add(Conv1D(filters = 20, kernel_size = 5,
                  kernel_initializer = RandomNormal(mean=0.0, 
                                                    stddev=0.05, 
                                                    seed=940320)))
  scnn.add(MaxPooling1D(pool_size = 2))
  scnn.add(Conv1D(filters = 20, kernel_size = 2, 
                  kernel_initializer = RandomNormal(mean=0.0, 
                                                    stddev=0.05, 
                                                    seed=940319)))
  scnn.add(MaxPooling1D(pool_size = 2))
  scnn.add(Flatten())
  scnn.add(Dense(300, activation = 'relu', 
                 kernel_initializer = RandomNormal(mean=0.0, 
                                                   stddev=0.05, 
                                                   seed=210394)))
  scnn.add(Dropout(0.5))
  scnn.add(Dense(1, activation = 'sigmoid'))
  
  scnn.compile(loss = 'binary_crossentropy', optimizer = 'sgd', 
               metrics = ['accuracy'])
  scnn.fit(X_fake[train], y_fake[train], batch_size = batch_size, 
           epochs = epochs, verbose = verbose, 
           validation_data = (X_fake[test], y_fake[test]))
  _, t_acc = scnn.evaluate(X_fake[train], y_fake[train], verbose = 0)
  _, acc = scnn.evaluate(X_fake[test], y_fake[test], verbose = 0)
  
  AUC =  metrics.roc_auc_score(y_fake[test], scnn.predict(X_fake[test]))
  
  print('Training accuracy: {:.2}'.format(t_acc))
  print('Validation accuracy: {:.2}'.format(acc))
  print('AUC: {:.2}'.format(AUC))
  ave_acc += acc
  ave_auc += AUC

print('Average accuracy: {:.3}'.format(ave_acc/n_splits*100))
print('Average AUC: {:.3}'.format(ave_auc/n_splits))

seq_len = 150
batch_size = 100
epochs = 25
verbose = 0
n_splits = 10

scnn = Sequential()
scnn.add(Embedding(vocab_size, EMBEDDING_DIM, 
                   input_length = seq_len, 
                   embeddings_initializer = Constant(embedding_matrix)))
scnn.add(Conv1D(filters = 20, kernel_size = 8,
                kernel_initializer = RandomNormal(mean=0.0, 
                                                  stddev=0.05, 
                                                  seed=940321)))
scnn.add(MaxPooling1D(pool_size = 2))
scnn.add(Conv1D(filters = 20, kernel_size = 5,
                kernel_initializer = RandomNormal(mean=0.0, 
                                                  stddev=0.05, 
                                                  seed=940321)))
scnn.add(MaxPooling1D(pool_size = 2))
scnn.add(Conv1D(filters = 20, kernel_size = 2,
                kernel_initializer = RandomNormal(mean=0.0, 
                                                  stddev=0.05, 
                                                  seed=940321)))
scnn.add(MaxPooling1D(pool_size = 2))
scnn.add(Flatten())
scnn.add(Dense(300, activation = 'relu',
               kernel_initializer = RandomNormal(mean=0.0, 
                                                 stddev=0.05,
                                                 seed=210394)))
scnn.add(Dropout(0.5))
scnn.add(Dense(1, activation = 'sigmoid'))
  
scnn.compile(loss = 'binary_crossentropy', optimizer = 'sgd', 
             metrics = ['accuracy'])
scnn.fit(X_fake, y_fake, batch_size = batch_size, epochs = epochs, 
         verbose = verbose)

PADDING = 'post'
fake_model = scnn

# choosing testimonies / eliminating outliers

trial_sequences = tokenizer.texts_to_sequences(trial['input'])
trial_sequences_no_stop = tokenizer.texts_to_sequences(trial['no_stopwords'])
trial_sequences_less_noise = tokenizer.texts_to_sequences(trial['less_noise'])
        
# TESTING NN        
        
max_len = 150

X_1 = pad_sequences(trial_sequences, maxlen = max_len, padding = PADDING)
X_2 = pad_sequences(trial_sequences_no_stop, maxlen = max_len, 
                    padding = PADDING)
X_3 = pad_sequences(trial_sequences_less_noise, maxlen = max_len, 
                    padding = PADDING)
y = trial['deceptive']

_, acc = fake_model.evaluate(X, y, verbose = 0)
AUC =  metrics.roc_auc_score(y, fake_model.predict(X))

print('Performance of CRNN model trained on fake news dataset and test\
      ed on differently preprocessed transcripts from Deception Detecti\
      on dataset:\n')
print('Tested on Deception Detection dataset:\nAccura\
      cy: {:.2}'.format(acc),'\nAUC: {:.2}'.format(AUC))

_, acc = fake_model.evaluate(X_1, y, verbose = 0)
AUC =  metrics.roc_auc_score(y, fake_model.predict(X_1))

print('\nTested on Deception Detection dataset without stopwords:\nAccura\
      cy: {:.2}'.format(acc),'\nAUC: {:.2}'.format(AUC))

_, acc = fake_model.evaluate(X_2, y, verbose = 0)
AUC =  metrics.roc_auc_score(y, fake_model.predict(X_2))

print('\nTested on Deception Detection dataset without stopwords\
      and without most common words:\nAccuracy: {:.2}'.format(acc),'\nA\
      UC: {:.2}'.format(AUC))

#PARALLEL CONV

from keras.initializers import Constant
from keras.layers import BatchNormalization

seq_len = 150
batch_size = 200
epochs = 100
verbose = 0
n_splits = 10

# function adapted from stackoverflow post of durjoy
# available at: https://stackoverflow.com/a/45304179

def build_model(Xt, yt, embeddings, verbose = verbose, 
                epochs = epochs, batch_size =  batch_size):

  seq_len = Xt.shape[1]
  filter_size = [(3), (5), (8)]
  feature_maps = 20
  dropout = 0.5
  
  mod = None
  
  input = Input(shape=(seq_len,), dtype = 'int32')
  
  embedding = Embedding(vocab_size, EMBEDDING_DIM, 
                        embeddings_initializer = Constant(
                                embedding_matrix))(input)
  
  reshape = Reshape((seq_len, EMBEDDING_DIM, 1))(embedding)
  
  conv0 = Conv2D(feature_maps, (filter_size[0], EMBEDDING_DIM), 
                 activation = 'relu', 
                 kernel_initializer='random_normal')(reshape)
  maxp0 = MaxPooling2D((seq_len - filter_size[0] + 1, 1), 
                       strides = (2, 1), padding = 'valid')(conv0)
  
  conv1 = Conv2D(feature_maps, (filter_size[1], EMBEDDING_DIM), 
                 activation = 'relu', 
                 kernel_initializer='random_normal')(reshape)
  maxp1 = MaxPooling2D((seq_len - filter_size[1] + 1, 1), 
                       strides = (3, 1), padding = 'valid')(conv1)
  
  conv2 = Conv2D(feature_maps, (filter_size[2], EMBEDDING_DIM), 
                 activation = 'relu', 
                 kernel_initializer='random_normal')(reshape)
  maxp2 = MaxPooling2D((seq_len - filter_size[2] + 1, 1), 
                       strides = (4, 1), padding = 'valid')(conv2)
  
  concat = concatenate(([maxp0, maxp1, maxp2]), axis = 1)

  flat = Flatten()(concat)
  
  den1 = Dense(300, activation = 'relu', 
               kernel_initializer = 'random_normal')(flat)
  
  drop = Dropout(dropout)(den1)
  
  den2 = Dense(1000, activation = 'relu', 
               kernel_initializer = 'random_normal')(flat)
  
  drop2 = Dropout(dropout)(den2)
  
  output = Dense(1, activation = 'sigmoid')(drop2)

  mod = Model(input, output)
  
  mod.compile(loss = 'binary_crossentropy', optimizer = 'sgd', 
              metrics = ['binary_accuracy'])
  
  mod.fit(Xt, yt, batch_size = batch_size, epochs = epochs, verbose = verbose)
  
  return mod

seq_len = 150
batch_size = 200
epochs = 100
verbose = 0
n_splits = 10

ave_acc = 0
ave_auc = 0

for train, test in StratifiedKFold(n_splits = n_splits).split(X_fake, y_fake):
  # referring to inputs and outputs: X(train), y(train), X(test), y(test)
  
  pcnn = build_model(X_fake[train], y_fake[train], 
                     embeddings = embedding_matrix, 
                     verbose = verbose, epochs = epochs, 
                     batch_size = batch_size)
  
  _, t_acc = pcnn.evaluate(X_fake[train], y_fake[train], verbose = 0)
  _, acc = pcnn.evaluate(X_fake[test], y_fake[test], verbose = 0)
  
  AUC =  metrics.roc_auc_score(y_fake[test], pcnn.predict(X_fake[test]))
  
  print('Training accuracy: {:.2}'.format(t_acc))
  print('Validation accuracy: {:.2}'.format(acc))
  print('AUC: {:.2}'.format(AUC))
  ave_acc += acc
  ave_auc += AUC

print('Average accuracy: {:.3}'.format(ave_acc/n_splits*100))
print('Average AUC: {:.3}'.format(ave_auc/n_splits))

seq_len = 150
batch_size = 100
epochs = 25
verbose = 0
n_splits = 10

pcnn = build_model(X_fake, y_fake, embeddings = embedding_matrix, 
                   verbose = verbose, epochs = epochs, batch_size = batch_size)

PADDING = 'post'
fake_model = pcnn

# choosing testimonies / eliminating outliers

trial_sequences = tokenizer.texts_to_sequences(trial['input'])
trial_sequences_no_stop = tokenizer.texts_to_sequences(trial['no_stopwords'])
trial_sequences_less_noise = tokenizer.texts_to_sequences(trial['less_noise'])
        
# TESTING NN        
        
max_len = 150

X_1 = pad_sequences(trial_sequences, maxlen = max_len, padding = PADDING)
X_2 = pad_sequences(trial_sequences_no_stop, maxlen = max_len, 
                    padding = PADDING)
X_3 = pad_sequences(trial_sequences_less_noise, maxlen = max_len, 
                    padding = PADDING)
y = trial['deceptive']

_, acc = fake_model.evaluate(X, y, verbose = 0)
AUC =  metrics.roc_auc_score(y, fake_model.predict(X))

print('Performance of Parallel CNN model trained on fake news dataset and te\
      sted on differently preprocessed transcripts from Deception Detection d\
      ataset:\n')
print('Tested on Deception Detection dataset:\nAccura\
      cy: {:.2}'.format(acc),'\nAUC: {:.2}'.format(AUC))

_, acc = fake_model.evaluate(X_1, y, verbose = 0)
AUC =  metrics.roc_auc_score(y, fake_model.predict(X_1))

print('\nTested on Deception Detection dataset without stopwords:\n\A\
      ccuracy: {:.2}'.format(acc),'\nAUC: {:.2}'.format(AUC))

_, acc = fake_model.evaluate(X_2, y, verbose = 0)
AUC =  metrics.roc_auc_score(y, fake_model.predict(X_2))

print('\nTested on Deception Detection dataset without stopwords and with\
      out most common words:\nAccuracy: {:.2}'.format(acc),'\nAUC: {:.2}\
      '.format(AUC))

# TEXTBLOB

from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer
nltk.download('movie_reviews')
nltk.download('punkt')

trial['textblob_analysis'] = trial['input'].apply(
        lambda x : TextBlob(x, analyzer = NaiveBayesAnalyzer()))

from sklearn import metrics

print('Obtaining sentiment...')
trial['sentiment'] = trial['textblob_analysis'].apply(
        lambda x : x.sentiment[1])
print('Obtaining subjectivity...')
trial['subjectivity'] = trial['textblob_analysis'].apply(
        lambda x : x.subjectivity)
print('Obtaining classification...')
trial['polarity'] = trial['textblob_analysis'].apply(lambda x : x.polarity)

trial['sentiment'] = trial['sentiment'].apply(lambda x : (x + 1)/2)

AUC_sentiment =  metrics.roc_auc_score(trial['deceptive'].values,
                                       trial['sentiment'].values)
AUC_subjectivity = metrics.roc_auc_score(trial['deceptive'].values,
                                         trial['subjectivity'].values)
AUC_polarity = metrics.roc_auc_score(trial['deceptive'].values,
                                     trial['polarity'].values)

print()
print('Sentiment: {:.3}'.format(AUC_sentiment))
print('Subjectivity: {:.3}'.format(AUC_subjectivity))
print('Polarity: {:.3}'.format(AUC_polarity))
