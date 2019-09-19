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

from scipy.sparse import hstack
from scipy.sparse import csr_matrix

seed(210394)
set_random_seed(210394)

nltk.download('stopwords')

'''
    ADDITIONAL FUNCTIONS:
'''

def remove_words(input, wordset):
  return ' '.join([w for w in input.split() if not w in wordset])

def optimize_model(X, y, groups, MODEL, n_splits, parameters):
  param_grid = ParameterGrid(parameters)
  best_score = 0
  best_params = dict()
  i = 1
  for param_set in param_grid:
    fold_score = 0
    for train, test in GroupKFold(n_splits = n_splits).split(X, y, groups = groups):
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
  for input in X:
    set_score, set_params = optimize_model(input, y, groups, MODEL, n_splits, parameters)
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
  for i in range(len(X)):
    set_score, set_params = optimize_model(X[i], y, groups, MODEL, n_splits, parameters)
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
    for train, test in GroupKFold(n_splits = n_splits).split(X, y, groups = groups):
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
  for input in X:
    set_score, set_params = optimize_auc(input, y, groups, MODEL, n_splits, parameters)
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
  for i in range(len(X)):
    set_score, set_params = optimize_auc(X[i], y, groups, MODEL, n_splits, parameters)
    scores.append(set_score)
    params.append(set_params)
    sets.append(i+1)
    num_of_sets = i+1
  return num_of_sets, sets, scores, params

def preprocess(text):
    text = str(text)
    text = re.sub(r"(\nhttps?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9]\.[^\s]{2,})", "url", text)
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

path = 'drive/My Drive/data/'

# LOADING DECEPTIVE

file_names = [i for i in listdir(
    join(path,'transcription/deceptive/')) if isfile(join(path, 'transcription/deceptive/', i))]

dec_texts = []

for name in file_names:
    tmp = join(path, 'transcription/deceptive/', name)
    with open(tmp, 'r',  encoding="utf8") as f:
        dec_texts.append(f.read())
    
# LOADING TRUTHFUL
        
file_names = [i for i in listdir(
    join(path,'transcription/truthful/')) if isfile(join(path,'transcription/truthful/', i))]

tru_texts = []

for name in file_names:
    tmp = join(path, 'transcription/truthful/', name)
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
  
# ELIMINATING OUTLIERS

trial = trial[trial['len']<140]
trial.reset_index(inplace = True, drop = True)

# REMOVE STOPWORDS
# stop_words = list of words to be removed
stop_words = set(stopwords.words('english'))

trial['no_stopwords'] = trial['input'].apply(lambda x : remove_words(x, stop_words))

# REMOVE THE MOST COMMON WORDS - NOISE
# NUMB = number of most freq words to be removed
NUMB = 10

texts = ' '.join(trial['no_stopwords'])
wordset = dict(FreqDist(texts.split()).most_common(NUMB)).keys()

trial['less_noise'] = trial['no_stopwords'].apply(lambda x : remove_words(x, wordset))

tfidf = TfidfVectorizer()
X_1 = tfidf.fit_transform(trial['input'])
X_2 = tfidf.fit_transform(trial['no_stopwords'])
X_3 = tfidf.fit_transform(trial['less_noise'])
X = [X_1, X_2, X_3]
y = trial['deceptive']
groups = trial['group']
n_splits = 10

'''
    MODEL GRID SELECTION FOR TESTING:
'''

models = [
    ('Naive Bayes',
     MultinomialNB,
     {'alpha': np.arange(0.05, 2, 0.05),
      'fit_prior': [True, False],
     }
    ),
    
    ('Decision Tree',
     DecisionTreeClassifier,
     {'criterion': ['gini', 'entropy'],
      'splitter': ['best', 'random'],
      'random_state': [940321],
     }
    ),
    
    ('Logistic Regression',
     LogisticRegression,
     {'dual': [False, True],
      'C': np.arange(0.05, 1.0, 0.05),
      'fit_intercept': [True, False],
      'solver': ['liblinear'],
      'random_state': [940321],
     }
    ),
    
    ('Support Vector Classifier',
     SVC,
     {'C': np.arange(0.5, 2.5, 0.5),
      'kernel': ['linear', 'poly', 'rbf'],
      'degree': [1, 2, 3],
      'probability': [True],
      'shrinking': [True, False],
      'decision_function_shape': ['ovo', 'ovr'],
      'gamma': ['auto', 'scale'],
      'random_state': [940321],
     }
    ),
    
    ('K-Nearest Neighbors',
     KNeighborsClassifier,
     {'algorithm': ['auto', 'brute'],
      'weights': ['uniform', 'distance'],
      'n_neighbors': range(2, 10, 1),
      'leaf_size': range(10, 50, 5),
     }
    ),
    
    ('Random Forest Classifier',
     RandomForestClassifier,
     {'n_estimators': range(5, 40, 5),
      'criterion': ['gini', 'entropy'],
      'bootstrap': [True, False],
      'random_state': [940321],
     }
    ),
    
    ('AdaBoost',
     AdaBoostClassifier,
     {'base_estimator': [DecisionTreeClassifier(max_depth=1)],
      'n_estimators': range(10, 100, 10),
      'random_state': [940321],
      'algorithm': ['SAMME', 'SAMME.R'],
      'learning_rate': np.arange(0.3, 1.2, 0.1)
     }
    )
]

'''
    MODEL TESTING WITH ACCURACY AND AUC:
'''
# TFIDF
tfidf = TfidfVectorizer()
X_1 = tfidf.fit_transform(trial['input'])
X_2 = tfidf.fit_transform(trial['no_stopwords'])
X_3 = tfidf.fit_transform(trial['less_noise'])
X = [X_1, X_2, X_3]
y = trial['deceptive']
groups = trial['group']
n_splits = 10

for MODEL in models:
  AUC_num_of_sets, AUC_sets, AUC_scores, AUC_params = show_all_auc(X, y, groups, MODEL[1], n_splits, MODEL[2])
  ACC_num_of_sets, ACC_sets, ACC_scores, ACC_params = show_all_sets(X, y, groups, MODEL[1], n_splits, MODEL[2])
  print(MODEL[0])
  for i in range(num_of_sets):
    print('    set: {}\n\tAUC: {:.2}\n\t  parameters: {}'.format(AUC_sets[i], AUC_scores[i], AUC_params[i]))
    print('        accuracy: {:.2}\n\t  parameters: {}'.format(ACC_scores[i], ACC_params[i]))
  print()
  
# 1-, 2- & 3-grams
ngrams = CountVectorizer(ngram_range = (1, 3), min_df = 0.0, max_df = 1.0)
Xn_1 = ngrams.fit_transform(trial['input'])
Xn_2 = ngrams.fit_transform(trial['no_stopwords'])
Xn_3 = ngrams.fit_transform(trial['less_noise'])
Xn = [Xn_1, Xn_2, Xn_3]
yn = trial['deceptive']
groups = trial['group']
n_splits = 10

for MODEL in models:
  AUC_num_of_sets, AUC_sets, AUC_scores, AUC_params = show_all_auc(Xn, yn, groups, MODEL[1], n_splits, MODEL[2])
  ACC_num_of_sets, ACC_sets, ACC_scores, ACC_params = show_all_sets(Xn, yn, groups, MODEL[1], n_splits, MODEL[2])
  print(MODEL[0])
  for i in range(num_of_sets):
    print('    set: {}\n\tAUC: {:.2}\n\t  parameters: {}'.format(AUC_sets[i], AUC_scores[i], AUC_params[i]))
    print('        accuracy: {:.2}\n\t  parameters: {}'.format(ACC_scores[i], ACC_params[i]))
  print()
  
#2-, 3-grams
n1grams = CountVectorizer(ngram_range = (2, 3), min_df = 0.0, max_df = 1.0)
X1n_1 = n1grams.fit_transform(trial['input'])
X1n_2 = n1grams.fit_transform(trial['no_stopwords'])
X1n_3 = n1grams.fit_transform(trial['less_noise'])
X1n = [X1n_1, X1n_2, X1n_3]
y1n = trial['deceptive']
groups = trial['group']
n_splits = 10

for MODEL in models:
  AUC_num_of_sets, AUC_sets, AUC_scores, AUC_params = show_all_auc(X1n, y1n, groups, MODEL[1], n_splits, MODEL[2])
  ACC_num_of_sets, ACC_sets, ACC_scores, ACC_params = show_all_sets(X1n, y1n, groups, MODEL[1], n_splits, MODEL[2])
  print(MODEL[0])
  for i in range(num_of_sets):
    print('    set: {}\n\tAUC: {:.2}\n\t  parameters: {}'.format(AUC_sets[i], AUC_scores[i], AUC_params[i]))
    print('        accuracy: {:.2}\n\t  parameters: {}'.format(ACC_scores[i], ACC_params[i]))
  print()
  
#TFIDF + 2-, 3-grams

tfidf = TfidfVectorizer()
X_1 = tfidf.fit_transform(trial['input'])
X_2 = tfidf.fit_transform(trial['no_stopwords'])
X_3 = tfidf.fit_transform(trial['less_noise'])

n1grams = CountVectorizer(ngram_range = (2, 3), min_df = 0.0, max_df = 1.0)
X1n_1 = n1grams.fit_transform(trial['input'])
X1n_2 = n1grams.fit_transform(trial['no_stopwords'])
X1n_3 = n1grams.fit_transform(trial['less_noise'])

Xf_1 = hstack([X_1, X1n_1])
Xf_2 = hstack([X_2, X1n_2])
Xf_3 = hstack([X_3, X1n_3])

X2n = [Xf_1.todense(), Xf_2.todense(), Xf_3.todense()]
y2n = trial['deceptive']
groups = trial['group']
n_splits = 10

for MODEL in models:
  AUC_num_of_sets, AUC_sets, AUC_scores, AUC_params = show_all_auc(X1n, y1n, groups, MODEL[1], n_splits, MODEL[2])
  ACC_num_of_sets, ACC_sets, ACC_scores, ACC_params = show_all_sets(X1n, y1n, groups, MODEL[1], n_splits, MODEL[2])
  print(MODEL[0])
  for i in range(num_of_sets):
    print('    set: {}\n\tAUC: {:.2}\n\t  parameters: {}'.format(AUC_sets[i], AUC_scores[i], AUC_params[i]))
    print('        accuracy: {:.2}\n\t  parameters: {}'.format(ACC_scores[i], ACC_params[i]))
  print()
