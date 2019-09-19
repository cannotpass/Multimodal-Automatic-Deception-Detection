from sklearn.naive_bayes import MultinomialNB

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(trial['input'])
y = trial['deceptive']
groups = trial['group']

n_splits = 10
FIT_PRIOR = False
ALPHA = 0.3

print('Before stopwords removal:\n')

ave_acc = 0
i = 1
for train, test in GroupKFold(n_splits = n_splits).split(X, y, groups = groups):
  # SHUFFLING INDICES
  shuffle(train)
  shuffle(test)
  # referring to inputs and outputs: X(train), y(train), X(test), y(test)
  naive_bayes = MultinomialNB(fit_prior = FIT_PRIOR,
                              alpha = ALPHA).fit(X[train], y[train])
  predicted = naive_bayes.predict(X[test])
  print(i, 'Accuracy score: {:.2}'.format(accuracy_score(y[test], predicted)))
  i+=1
  ave_acc += accuracy_score(y[test], predicted)
  
print('Average accuracy: {:.2}'.format(ave_acc/n_splits))

from sklearn.tree import DecisionTreeClassifier as DTC

from random import shuffle

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(trial['input'])
y = trial['deceptive']
groups = trial['group']

n_splits = 10
CRITERION = 'gini'
SPLITTER = 'random'
RANDOM_STATE = 940321

print('Before stopwords removal:\n')

ave_acc = 0
i = 1
for train, test in GroupKFold(n_splits = n_splits).split(X, y, groups = groups):
  # SHUFFLING INDICES
  shuffle(train)
  shuffle(test)
  # referring to inputs and outputs: X(train), y(train), X(test), y(test)
  dt = DTC(criterion = CRITERION,
           splitter = SPLITTER,
           random_state = RANDOM_STATE).fit(X[train], y[train])
  predicted = dt.predict(X[test])
  print(i, 'Accuracy score: {:.2}'.format(accuracy_score(y[test], predicted)))
  i+=1
  ave_acc += accuracy_score(y[test], predicted)
  
print('Average accuracy: {:.2}'.format(ave_acc/n_splits))

from sklearn.linear_model import LogisticRegression as LogReg

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(trial['input'])
y = trial['deceptive']
groups = trial['group']

n_splits = 10
SOLVER = 'liblinear'
DUAL = True
RANDOM_STATE = 940321
PENALTY = 'l2'

print('\nAfter stopwords removal:\n')

ave_acc = 0
i = 1
for train, test in GroupKFold(n_splits = n_splits).split(X, y, groups = groups):
  # SHUFFLING INDICES
  shuffle(train)
  shuffle(test)
  # referring to inputs and outputs: X(train), y(train), X(test), y(test)
  lr = LR(dual = DUAL,
          penalty = PENALTY,
          solver = SOLVER,
          random_state = RANDOM_STATE).fit(X[train], y[train])
  predicted = lr.predict(X[test])
  print(i, 'Accuracy score: {:.2}'.format(accuracy_score(y[test], predicted)))
  i+=1
  ave_acc += accuracy_score(y[test], predicted)
  
print('Average accuracy: {:.2}'.format(ave_acc/n_splits))

from sklearn.svm import SVC

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(trial['no_stopwords'])
y = trial['deceptive']
groups = trial['group']

n_splits = 10
C = 2.0
KERNEL = 'rbf'
GAMMA = 'scale'

print('\nAfter stopwords removal:\n')

ave_acc = 0
i = 1
for train, test in GroupKFold(n_splits = n_splits).split(X, y, groups = groups):
  # SHUFFLING INDICES
  shuffle(train)
  shuffle(test)
  # referring to inputs and outputs: X(train), y(train), X(test), y(test)
  svm = SVC(C = C, kernel = KERNEL, gamma = GAMMA).fit(X[train], y[train])
  predicted = svm.predict(X[test])
  print(i, 'Accuracy score: {:.2}'.format(accuracy_score(y[test], predicted)))
  i+=1
  ave_acc += accuracy_score(y[test], predicted)
  
print('Average accuracy: {:.2}'.format(ave_acc/n_splits))
