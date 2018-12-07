import numpy as np
import pandas as pd
from utils import write_test
from xgboost.sklearn import XGBClassifier
from sklearn import ensemble, gaussian_process, linear_model, naive_bayes, neighbors, svm, tree
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, log_loss

it = 20

sss = StratifiedShuffleSplit(n_splits=it, train_size=0.9, random_state=0)

df = pd.read_csv('data/train_process.csv')
#df = pd.read_csv('data/train', header=None)
x = np.array(df.iloc[:, 2:])
y = np.array(df.iloc[:, 1])

df = pd.read_csv('data/test_process.csv')
#df = pd.read_csv('data/test', header=None)
id = df.iloc[:, 0].tolist()
test = np.array(df.iloc[:, 2:])

df = pd.read_csv('data/unlabel_process.csv')
unlabel_x = np.array(df.iloc[:, 2:])

param_grid = {'criterion': ['gini', 'entropy'], 'max_depth': [2,4,6,8,10,None], 'random_state': [0]}
#param_grid = {'learning_rate': [0.01, 0.005], 'n_estimators': [100, 150, 200, 250, 300, 350], 'min_child_weight': [1, 2, 3, 4], 'criterion': ['gini', 'entropy'], 'max_depth': [2,4,6,8], 'seed': [0]}
#clf = GridSearchCV(tree.DecisionTreeClassifier(), param_grid=param_grid, scoring = 'roc_auc')#, cv = sss)
#clf = ensemble.AdaBoostClassifier(n_estimators=200,random_state=0,learning_rate=0.05)
clf = svm.SVC()
#clf.fit(x, y)

#it = 1
#sss = StratifiedShuffleSplit(n_splits=it, train_size=0.8, random_state=5)

def train():
  acc_train, acc_test = 0.0, 0.0
  step = 0

  for train_i, test_i in sss.split(x, y):
    print ("Training step {} ... ".format(step))
    step += 1
    x_train, x_test = x[train_i], x[test_i]
    y_train, y_test = y[train_i], y[test_i]

    clf.fit(x_train, y_train)
    #unlabel_y = clf.predict(unlabel_x)
    #clf.fit(unlabel_x, unlabel_y)

    if step <= it:
      y_pred_train = clf.predict(x_train)
      y_pred = clf.predict(x_test)
      acc_train += accuracy_score(y_train, y_pred_train)
      acc_test += accuracy_score(y_test, y_pred)

  acc_train /= it
  acc_test /= it
  print (round((acc_train),3))
  print (round((acc_test),3))

#print clf.best_params_
#print clf.cv_results_['mean_train_score'][clf.best_index_]
#print clf.cv_results_['mean_test_score'][clf.best_index_]

train()

result = clf.predict(test)
write_test(id, result)

