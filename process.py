import pandas as pd
import numpy as np
from IPython import get_ipython
import seaborn as sns
import matplotlib.pyplot as plt
import os, csv, random

data_dir = 'data'

random.seed(0)
col = ['PassengerId', 'Survived', 'Pclass', 'Title', 'Sex', 'Age', 'Family', 'Fare', 'Cabin', 'Embarked']
cls = {'PassengerId': 1, 'Survived': 1, 'Pclass': 3, 'Title': 3, 'Sex': 2, 'Age': 7, 'Family': 3, 'Fare': 5, 'Cabin': 3, 'Embarked': 3}

def one_hot(s, k):
  x = [0] * k
  x[s] = 1
  return x

def xv(r):
  x = []
  x.append(r['PassengerId'])

  if 'Survived' in r: 
    x.append(r['Survived'])
  else:
    x.append(0)

  #x.append(r['Pclass'])
  #x.append(r['Title'])
  x.append(r['Sex'])
  #x.append(r['Age'])
  #x.append(r['Family'])
  #x.append(r['Fare'])
  x.append(r['Cabin'])
  #x.append(r['Embarked'])
  x.extend(one_hot(r['Pclass'], 3))
  x.extend(one_hot(r['Title'], 4))
  x.extend(one_hot(r['Age'], 7))
  x.extend(one_hot(r['Family'], 3))
  x.extend(one_hot(r['Fare'], 5))
  #x.extend(one_hot(r['Cabin'], 3))
  #x.extend(one_hot(r['Embarked'], 3))

  return x

def remap(df, col, n, fill=None):
  df.loc[ df[col] <= n[0], col] = 0
  for i in range(len(n)-1):
    df.loc[ (n[i] < df[col]) & (df[col] <= n[i+1]), col] = i + 1
  df.loc[ n[-1] < df[col], col] = len(n)
  if fill:
    df[col] = df[col].fillna(len(n)+1)
  #print df.groupby(col).size(), '\n'



for f in ['train', 'test']:
  df = pd.read_csv('{}/{}.csv'.format(data_dir, f))

  if f == 'test':
    df.insert(1, 'Survived', 0)

  # Pclass
  df['Pclass'] = df['Pclass'].map(lambda x: x-1)
  
  # Title
  df['Title'] = df['Name'].str.extract('(\w+)\.', expand=False)
  df['Title'] = df['Title'].replace(['Mlle', 'Ms', 'Lady'], 'Miss')
  df['Title'] = df['Title'].replace(['Countess', 'Mme', 'Dona'], 'Mrs')
  df.loc[ (df['Title'] == 'Dr') & (df['Sex'] == 'female'), 'Title'] = 'Mrs'
  df['Title'] = df['Title'].replace(['Capt', 'Col', 'Don', 'Dr', 'Jonkheer', 'Major', 'Rev', 'Sir'], 'Mr')
  df['Title'] = df['Title'].map({'Mr':0, 'Master':1, 'Mrs':2, 'Miss':3})

  # Sex
  df['Sex'] = df['Sex'].map({'male':0 , 'female':1})

  # Age
  """
  df['Age'] = df['Age'].fillna(-1)
  df.loc[df['Survived']==0, 'Age'].plot.hist(bins = [i*5-5 for i in range(17)], color=(1,1,0,0.5))
  df.loc[df['Survived']==1, 'Age'].plot.hist(bins = [i*5-5 for i in range(17)], color=(0.3, 0.2, 1, 0.5))
  plt.show()
  """

  remap(df, 'Age', [5, 15, 36, 55, 64], True)

  # Family
  df['Family'] = df['SibSp'] + df['Parch']
  remap(df, 'Family', [0, 3])

  # Fare
  """
  df.loc[df['Survived']==0, 'Fare'].plot.hist(color=(1,1,0,0.5), bins = 50)
  df.loc[df['Survived']==1, 'Fare'].plot.hist(color=(0.3, 0.2, 1, 0.5), bins = 50)
  plt.show()
  df['Fare'] = pd.qcut(df['Fare'], 4)
  """
  df['Fare'] = df['Fare'].fillna(df['Fare'].mode()[0])
  remap(df, 'Fare', [4, 10, 20, 45])

  # Cabin
  df['Cabin'] = df['Cabin'].fillna('T')
  df['Cabin'] = df['Cabin'].str.extract('(\w)', expand=False)
  df['Cabin'] = df['Cabin'].replace(['B', 'C', 'D', 'E'], 0)
  df['Cabin'] = df['Cabin'].replace(['A', 'F', 'G'], 1)
  df['Cabin'] = df['Cabin'].replace('T', 2)
  #df['Cabin'] = df['Cabin'].map(lambda x: 0 if pd.isnull(x) else 1)

  # Embarked
  df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
  df['Embarked'] = df['Embarked'].map({'C':0, 'Q':1, 'S':2})
 
  #del df['Cabin']
  #del df['Embarked']
  del df['Name']
  del df['SibSp']
  del df['Parch']
  del df['Ticket']
  
  df = df.astype('int32')
  df = df[col]

  """
  print df.isnull().sum(), '\n'
  if f == 'train':
    print pd.crosstab(df['Age'], df['Survived'])
    print df.groupby('Cabin').size()
  """
  
  df.to_csv('{}/{}_process.csv'.format(data_dir, f), index=False)
  df = pd.read_csv('{}/{}_process.csv'.format(data_dir, f))
  
  cf = csv.writer(open('{}/{}'.format(data_dir, f), 'w'))
  for i, r in df.iterrows():
    cf.writerow(xv(r))


# unlabel data
num = 10000
d = {}
for c in col:
  d[c] = [random.randint(0, cls[c]-1) for i in range(num)]

df = pd.DataFrame(data=d)
df = df[col]
df.to_csv('{}/{}_process.csv'.format(data_dir, 'unlabel'), index=False)


os.system('head -n 791 data/train > data/train_s')
os.system('tail -n 100 data/train > data/val')

