import pandas as pd
import os, csv

data_dir = 'data'

def _map(x, x_list, s=False):
  if pd.isnull(x):
    return -1
  for i, k in enumerate(x_list):
    if k in x:
      if s:
        return k
      return i
  return -1


def _Name(x):
  n = x['Name']
  if n in ['Mr', 'Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col']:
    return 0
  elif n in ['Mrs', 'Countess', 'Mme']:
    return 1
  elif n in ['Miss', 'Mlle', 'Ms']:
    return 2
  elif n =='Dr':
    if x['Sex']=='male':
      return 0
    else:
      return 1
  else:
    return 3

name = ['Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev', 'Dr', 'Ms', 'Mlle','Col', 'Capt', 'Mme', 'Countess', 'Don', 'Jonkheer']
cabin = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'T']
#cabin = ['A', 'B', 'C', 'D', 'E', 'F']
embarked = ['C', 'Q', 'S']

def one_hot(s, k):
  x = [0.0] * k
  x[int(s)] = 1.0
  return x

def xv(r):
  x = []
  x.append(r['PassengerId'])

  if 'Survived' in r: 
    x.append(r['Survived'])
  else:
    x.append(0)

  x.append(r['Sex'])
  x.append(r['Family'])
  x.append(r['Fare'])
  #x.append(r['Fare_range'])
  x.append(r['Pclass'])
  #x.extend(one_hot(r['Pclass']-1, 3))

  #x.extend(one_hot(r['Name'], 4))
  x.extend(one_hot(r['Age_range'], 9))
  #x.append(r['Age'])
  x.extend(one_hot(r['Cabin'], 9))
  x.extend(one_hot(r['Embarked'], 3))


  return x

for f in ['train', 'test']:
  df = pd.read_csv('{}/{}.csv'.format(data_dir, f))

  df['Name'] = df['Name'].map(lambda x: _map(x, name, True))
  df['Name'] = df.apply(_Name, axis=1)
  #df['Pclass'] = df['Pclass'].map(lambda x: round(float(4-x)/3, 3) if not pd.isnull(x) else 0)
  df['Sex'] = df['Sex'].map(lambda x: 1 if x == 'female' else 0)
  df['Fare_range'] = df['Fare'].map(lambda x: min(int(x)/10 + 1, 21) if not pd.isnull(x) else 0)
  #df['Fare'] = df['Fare'].map(lambda x: int(x) if not pd.isnull(x) else 0)
  df['Age_range'] = df['Age'].map(lambda x: min(int(x)/10, 7) if not pd.isnull(x) else -1)
  df['Age'] = df['Age'].map(lambda x: int(x) if not pd.isnull(x) else 0)
  df['Cabin'] = df['Cabin'].map(lambda x: _map(x, cabin))
  df['Embarked'] = df['Embarked'].map(lambda x: _map(x, embarked))
  df['Family'] = df['SibSp'] + df['Parch']
  del df['Ticket']
  #del df['Name']
  #del df['SibSp']
  #del df['Parch']

  df.to_csv('{}/{}_process.csv'.format(data_dir, f), index=False)
  df = pd.read_csv('{}/{}_process.csv'.format(data_dir, f))
  
  cf = csv.writer(open('{}/{}'.format(data_dir, f), 'w'))
  for i, r in df.iterrows():
    cf.writerow(xv(r))

os.system('head -n 791 data/train > data/train_s')
os.system('tail -n 100 data/train > data/val')

