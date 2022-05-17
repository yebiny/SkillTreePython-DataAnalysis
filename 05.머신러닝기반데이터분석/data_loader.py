import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
  
def mpg_loader(test_size=0.2, rs=1):
  df = sns.load_dataset('mpg')
  print('===== 데이터 확인 ====')
  print('* shape: ',df.shape)
  print(df.head())

  print('===== 범주 자료 제거 ====')
  df = df.drop(['origin', 'name'], axis=1)
  print('* shape: ',df.shape)

  print('===== 결측치 제거 ====')
  df = df.dropna(axis=0)
  print('* shape: ',df.shape)

  print('===== 정규화 ====')
  df=df.apply(lambda x: (x-x.mean())/ x.std(), axis=0)
  print(df.head())

  print('===== 넘파이 변환 ====')
  ds = df.to_numpy()
  X, y = ds[:, 1:], ds[:, 0]
  print(X.shape, y.shape)

  print('===== 데이터 분할 ====')

  X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=rs)
  print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
  return X_train, X_test, y_train, y_test



from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

def iris_loader(n_class, test_size=0.2, rs=1):
  iris = load_iris()
  X = iris.data
  y = iris.target
  print(X.shape, y.shape, X[0], y[0])
  
  if n_class==2:
    X = X[y!=2]
    y = y[y!=2]
    y = np.where(y==1, 1, 0)
  if n_class==3:
    ohe = OneHotEncoder(sparse=False)
    y = ohe.fit_transform(np.expand_dims(y,1))
  
  X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=rs)
   
  print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
  return X_train, X_test, y_train, y_test