print('##########################################################')
print('## 예제7-1. 단순회귀분석                                   ')
print('##########################################################')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv('./part7/auto-mpg.csv', header=None)

df.columns=['mpg','cylinders', 'displacement', 'horsepower', 'weight',
            'acceleration', 'model year', 'origin', 'name']

print(df.head())
print('\n')

pd.set_option('display.max_columns', 10)
print(df.head())

print(df.info())
print('\n')

print(df.describe())

print(df['horsepower'].unique())
print('\n')

df['horsepower'].replace('?', np.nan, inplace=True)
df.dropna(subset=['horsepower'], axis=0, inplace=True)
df['horsepower']=df['horsepower'].astype('float')

print(df.describe())

ndf=df[['mpg', 'cylinders', 'horsepower', 'weight']]
print(ndf.head())

ndf.plot(kind='scatter', x='weight', y='mpg', c='coral', s=10, figsize=(10,5))
plt.show()
plt.close()


fig=plt.figure(figsize=(10,5))
ax1=fig.add_subplot(1,2,1)
ax2=fig.add_subplot(1,2,2)
sns.regplot(x='weight',y='mpg', data=ndf, ax=ax1)
sns.regplot(x='weight',y='mpg', data=ndf, ax=ax2, fit_reg=False)
plt.show()
plt.close()
plt.close()


sns.jointplot(x='weight', y='mpg', data=ndf)
sns.jointplot(x='weight', y='mpg', kind='reg', data=ndf)
plt.show()
plt.close()
plt.close()


grid_ndf=sns.pairplot(ndf)
plt.show()
plt.close()


X=ndf[['weight']]
y=ndf['mpg']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,
                                                  y,
                                                  test_size=0.3,
                                                  random_state=10)

print('train data 개수: ', len(X_train))
print('test data 개수: ', len(X_test))



from sklearn.linear_model import LinearRegression

lr=LinearRegression()
lr.fit(X_train, y_train)

r_square=lr.score(X_test, y_test)
print(r_square)


print('기울기 a: ', lr.coef_)
print('\n')

print('y절편 b:', lr.intercept_)


y_hat=lr.predict(X)

plt.figure(figsize=(10,5))
ax1=sns.distplot(y, hist=False, label="y")
ax2=sns.distplot(y_hat, hist=False, label="y_hat", ax=ax1)
plt.show()
plt.close()


print('##########################################################')
print('## 예제7-2. 다항회귀분석                                   ')
print('##########################################################')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv('./part7/auto-mpg.csv', header=None)

df.columns=['mpg','cylinders', 'displacement', 'horsepower', 'weight',
            'acceleration', 'model year', 'origin', 'name']

pd.set_option('display.max_columns', 10)

df['horsepower'].replace('?', np.nan, inplace=True)
df.dropna(subset=['horsepower'], axis=0, inplace=True)
df['horsepower']=df['horsepower'].astype('float')

ndf=df[['mpg', 'cylinders', 'horsepower', 'weight']]

X=ndf[['weight']]
y=ndf['mpg']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,
                                                  y,
                                                  test_size=0.3,
                                                  random_state=10)

print('train data : ',X_train.shape)
print('test data : ', X_test.shape)

