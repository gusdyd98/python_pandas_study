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

print('훈련 데이터: ', X_train.shape)
print('검증 데이터: ', X_train.shape)

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

poly=PolynomialFeatures(degree=2)
X_train_poly=poly.fit_transform(X_train)

print('원 데이터 : ', X_train.shape)
print('2차향 변환 데이터 : ', X_train_poly.shape)

pr=LinearRegression()
pr.fit(X_train_poly, y_train)

X_test_poly=poly.fit_transform(X_test)
r_square=pr.score(X_test_poly, y_test)
print(r_square)


y_hat_test=pr.predict(X_test_poly)

fig=plt.figure(figsize=(10,5))
ax=fig.add_subplot(1,1,1)
ax.plot(X_train, y_train, 'o', label='Train Data')
ax.plot(X_test, y_hat_test, 'r+', label='Predicted Value')
ax.legend(loc='best')
plt.xlabel('weight')
plt.ylabel('mpg')
plt.show()
plt.close()



X_poly=poly.fit_transform(X)
y_hat=pr.predict(X_poly)

plt.figure(figsize=(10,5))
ax1=sns.distplot(y    , hist=False, label='y'            )
ax2=sns.distplot(y_hat, hist=False, label='y_hat', ax=ax1)
plt.show()
plt.close()


print('##########################################################')
print('## 예제7-3. 다중회귀분석                                   ')
print('##########################################################')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv('./part7/auto-mpg.csv', header=None)

df.columns=['mpg','cylinders','displacement', 'horsepower', 'weight',
            'acceleration', 'model year', 'origin', 'name']

df['horsepower'].replace('?', np.nan, inplace=True)
df.dropna(subset=['horsepower'], axis=0, inplace=True)
df['horsepower']=df['horsepower'].astype('float')

ndf=df[['mpg', 'cylinders', 'horsepower', 'weight']]

X=ndf[['cylinders', 'horsepower', 'weight']]
y=ndf['mpg']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)

print('훈련 데이터 : ', X_train.shape)
print('검증 데이터 : ', X_test .shape)


print('##########################################################')
print('## 예제7-4. KNN (k-Nearest-Neighbors) 분류 알고리즘          ')
print('##########################################################')
import pandas as pd
import seaborn as sns

df=sns.load_dataset('titanic')

print(df.head())

pd.set_option('display.max_columns', 15)
print(df.head())
print(df.info())

rdf=df.drop(['deck', 'embark_town'], axis=1)
print(rdf.columns.values)

rdf=rdf.dropna(subset=['age'], how='any', axis=0)
print(len(rdf))

most_freq=rdf['embarked'].value_counts(dropna=True).idxmax()
print(most_freq)
print('\n')

print(rdf.describe(include='all'))
print('\n')

rdf['embarked'].fillna(most_freq, inplace=True)

ndf=rdf[['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'embarked']]
print(ndf.head())

onehot_sex=pd.get_dummies(ndf['sex'])
print(onehot_sex)

ndf=pd.concat([ndf, onehot_sex], axis=1)
print(ndf.head())

onehot_embarked=pd.get_dummies(ndf['embarked'], prefix='town')
print(onehot_embarked)

ndf=pd.concat([ndf, onehot_embarked], axis=1)
print(ndf.head())


ndf.drop(['sex', 'embarked'], axis=1, inplace=True)
print(ndf.head())

X=ndf[['pclass', 'age', 'sibsp', 'parch', 'female', 'male',
       'town_C', 'town_Q', 'town_S']]
y=ndf['survived']

from sklearn import preprocessing
X=preprocessing.StandardScaler().fit(X).transform(X)

print(X)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.3, random_state=10)

print('train data 개수: ', X_train.shape)
print('test  data 개수: ', X_test.shape)


from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train, y_train)

y_hat=knn.predict(X_test)

print(y_hat[0:10])
print(y_test.values[0:10])


from sklearn import metrics
knn_matrix=metrics.confusion_matrix(y_test, y_hat)
print(knn_matrix)

knn_report=metrics.classification_report(y_test, y_hat)
print(knn_report)



print('##########################################################')
print('## 예제7-5. SVM 모형           ')
print('##########################################################')
import pandas as pd
import seaborn as sns

df=sns.load_dataset('titanic')

print(df.head())

pd.set_option('display.max_columns', 15)
print(df.head())
print(df.info())

rdf=df.drop(['deck', 'embark_town'], axis=1)
print(rdf.columns.values)

rdf=rdf.dropna(subset=['age'], how='any', axis=0)
print(len(rdf))

most_freq=rdf['embarked'].value_counts(dropna=True).idxmax()
print(most_freq)
print('\n')

print(rdf.describe(include='all'))
print('\n')

rdf['embarked'].fillna(most_freq, inplace=True)

ndf=rdf[['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'embarked']]
print(ndf.head())

onehot_sex=pd.get_dummies(ndf['sex'])
print(onehot_sex)

ndf=pd.concat([ndf, onehot_sex], axis=1)
print(ndf.head())

onehot_embarked=pd.get_dummies(ndf['embarked'], prefix='town')
print(onehot_embarked)

ndf=pd.concat([ndf, onehot_embarked], axis=1)
print(ndf.head())


ndf.drop(['sex', 'embarked'], axis=1, inplace=True)
print(ndf.head())

X=ndf[['pclass', 'age', 'sibsp', 'parch', 'female', 'male',
       'town_C', 'town_Q', 'town_S']]
y=ndf['survived']

from sklearn import preprocessing
X=preprocessing.StandardScaler().fit(X).transform(X)

print(X)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.3, random_state=10)

print('train data 개수: ', X_train.shape)
print('test  data 개수: ', X_test.shape)


from sklearn import svm

svm_model=svm.SVC(kernel='rbf')

svm_model.fit(X_train, y_train)

y_hat=svm_model.predict(X_test)

print(y_hat[0:10])
print(y_test.values[0:10])

from sklearn import metrics

svm_matrix=metrics.confusion_matrix(y_test, y_hat)
print(svm_matrix)
print('\n')

svm_report=metrics.classification_report(y_test, y_hat)
print(svm_report)



print('##########################################################')
print('## 예제7-6. Decision Tree 모형           ')
print('##########################################################')
import pandas as pd
import numpy as np

uci_path='https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data'
df=pd.read_csv(uci_path, header=None)

df.columns=['id', 'clump', 'cell_size', 'cell_shape', 'adhesion', 'epithlial',
            'bare_nuclei', 'chromatin', 'normal_nucleoli', 'mitoses', 'class']

pd.set_option('display.max_columns', 15)

print(df.head())
print('\n')

print(df.info())
print('\n')

print(df.describe())

print(df['bare_nuclei'].unique())
print('\n')

df['bare_nuclei'].replace('?', np.nan, inplace=True)
df.dropna(subset=['bare_nuclei'], axis=0, inplace=True)
df['bare_nuclei']=df['bare_nuclei'].astype('int')

print(df.describe())


X=df[['clump', 'cell_size', 'cell_shape', 'adhesion', 'epithlial',
      'bare_nuclei', 'chromatin', 'normal_nucleoli', 'mitoses']]
y=df['class']

from sklearn import preprocessing
X=preprocessing.StandardScaler().fit(X).transform(X)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

print('train data 개수 : ', X_train.shape)
print('test  data 개수 : ', X_test.shape)

from sklearn import tree

tree_model=tree.DecisionTreeClassifier(criterion='entropy', max_depth=5)

tree_model.fit(X_train, y_train)

y_hat=tree_model.predict(X_test)


print(y_hat[0:10])
print(y_test.values[0:10])

from sklearn import metrics

tree_matrix=metrics.confusion_matrix(y_test, y_hat)
print(tree_matrix)
print('\n')

tree_report=metrics.classification_report(y_test, y_hat)
print(tree_report)


print('##########################################################')
print('## 예제7-7. k-means 군집 분석            ')
print('##########################################################')
import pandas as pd
import matplotlib.pyplot as plt

uci_path='https://archive.ics.uci.edu/ml/machine-learning-databases/\
00292/Wholesale%20customers%20data.csv'
df=pd.read_csv(uci_path, header=0)

print(df.head())
print('\n')

print(df.info())
print('\n')

print(df.describe())

X=df.iloc[:, :]
print(X[:5])
print('\n')

from sklearn import preprocessing
X = preprocessing.StandardScaler().fit(X).transform(X)

print(X[:5])

from sklearn import cluster

kmeans=cluster.KMeans(init='k-means++', n_clusters=5, n_init=10)

kmeans.fit(X)

cluster_label=kmeans.labels_
print(cluster_label)
print('\n')

df['Cluster']=cluster_label
print(df.head())


df.plot(kind='scatter', x='Grocery', y='Frozen', c='Cluster', cmap='Set1',
        colorbar=False, figsize=(10,10))
df.plot(kind='scatter', x='Milk', y='Delicassen', c='Cluster', cmap='Set1',
        colorbar=True, figsize=(10,10))

plt.show()
plt.close()
plt.close()

mask=(df['Cluster']==0) | (df['Cluster']==4)
ndf=df[~mask]


ndf.plot(kind='scatter', x='Grocery', y='Frozen', c='Cluster', cmap='Set1',
        colorbar=False, figsize=(10,10))
ndf.plot(kind='scatter', x='Milk', y='Delicassen', c='Cluster', cmap='Set1',
        colorbar=True, figsize=(10,10))

plt.show()
plt.close()
plt.close()


print('##########################################################')
print('## 예제7-8. DBSCAN  군집 분석            ')
print('##########################################################')
import pandas as pd
import folium

file_path='./part7/2016_middle_shcool_graduates_report.xlsx'

df=pd.read_excel(file_path, header=0)

pd.set_option('display.width', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 10)
pd.set_option('display.max_colwidth', 20)
pd.set_option('display.unicode.east_asian_width', True)

print(df.columns.values)

print(df.head())
print('\n')

print(df.info())
print('\n')

print(df.describe())

mschool_map=folium.Map(location=[37.55, 126.98], tiles='Stamen Terrain',
                       zoom_start=12)

for name, lat, lng in zip(df.학교명, df.위도, df.경도):
    folium.CircleMarker([lat, lng],
                        radius=5,
                        color='brown',
                        fill=True,
                        fill_color='coral',
                        fill_opacity=0.7,
                        popup=name
                        ).add_to(mschool_map)

    mschool_map.save('./seoul_mschool_location.html')

from sklearn import preprocessing

label_encoder=preprocessing.LabelEncoder()
onehot_encoder=preprocessing.OneHotEncoder()

onehot_location=label_encoder.fit_transform(df['지역'])
onehot_code=label_encoder.fit_transform(df['코드'])
onehot_type=label_encoder.fit_transform(df['유형'])
onehot_day=label_encoder.fit_transform(df['주야'])


df['location']=onehot_location
df['code']=onehot_code
df['type']=onehot_type
df['day']=onehot_day

print(df.head())

from sklearn import cluster

columns_list=[9,10,13]
X=df.iloc[:, columns_list]
print(X[:5])
print('\n')

X=preprocessing.StandardScaler().fit(X).transform(X)

dbm=cluster.DBSCAN(eps=0.2, min_samples=5)

dbm.fit(X)

cluster_label=dbm.labels_
print(cluster_label)
print('\n')



