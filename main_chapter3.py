print('#####################################')
print('##  예제 3-1.  데이터 살펴보기         ##')
print('#####################################')
import pandas as pd

df=pd.read_csv('./part3/auto-mpg.csv', header=None)

df.columns=['mpg','cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin', 'name']

print(df.head())
print('\n')
print(df.tail)


print(df.shape)

print(df.info())

print(df.dtypes)
print('\n')

print(df.mpg.dtypes)

print(df.describe())
print('\n')
print(df.describe(include='all'))




print('#####################################')
print('##  예제 3-2.  데이터 개수 확인        ##')
print('#####################################')
import pandas as pd

df=pd.read_csv('./part3/auto-mpg.csv', header=None)

df.columns=['mpg','cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin', 'name']

print(df.count())
print('\n')

print(type(df.count()))

print(df['origin'])
unique_values=df['origin'].value_counts()
print(unique_values)
print('\n')

print(type(unique_values))




print('#####################################')
print('##  예제 3-3.  통계함수 적용하기        ')
print('#####################################')
import pandas as pd

df=pd.read_csv('./part3/auto-mpg.csv', header=None)

df.columns=['mpg','cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin', 'name']

print('Use mean()')
print(df.mean())
print('\n')

print(df['mpg'].mean())
print(df.mpg.mean())
print('\n')
print(df[['mpg','weight']].mean())

print('\n')

print('Use median()')
print(df.median())
print('\n')
print(df['mpg'].median())

print('Use max()')
print(df.max())
print('\n')
print(df['mpg'].max())

print('Use min()')
print(df.min())
print('\n')
print(df['mpg'].min())

print('Use std()')
print(df.std())
print('\n')
print(df['mpg'].std())

print('---------')
print('Use corr()')
print('---------')
print(df.corr())
print('\n')
print(df[['mpg','weight']].corr())


print('#####################################')
print('##  예제 3-4.  선그래프 그리기         ')
print('#####################################')
import pandas as pd

df=pd.read_excel('./part3/남북한발전전력량.xlsx')

print(df.head(6))
df_ns=df.iloc[[0,5], 2:]
df_ns.index=['South','North']
print(df_ns.columns)
print(df_ns.columns.map(int))
df_ns.columns=df_ns.columns.map(int)
print(df_ns.head())
print('\n')

df_ns.plot()

tdf_ns=df_ns.T
print(tdf_ns.head())
print('\n')
tdf_ns.plot()


print('#####################################')
print('##  예제 3-5.  막대그래프 그리기         ')
print('#####################################')
import pandas as pd

df=pd.read_excel('./part3/남북한발전전력량.xlsx')

print(df.head(6))
df_ns=df.iloc[[0,5], 2:]
df_ns.index=['South','North']
print(df_ns.columns)
print(df_ns.columns.map(int))
df_ns.columns=df_ns.columns.map(int)
print(df_ns.head())
print('\n')

tdf_ns=df_ns.T
print(tdf_ns.head())
print('\n')
tdf_ns.plot(kind='bar')


print('#####################################')
print('##  예제 3-6.  히스토그램 그리기         ')
print('#####################################')
import pandas as pd

df=pd.read_excel('./part3/남북한발전전력량.xlsx')

print(df.head(6))
df_ns=df.iloc[[0,5], 2:]
df_ns.index=['South','North']
print(df_ns.columns)
print(df_ns.columns.map(int))
df_ns.columns=df_ns.columns.map(int)
print(df_ns.head())
print('\n')

tdf_ns=df_ns.T
print(tdf_ns.head())
print('\n')
tdf_ns.plot(kind='hist')


print('#####################################')
print('##  예제 3-7.  산점도 그리기         ')
print('#####################################')
import pandas as pd

df=pd.read_csv('./part3/auto-mpg.csv', header=None)

df.columns=['mpg','cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin', 'name']

df.plot(x='weight', y='mpg', kind='scatter')


print('#####################################')
print('##  예제 3-7.  박스 플롯 그리기         ')
print('#####################################')
import pandas as pd

df=pd.read_csv('./part3/auto-mpg.csv', header=None)

df.columns=['mpg','cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin', 'name']

df[['mpg','cylinders']].plot(kind='box')






