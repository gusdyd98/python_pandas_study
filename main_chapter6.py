print('##############################################')
print('## 예제 6-1. 시리즈 원소에 apply() 적용하기 ')
print('##############################################')

import seaborn as sns

titanic=sns.load_dataset('titanic')
df=titanic.loc[:, ['age', 'fare']]
print(df.head())
df['ten']=10
print(df.head())


def add_10(n):
    return n+10

def add_two_obj(a,b):
    return a+b

print(add_10(10))
print(add_two_obj(10, 10))

sr1=df['age'].apply(add_10)
print(sr1.head())
print('\n')

sr2=df['age'].apply(add_two_obj, b=10)
print(sr2.head())
print('\n')

sr3=df['age'].apply(lambda x: add_10(x))
print(sr3.head())
print('\n')


print('##############################################')
print('## 예제 6-2. 데이터프레임 원소에 applymap() 적용하기')
print('##############################################')

import seaborn as sns

titanic=sns.load_dataset('titanic')
df=titanic.loc[:, ['age', 'fare']]
print(df.head())
print('\n')

def add_10(n):
    return n+10

df_map=df.applymap(add_10)
print(df_map.head())


print('##############################################')
print('## 예제 6-3. 데이터프레임 원소에 apply(axis=0) 적용하기')
print('##############################################')

import seaborn as sns

titanic=sns.load_dataset('titanic')
df=titanic.loc[:, ['age', 'fare']]
print(df.head())
print('\n')

def missing_value(series):
    return series.isnull()

result=df.apply(missing_value, axis=0)
print(result.head())
print('\n')
print(type(result))

print('##############################################')
print('## 예제 6-4. 데이터프레임 원소에 apply(axis=0) 적용하기')
print('##############################################')

import seaborn as sns

titanic=sns.load_dataset('titanic')
df=titanic.loc[:, ['age', 'fare']]
print(df.head())
print('\n')

def min_max(x):
    return x.max() - x.min()

result=df.apply(min_max)
print(result.head())
print('\n')
print(type(result))

print('##############################################')
print('## 예제 6-5. 데이터프레임에 apply() 적용하기')
print('##############################################')

import seaborn as sns

titanic=sns.load_dataset('titanic')
df=titanic.loc[:, ['age', 'fare']]
df['ten']=10
print(df.head())
print('\n')

def add_two_obj(a,b):
    return a + b

df['add']=df.apply(lambda x: add_two_obj(x['age'], x['ten']), axis=1)
print(df.head())


print('##############################################')
print('## 예제 6-6. 데이터프레임에 pipe() 적용하기')
print('##############################################')

import seaborn as sns

titanic=sns.load_dataset('titanic')
df=titanic.loc[:, ['age', 'fare']]
print(df.head())
print('\n')

def missing_value(x):
    return x.isnull()

def missing_count(x):
    return missing_value(x).sum()

def total_number_missing(x):
    return missing_count(x).sum()

result_df=df.pipe(missing_value)
print(result_df.head())
print(type(result_df))


result_series=df.pipe(missing_count)
print(result_series)
print(type(result_series))

result_value=df.pipe(total_number_missing)
print(result_value)
print(type(result_value))


print('##############################################')
print('## 예제 6-7. 열 순서 바꾸기           ')
print('##############################################')

import seaborn as sns

titanic=sns.load_dataset('titanic')
df=titanic.loc[0:4, 'survived':'age']
print(df)
print('\n')

columns=list(df.columns.values)
print(columns, '\n')

columns_sorted = sorted(columns)
print(columns_sorted, '\n')
df_sorted=df[columns_sorted]
print(df_sorted, '\n')

columns_reversed=list(reversed(columns))
df_reversed=df[columns_reversed]
print(df_reversed, '\n')


columns_customed=['pclass', 'sex', 'age', 'survived']
df_customed=df[columns_customed]
print(df_customed)


print('##############################################')
print('## 예제 6-8. 열 분리하기             ')
print('##############################################')

import pandas as pd
import seaborn as sns

df=pd.read_excel('./part6/주가데이터.xlsx')
print(df.head())
print(df.dtypes)
print('\n')

df['연월일']=df['연월일'].astype('str')
dates=df['연월일'].str.split('-')
print(dates.head())

df['연']=dates.str.get(0)
df['월']=dates.str.get(1)
df['일']=dates.str.get(2)
print(df.head())


print('##############################################')
print('## 예제 6-9. 불린 인덱싱             ')
print('##############################################')

import pandas as pd
import seaborn as sns

titanic=sns.load_dataset('titanic')

mask1=(titanic.age >= 10) & (titanic.age < 20)
df_teenage=titanic.loc[mask1, :]
print(df_teenage.head())

mask2=(titanic.age<10) & (titanic.sex == 'female')
df_female_under10 = titanic.loc[mask2, :]
print(df_female_under10.head())

mask3=(titanic.age<10) | (titanic.age>=60)
df_under10_morethan60 = titanic.loc[mask3,['age', 'sex', 'alone']]
print(df_under10_morethan60.head())


print('##############################################')
print('## 예제 6-10. isin() 필터링             ')
print('##############################################')

import pandas as pd
import seaborn as sns

titanic=sns.load_dataset('titanic')

pd.set_option('display.max_columns', 10)

mask3=titanic['sibsp']==3
mask4=titanic['sibsp']==4
mask5=titanic['sibsp']==5
df_boolean=titanic[mask3|mask4|mask5]
print(df_boolean.head())

isin_filter=titanic['sibsp'].isin([3,4,5])
df_isin=titanic[isin_filter]
print(df_isin.head())



print('##############################################')
print('## 예제 6-11. 데이터프레임 연결(concatenate)      ')
print('##############################################')

import pandas as pd

df1=pd.DataFrame({'a':['a0', 'a1', 'a2', 'a3'],
                  'b':['b0', 'b1', 'b2', 'b3'],
                  'c':['c0', 'c1', 'c2', 'c3']},
                 index=[0,1,2,3])

df2=pd.DataFrame({'a':['a2', 'a3', 'a4', 'a5'],
                  'b':['b2', 'b3', 'b4', 'b5'],
                  'c':['c2', 'c3', 'c4', 'c5'],
                  'd':['d2', 'd3', 'd4', 'd5']},
                 index=[2,3,4,5])

print(df1, '\n')
print(df2, '\n')

result1=pd.concat([df1, df2])
print(result1, '\n')

result2=pd.concat([df1, df2], ignore_index=True)
print(result2, '\n')

result3=pd.concat([df1, df2], axis=1)
print(result3, '\n')

result3_in=pd.concat([df1, df2], axis=1, join='inner')
print(result3_in, '\n')

sr1=pd.Series(['e0', 'e1', 'e2', 'e3'], name='e')
sr2=pd.Series(['f0', 'f1', 'f2'], name='f', index=[3,4,5])
sr3=pd.Series(['g0', 'g1', 'g2', 'g3'], name='g')

result4=pd.concat([df1, sr1], axis=1)
print(result4, '\n')

result5=pd.concat([df2, sr2], axis=1, sort=True)
print(result5, '\n')

result6=pd.concat([sr1, sr3], axis=1)
print(result6, '\n')

result7=pd.concat([sr1, sr3], axis=0)
print(result7, '\n')


