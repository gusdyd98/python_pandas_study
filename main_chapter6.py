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


