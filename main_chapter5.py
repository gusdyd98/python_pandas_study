print('######################################')
print('##  예제 5-1. 누락 데이터 확인하기   ')
print('######################################')
import pandas as pd
import seaborn as sns

pd.get_option("display.max_columns", 999)
pd.set_option("display.max_columns", None)

df=sns.load_dataset('./part5/titanic')

df.head()


df.info()

nan_deck=df['deck'].value_counts(dropna=False)
nan_deck

print(df.head().isnull())

print(df.head().notnull())

print(df.head().isnull().sum(axis=0))


print('######################################')
print('##  예제 5-2. 누락 데이터 제거하기   ')
print('######################################')
import pandas as pd
import seaborn as sns

pd.get_option("display.max_columns", 999)
pd.set_option("display.max_columns", None)

df=sns.load_dataset('./part5/titanic')

missing_df=df.isnull()
for col in missing_df.columns:
    missing_count=missing_df[col].value_counts()

    try:
        print(col, ': ', missing_count[True])
    except:
        print(col, ': ', 0)


df_thresh=df.dropna(axis=1, thresh=500)
print(df_thresh.columns)

df_age=df.dropna(subset=['age'], how='any', axis=0)
print(len(df_age))


print('######################################')
print('##  예제 5-3. 평균으로 누락 데이터 바꾸기   ')
print('######################################')
import pandas as pd
import seaborn as sns

pd.get_option("display.max_columns", 999)
pd.set_option("display.max_columns", None)

df=sns.load_dataset('./part5/titanic')

print(df['age'].head(10))
print('\n')

mean_age=df['age'].mean(axis=0)
df['age'].fillna(mean_age, inplace=True)

print(df['age'].head(10))


print('######################################')
print('##  예제 5-4. 가장 많이 나타나는 값으로 바꾸기    ')
print('######################################')
import pandas as pd
import seaborn as sns

pd.get_option("display.max_columns", 999)
pd.set_option("display.max_columns", None)

df=sns.load_dataset('./part5/titanic')

print(df['embark_town'][825:830])
print('\n')

most_freq=df['embark_town'].value_counts(dropna=True).idxmax()
print(most_freq)
print('\n')

df['embark_town'].fillna(most_freq, inplace=True)

print(df['embark_town'][825:830])


print('######################################')
print('##  예제 5-5. 이웃한 값으로 바꾸기     ')
print('######################################')
import pandas as pd
import seaborn as sns

pd.get_option("display.max_columns", 999)
pd.set_option("display.max_columns", None)

df=sns.load_dataset('./part5/titanic')

print(df['embark_town'][825:830])
print('\n')

df['embark_town'].fillna(method='ffill', inplace=True)

print(df['embark_town'][825:830])

print('######################################')
print('##  예제 5-6. 중복 데이터 확인하기     ')
print('######################################')
import pandas as pd
import seaborn as sns

pd.get_option("display.max_columns", 999)
pd.set_option("display.max_columns", None)

df=pd.DataFrame({'c1':['a','a','b','a','b'],
                 'c2':[1,1,1,2,2],
                 'c3':[1,1,2,2,2]})

print(df)
print('\n')

df_dup=df.duplicated()
print(df_dup)
print('\n')

col_dup=df['c2'].duplicated()
print(col_dup)

print('######################################')
print('##  예제 5-7. 중복 데이터 제거하기     ')
print('######################################')
import pandas as pd
import seaborn as sns

pd.get_option("display.max_columns", 999)
pd.set_option("display.max_columns", None)

df=pd.DataFrame({'c1':['a','a','b','a','b'],
                 'c2':[1,1,1,2,2],
                 'c3':[1,1,2,2,2]})

print(df)
print('\n')

df2=df.drop_duplicates()
print(df2)
print('\n')

df3=df.drop_duplicates(subset=['c2','c3'])
print(df3)

