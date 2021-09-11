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

print('######################################')
print('##  예제 5-8. 단위 환산하기     ')
print('######################################')
import pandas as pd
import seaborn as sns

pd.get_option("display.max_columns", 999)
pd.set_option("display.max_columns", None)

df=pd.read_csv('./part5/auto-mpg.csv', header= None)

df.columns=['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin', 'name']

print(df.head(3))
print('\n')

mpg_to_kpl=1.60934/3.78541

df['kpl']=df['mpg'] * mpg_to_kpl
print(df.head(3))
print('\n')

df['kpl']=df['kpl'].round(2)
print(df.head(3))


print('######################################')
print('##  예제 5-9. 자료형 변환하기     ')
print('######################################')
import pandas as pd

pd.get_option("display.max_columns", 999)
pd.set_option("display.max_columns", None)

df=pd.read_csv('./part5/auto-mpg.csv', header= None)

df.columns=['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin', 'name']

print(df.dtypes)
print('\n')


print(df['horsepower'].unique())
print('\n')

import numpy as np

df['horsepower'].replace('?', np.nan, inplace=True)
df.dropna(subset=['horsepower'], axis=0, inplace=True)
df['horsepower']=df['horsepower'].astype('float')

print(df['horsepower'].dtypes)


print(df['horsepower'].unique())


print(df['origin'].unique())
df['origin'].replace({1:'USA', 2:'EU', 3:'JPN'}, inplace=True)
print(df['origin'].unique())
print(df['origin'].dtypes)


df['origin']=df['origin'].astype('category')
print(df['origin'].dtypes)

df['origin']=df['origin'].astype('str')
print(df['origin'].dtypes)


print(df['model year'].sample(3))
df['model year']=df['model year'].astype('category')
print(df['model year'].sample(3))


print('######################################')
print('##  예제 5-10. 데이터 구분하기      ')
print('######################################')
import pandas as pd
import numpy as np

pd.get_option("display.max_columns", 999)
pd.set_option("display.max_columns", None)

df=pd.read_csv('./part5/auto-mpg.csv', header= None)

df.columns=['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin', 'name']

df['horsepower'].replace('?', np.nan, inplace=True)
df.dropna(subset=['horsepower'], axis=0, inplace=True)
df['horsepower']=df['horsepower'].astype('float')

count, bin_dividers = np.histogram(df['horsepower'], bins=3)
print(count)
print(bin_dividers)

bin_names=['저출력', '보통출력', '고출력']

df['hp_bin']=pd.cut(x=df['horsepower'],
                    bins=bin_dividers,
                    labels=bin_names,
                    include_lowest=True)

print(df[['horsepower', 'hp_bin']].head(15))


print('######################################')
print('##  예제 5-11. 더미 변수 구하기      ')
print('######################################')
import pandas as pd
import numpy as np

pd.get_option("display.max_columns", 999)
pd.set_option("display.max_columns", None)

df=pd.read_csv('./part5/auto-mpg.csv', header= None)

df.columns=['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin', 'name']

df['horsepower'].replace('?', np.nan, inplace=True)
df.dropna(subset=['horsepower'], axis=0, inplace=True)
df['horsepower']=df['horsepower'].astype('float')

count, bin_dividers = np.histogram(df['horsepower'], bins=3)
print(count)
print(bin_dividers)

bin_names=['저출력', '보통출력', '고출력']

df['hp_bin']=pd.cut(x=df['horsepower'],
                    bins=bin_dividers,
                    labels=bin_names,
                    include_lowest=True)

horsepower_dummies=pd.get_dummies(df['hp_bin'])
print(horsepower_dummies.head(15))


print('######################################')
print('##  예제 5-12. 원핫인코딩      ')
print('######################################')
import pandas as pd
import numpy as np
from sklearn import preprocessing

pd.get_option("display.max_columns", 999)
pd.set_option("display.max_columns", None)

df=pd.read_csv('./part5/auto-mpg.csv', header= None)

df.columns=['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin', 'name']

df['horsepower'].replace('?', np.nan, inplace=True)
df.dropna(subset=['horsepower'], axis=0, inplace=True)
df['horsepower']=df['horsepower'].astype('float')

count, bin_dividers = np.histogram(df['horsepower'], bins=3)
print(count)
print(bin_dividers)

bin_names=['저출력', '보통출력', '고출력']

df['hp_bin']=pd.cut(x=df['horsepower'],
                    bins=bin_dividers,
                    labels=bin_names,
                    include_lowest=True)

horsepower_dummies=pd.get_dummies(df['hp_bin'])
print(horsepower_dummies.head(15))

label_encoder=preprocessing.LabelEncoder()
onehot_encoder=preprocessing.OneHotEncoder()

onehot_labeled=label_encoder.fit_transform(df['hp_bin'].head(15))
print(onehot_labeled)
print(type(onehot_labeled))

onehot_reshaped=onehot_labeled.reshape(len(onehot_labeled), 1)
print(onehot_reshaped)
print(type(onehot_labeled))

onehot_fitted=onehot_encoder.fit_transform(onehot_reshaped)
print(onehot_fitted)
print(type(onehot_fitted))



print('######################################')
print('##  예제 5-13. 정규화      ')
print('######################################')
import pandas as pd
import numpy as np
from sklearn import preprocessing

pd.get_option("display.max_columns", 999)
pd.set_option("display.max_columns", None)

df=pd.read_csv('./part5/auto-mpg.csv', header= None)

df.columns=['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin', 'name']

df['horsepower'].replace('?', np.nan, inplace=True)
df.dropna(subset=['horsepower'], axis=0, inplace=True)
df['horsepower']=df['horsepower'].astype('float')

print(df.horsepower.describe())
print('\n')

df.horsepower  = df.horsepower/abs(df.horsepower.max())

print(df.horsepower.head())
print('\n')
print(df.horsepower.describe())
print('\n')

min_x=df.horsepower-df.horsepower.min()
min_max=df.horsepower.max()-df.horsepower.min()
df.horsepower = min_x/min_max

print(df.horsepower.head())
print('\n')
print(df.horsepower.describe())


print('######################################')
print('##  예제 5-14. 문자열을 Timestamp로 변환      ')
print('######################################')
import pandas as pd
from sklearn import preprocessing

pd.get_option("display.max_columns", 999)
pd.set_option("display.max_columns", None)

df=pd.read_csv('./part5/stock-data.csv')

print(df.head())
print('\n')
print(df.info())

df['new_Date']=pd.to_datetime(df['Date'])


print(df.head())
print('\n')
print(df.info())
print('\n')
print(type(df['new_Date'][0]))


print('######################################')
print('##  예제 5-15. 문자열을 Timestamp로 변환      ')
print('######################################')

df.set_index('new_Date', inplace=True)
df.drop('Date', axis=1, inplace=True)

print(df.head())
print('\n')
print(df.info())


print('######################################')
print('##  예제 5-16. Timestamp를 Period로 변환  ')
print('######################################')
import pandas as pd

pd.get_option("display.max_columns", 999)
pd.set_option("display.max_columns", None)

dates=['2018-01-01', '2020-03-01', '2021-06-01']

ts_dates=pd.to_datetime(dates)
print(ts_dates)
print('\n')

pr_day=ts_dates.to_period(freq='D')
print(pr_day)

pr_month=ts_dates.to_period(freq='M')
print(pr_month)

pr_year=ts_dates.to_period(freq='A')
print(pr_year)


print('######################################')
print('##  예제 5-17. Timestamp 배열 만들기   ')
print('######################################')
import pandas as pd

pd.get_option("display.max_columns", 999)
pd.set_option("display.max_columns", None)

ts_ms=pd.date_range(start='2019-01-01',
                    end=None,
                    periods=6,
                    freq='MS',
                    tz='Asia/Seoul')
print(ts_ms)
print('\n')


ts_me=pd.date_range(start='2019-01-01',
                    periods=6,
                    freq='M',
                    tz='Asia/Seoul')
print(ts_me)
print('\n')


ts_3m=pd.date_range(start='2019-01-01',
                    periods=6,
                    freq='3M',
                    tz='Asia/Seoul')
print(ts_3m)
print('\n')

print('######################################')
print('##  예제 5-18. Period    배열 만들기   ')
print('######################################')
import pandas as pd

pd.get_option("display.max_columns", 999)
pd.set_option("display.max_columns", None)

pr_m=pd.period_range(start='2019-01-01',
                    end=None,
                    periods=3,
                    freq='M')
print(pr_m)
print('\n')

pr_h=pd.period_range(start='2019-01-01',
                     end=None,
                     periods=3,
                     freq='H')
print(pr_h)
print('\n')


pr_2h=pd.period_range(start='2019-01-01',
                     end=None,
                     periods=3,
                     freq='2H')
print(pr_2h)
print('\n')

print('######################################')
print('##  예제 5-19. 날짜 데이터 분리    ')
print('######################################')
import pandas as pd

pd.get_option("display.max_columns", 999)
pd.set_option("display.max_columns", None)

df=pd.read_csv('./part5/stock-data.csv')


df['new_Date']=pd.to_datetime(df['Date'])


print(df.head())
print('\n')

df['Year']=df['new_Date'].dt.year
df['Month']=df['new_Date'].dt.month
df['Day']=df['new_Date'].dt.day

print(df.head())


df['Date_yr']=df['new_Date'].dt.to_period(freq='A')
df['Date_m']=df['new_Date'].dt.to_period(freq='M')
print(df.head())

df.set_index('Date_m', inplace=True)
print(df.head())


print('######################################')
print('##  예제 5-20. 날짜 인덱스 활용     ')
print('######################################')
import pandas as pd

pd.get_option("display.max_columns", 999)
pd.set_option("display.max_columns", None)

df=pd.read_csv('./part5/stock-data.csv')


df['new_Date']=pd.to_datetime(df['Date'])
df.set_index('new_Date', inplace=True)

print(df.head())
print('\n')
print(df.index)


df_y=df['2018']
print("df_y.head()")
print(df_y.head())
print('\n')
df_ym=df.loc['2018-07']
print(df_ym)
print('\n')
df_ym_cols=df.loc['2018-07','Start':'High']
print(df_ym_cols)
print('\n')
df_ymd=df['2018-07-02']
print(df_ymd)
print('\n')
df_ymd_range=df['2018-06-20':'2018-06-25']
print(df_ymd_range)


today=pd.to_datetime('2018-12-25')
df['time_delta']=today-df.index
df.set_index('time_delta', inplace=True)
df_180=df['180 days':'189 days']
print(df_180)

