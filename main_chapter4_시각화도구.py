print('######################################')
print('##  예제 4-1. 선 그래프 그리기 ')
print('######################################')

import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_excel('./part4/시도별 전출입 인구수.xlsx', na_values=0, header=0)

print(df.head())


df=df.fillna(method='ffill')

print(df.head())

mask=(df['전출지별'] == '서울특별시') & (df['전입지별'] != '서울특별시')
df_seoul=df[mask]
print(df_seoul.head())

df_seoul=df_seoul.drop(['전출지별'], axis=1)
df_seoul.rename({'전입지별':'전입지'}, axis=1, inplace=True)
df_seoul.set_index('전입지', inplace=True)

print(df_seoul.head())

sr_one=df_seoul.loc['경기도']
print(sr_one.head())
sr_one.head()

plt.plot(sr_one.index, sr_one.values)

plt.plot(sr_one)


print('######################################')
print('##  예제 4-2. 차트 제목, 축 이름 추가 ')
print('######################################')

import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_excel('./part4/시도별 전출입 인구수.xlsx', na_values=0, header=0)

print(df.head())


df=df.fillna(method='ffill')

print(df.head())

mask=(df['전출지별'] == '서울특별시') & (df['전입지별'] != '서울특별시')
df_seoul=df[mask]
print(df_seoul.head())

df_seoul=df_seoul.drop(['전출지별'], axis=1)
df_seoul.rename({'전입지별':'전입지'}, axis=1, inplace=True)
df_seoul.set_index('전입지', inplace=True)

print(df_seoul.head())

sr_one=df_seoul.loc['경기도']
print(sr_one.head())
sr_one.head()

plt.plot(sr_one.index, sr_one.values)

plt.title('서울->경기 인구 이동')

plt.xlabel('기간')
plt.ylabel('이동 인구수')

plt.show()

print('######################################')
print('##  예제 4-3. 폰트 문제 해결하기 ')
print('######################################')

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

font_path='./part4/malgun.ttf'
font_name=font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)

df=pd.read_excel('./part4/시도별 전출입 인구수.xlsx', na_values=0, header=0)

print(df.head())


df=df.fillna(method='ffill')

print(df.head())

mask=(df['전출지별'] == '서울특별시') & (df['전입지별'] != '서울특별시')
df_seoul=df[mask]
print(df_seoul.head())

df_seoul=df_seoul.drop(['전출지별'], axis=1)
df_seoul.rename({'전입지별':'전입지'}, axis=1, inplace=True)
df_seoul.set_index('전입지', inplace=True)

print(df_seoul.head())

sr_one=df_seoul.loc['경기도']
print(sr_one.head())
sr_one.head()

plt.plot(sr_one.index, sr_one.values)

plt.title('서울->경기 인구 이동')

plt.xlabel('기간')
plt.ylabel('이동 인구수')

plt.show()


print('######################################')
print('##  예제 4-4. 그래프 꾸미기  ')
print('######################################')

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

font_path='./part4/malgun.ttf'
font_name=font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)

df=pd.read_excel('./part4/시도별 전출입 인구수.xlsx', na_values=0, header=0)

print(df.head())


df=df.fillna(method='ffill')

print(df.head())

mask=(df['전출지별'] == '서울특별시') & (df['전입지별'] != '서울특별시')
df_seoul=df[mask]
print(df_seoul.head())

df_seoul=df_seoul.drop(['전출지별'], axis=1)
df_seoul.rename({'전입지별':'전입지'}, axis=1, inplace=True)
df_seoul.set_index('전입지', inplace=True)

print(df_seoul.head())

sr_one=df_seoul.loc['경기도']
print(sr_one.head())
sr_one.head()

plt.figure(figsize=(14,5))
plt.xticks(rotation='vertical')

plt.plot(sr_one.index, sr_one.values)

plt.title('서울->경기 인구 이동')
plt.xlabel('기간')
plt.ylabel('이동 인구수')

plt.legend(labels=['서울->경기'], loc='best')

plt.show()



print('######################################')
print('##  예제 4-5. 스타일 서식 지정           ')
print('######################################')

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

font_path='./part4/malgun.ttf'
font_name=font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)

df=pd.read_excel('./part4/시도별 전출입 인구수.xlsx', na_values=0, header=0)

print(df.head())


df=df.fillna(method='ffill')

print(df.head())

mask=(df['전출지별'] == '서울특별시') & (df['전입지별'] != '서울특별시')
df_seoul=df[mask]
print(df_seoul.head())

df_seoul=df_seoul.drop(['전출지별'], axis=1)
df_seoul.rename({'전입지별':'전입지'}, axis=1, inplace=True)
df_seoul.set_index('전입지', inplace=True)

print(df_seoul.head())

sr_one=df_seoul.loc['경기도']
print(sr_one.head())
sr_one.head()

plt.style.use('ggplot')

plt.figure(figsize=(14,5))
plt.xticks(size=10, rotation='vertical')

plt.plot(sr_one.index, sr_one.values, marker='o', markersize=10)

plt.title('서울->경기 인구 이동', size=10)
plt.xlabel('기간', size=20)
plt.ylabel('이동 인구수', size=20)

plt.legend(labels=['서울->경기'], loc='best', fontsize=15)

plt.show()



print('######################################')
print('##  예제 4-6. Matplotlib 스타일 리스트 출력           ')
print('######################################')

import matplotlib.pyplot as plt

print(plt.style.available)


print('######################################')
print('##  예제 4-7. Matplotlib 스타일 리스트 출력           ')
print('######################################')

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

font_path='./part4/malgun.ttf'
font_name=font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)

df=pd.read_excel('./part4/시도별 전출입 인구수.xlsx', na_values=0, header=0)

print(df.head())


df=df.fillna(method='ffill')

print(df.head())

mask=(df['전출지별'] == '서울특별시') & (df['전입지별'] != '서울특별시')
df_seoul=df[mask]
print(df_seoul.head())

df_seoul=df_seoul.drop(['전출지별'], axis=1)
df_seoul.rename({'전입지별':'전입지'}, axis=1, inplace=True)
df_seoul.set_index('전입지', inplace=True)

print(df_seoul.head())

sr_one=df_seoul.loc['경기도']
print(sr_one.head())
sr_one.head()

plt.style.use('ggplot')

plt.figure(figsize=(14,5))
plt.xticks(size=10, rotation='vertical')

plt.plot(sr_one.index, sr_one.values, marker='o', markersize=10)

plt.title('서울->경기 인구 이동', size=10)
plt.xlabel('기간', size=20)
plt.ylabel('이동 인구수', size=20)

plt.legend(labels=['서울->경기'], loc='best', fontsize=15)


print(plt.style.available)

plt.ylim(50000,800000)

plt.annotate('',
             xy=(20, 620000),
             xytext=(2,290000),
             xycoords='data',
             arrowprops=dict(arrowstyle='->', color='skyblue', lw=5),
             )

plt.annotate('',
             xy=(47,450000),
             xytext=(30,580000),
             xycoords='data',
             arrowprops=dict(arrowstyle='->', color='olive', lw=5),
             )

plt.annotate('인구 이동 증가(1970-1995)',
             xy=(40,560000),
             rotation=11,
             va='baseline',
             ha='center',
             fontsize=15,
             )

plt.show()



print('######################################')
print('##  예제 4-8. Matplotlib 소개            ')
print('######################################')

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

font_path='./part4/malgun.ttf'
font_name=font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)

df=pd.read_excel('./part4/시도별 전출입 인구수.xlsx', na_values=0, header=0)

print(df.head())


df=df.fillna(method='ffill')

print(df.head())

mask=(df['전출지별'] == '서울특별시') & (df['전입지별'] != '서울특별시')
df_seoul=df[mask]
print(df_seoul.head())

df_seoul=df_seoul.drop(['전출지별'], axis=1)
df_seoul.rename({'전입지별':'전입지'}, axis=1, inplace=True)
df_seoul.set_index('전입지', inplace=True)

print(df_seoul.head())

sr_one=df_seoul.loc['경기도']
print(sr_one.head())
sr_one.head()

plt.style.use('ggplot')

plt.figure(figsize=(14,5))
plt.xticks(size=10, rotation='vertical')

plt.plot(sr_one.index, sr_one.values, marker='o', markersize=10)

plt.title('서울->경기 인구 이동', size=10)
plt.xlabel('기간', size=20)
plt.ylabel('이동 인구수', size=20)

plt.legend(labels=['서울->경기'], loc='best', fontsize=15)


print(plt.style.available)

fig=plt.figure(figsize=(10,10))

ax1=fig.add_subplot(2,1,1)
ax2=fig.add_subplot(2,1,2)

ax1.plot(sr_one, 'o', markersize=10)
ax2.plot(sr_one, marker='o', markerfacecolor='green', markersize=10,
         color='olive', linewidth=2, label='서울 -> 경기')

ax2.legend(loc='best')

ax1.set_ylim(50000,800000)
ax2.set_ylim(50000,800000)

ax1.set_xticklabels(sr_one.index, rotation=75)
ax2.set_xticklabels(sr_one.index, rotation=75)

plt.show()


print('######################################')
print('##  예제 4-9. axe 객체 그래프 꾸미기         ')
print('######################################')

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

font_path='./part4/malgun.ttf'
font_name=font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)

df=pd.read_excel('./part4/시도별 전출입 인구수.xlsx', na_values=0, header=0)

print(df.head())


df=df.fillna(method='ffill')

print(df.head())

mask=(df['전출지별'] == '서울특별시') & (df['전입지별'] != '서울특별시')
df_seoul=df[mask]
print(df_seoul.head())

df_seoul=df_seoul.drop(['전출지별'], axis=1)
df_seoul.rename({'전입지별':'전입지'}, axis=1, inplace=True)
df_seoul.set_index('전입지', inplace=True)

print(df_seoul.head())

sr_one=df_seoul.loc['경기도']
print(sr_one.head())
sr_one.head()

plt.style.use('ggplot')

plt.figure(figsize=(14,5))
plt.xticks(size=10, rotation='vertical')

plt.plot(sr_one.index, sr_one.values, marker='o', markersize=10)

plt.title('서울->경기 인구 이동', size=10)
plt.xlabel('기간', size=20)
plt.ylabel('이동 인구수', size=20)

plt.legend(labels=['서울->경기'], loc='best', fontsize=15)


print(plt.style.available)

fig=plt.figure(figsize=(20,5))

ax=fig.add_subplot(1,1,1)

ax.plot(sr_one, marker='o', markerfacecolor='orange', markersize=10,
         color='olive', linewidth=2, label='서울 -> 경기')

ax.legend(loc='best')

ax.set_ylim(50000,800000)
ax.set_title('서울 -> 경기 인구 이동', size=20)

ax.set_xlabel('기간', size=12)
ax.set_ylabel('이동 인구수', size=12)

ax.set_xticklabels(sr_one.index, rotation=75)

ax.tick_params(axis='x', labelsize=10)
ax.tick_params(axis='y', labelsize=10)

plt.show()


print('######################################')
print('##  예제 4-10. 같은 화면에 그래프 추가하기   ')
print('######################################')

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

font_path='./part4/malgun.ttf'
font_name=font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)

df=pd.read_excel('./part4/시도별 전출입 인구수.xlsx', na_values=0, header=0)

print(df.head())


df=df.fillna(method='ffill')

print(df.head())

mask=(df['전출지별'] == '서울특별시') & (df['전입지별'] != '서울특별시')
df_seoul=df[mask]
print(df_seoul.head())

df_seoul=df_seoul.drop(['전출지별'], axis=1)
df_seoul.rename({'전입지별':'전입지'}, axis=1, inplace=True)
df_seoul.set_index('전입지', inplace=True)

print(df_seoul.head())

sr_one=df_seoul.loc['경기도']
print(sr_one.head())
sr_one.head()

plt.style.use('ggplot')

plt.figure(figsize=(14,5))
plt.xticks(size=10, rotation='vertical')

plt.plot(sr_one.index, sr_one.values, marker='o', markersize=10)

plt.title('서울->경기 인구 이동', size=10)
plt.xlabel('기간', size=20)
plt.ylabel('이동 인구수', size=20)

plt.legend(labels=['서울->경기'], loc='best', fontsize=15)


print(plt.style.available)

fig=plt.figure(figsize=(20,5))

ax=fig.add_subplot(1,1,1)

ax.plot(sr_one, marker='o', markerfacecolor='orange', markersize=10,
        color='olive', linewidth=2, label='서울 -> 경기')

ax.legend(loc='best')

ax.set_ylim(50000,800000)
ax.set_title('서울 -> 경기 인구 이동', size=20)

ax.set_xlabel('기간', size=12)
ax.set_ylabel('이동 인구수', size=12)

ax.set_xticklabels(sr_one.index, rotation=75)

ax.tick_params(axis='x', labelsize=10)
ax.tick_params(axis='y', labelsize=10)

plt.show()


col_years=list(map(str, range(1970,2018)))
df_3=df_seoul.loc[['충청남도','경상북도','강원도'], col_years]

plt.style.use('ggplot')

fig=plt.figure(figsize=(20,5))
ax=fig.add_subplot(1,1,1)

ax.plot(col_years, df_3.loc['충청남도',:], marker='o', markerfacecolor='green',
        markersize=10, color='olive', linewidth=2, label='서울 -> 충남')

ax.plot(col_years, df_3.loc['경상북도',:], marker='o', markerfacecolor='blue',
        markersize=10, color='skyblue', linewidth=2, label='서울 -> 경북')

ax.plot(col_years, df_3.loc['강원도',:], marker='o', markerfacecolor='red',
        markersize=10, color='magenta', linewidth=2, label='서울 -> 강원')

ax.legend(loc='best')

ax.set_title('서울 -> 충남, 경북, 강원 인구 이동', size=20)

ax.set_xlabel('기간', size=12)
ax.set_ylabel('이동 인구수', size=12)

ax.set_xticklabels(col_years, rotation=90)

ax.tick_params(axis='x', labelsize=10)
ax.tick_params(axis='y', labelsize=10)

plt.show()

print('######################################')
print('##  예제 4-11. 화면 4분할 그래프 그리기    ')
print('######################################')

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

font_path='./part4/malgun.ttf'
font_name=font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)

df=pd.read_excel('./part4/시도별 전출입 인구수.xlsx', na_values=0, header=0)

print(df.head())


df=df.fillna(method='ffill')

print(df.head())

mask=(df['전출지별'] == '서울특별시') & (df['전입지별'] != '서울특별시')
df_seoul=df[mask]
print(df_seoul.head())

df_seoul=df_seoul.drop(['전출지별'], axis=1)
df_seoul.rename({'전입지별':'전입지'}, axis=1, inplace=True)
df_seoul.set_index('전입지', inplace=True)

print(df_seoul.head())

sr_one=df_seoul.loc['경기도']
print(sr_one.head())
sr_one.head()


col_years=list(map(str, range(1970,2018)))
df_4=df_seoul.loc[['충청남도','경상북도','강원도', '전라남도'], col_years]

plt.style.use('ggplot')

fig=plt.figure(figsize=(10,10))

ax1=fig.add_subplot(2,2,1)
ax2=fig.add_subplot(2,2,2)
ax3=fig.add_subplot(2,2,3)
ax4=fig.add_subplot(2,2,4)

ax1.plot(col_years, df_4.loc['충청남도',:], marker='o', markerfacecolor='green',
        markersize=10, color='olive', linewidth=2, label='서울 -> 충남')

ax2.plot(col_years, df_4.loc['경상북도',:], marker='o', markerfacecolor='blue',
        markersize=10, color='skyblue', linewidth=2, label='서울 -> 경북')

ax3.plot(col_years, df_4.loc['강원도',:], marker='o', markerfacecolor='red',
        markersize=10, color='magenta', linewidth=2, label='서울 -> 강원')

ax4.plot(col_years, df_4.loc['전라남도',:], marker='o', markerfacecolor='orange',
         markersize=10, color='yellow', linewidth=2, label='서울 -> 전남')

ax1.legend(loc='best')
ax2.legend(loc='best')
ax3.legend(loc='best')
ax4.legend(loc='best')

ax1.set_title('서울 -> 충남 인구 이동', size=15)
ax2.set_title('서울 -> 경북 인구 이동', size=15)
ax3.set_title('서울 -> 강원 인구 이동', size=15)
ax4.set_title('서울 -> 전남 인구 이동', size=15)

ax1.set_xticklabels(col_years, rotation=90)
ax2.set_xticklabels(col_years, rotation=90)
ax3.set_xticklabels(col_years, rotation=90)
ax4.set_xticklabels(col_years, rotation=90)

plt.show()

print('######################################')
print('##  예제 4-12. matplotlib 스타일 리스트 출력  ')
print('######################################')

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

colors = {}

for name, hex in matplotlib.colors.cnames.items():
    colors[name]=hex

print(colors)

