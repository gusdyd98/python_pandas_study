# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

#######################################################################################
###    Example.1-1
#######################################################################################
import pandas as pd

dict_data = {'a':1, 'b':2, 'c':3}

sr=pd.Series(dict_data)

sr

print(type(sr))
print('\n')

print(sr)

#######################################################################################
#######################################################################################
###    Example.1-2
#######################################################################################
import pandas as pd

list_data=['2019-01-02', 3.14, 'ABC', 100, True]
sr=pd.Series(list_data)
print(sr)

idx=sr.index
val=sr.values
print(idx)
print('\n')
print(val)

#######################################################################################
#######################################################################################
###    Example.1-3
#######################################################################################
import pandas as pd

tup_data=('영인', '2010-05-01', '여', True)
sr=pd.Series(tup_data, index=['이름', '생년월일', '성별', '학생여부'])
print(sr)

print('\n')
print(sr[0])
print(sr['이름'])


print('\n')
print(sr[[1,2]])
print('\n')
print(sr[['생년월일','성별']])

print('\n')
print(sr[1:2])
print('\n')
print(sr['생년월일':'성별'])
#######################################################################################
#######################################################################################
###    Example.1-4
#######################################################################################
import pandas as pd

dict_data={'c0':[1,2,3], 'c1':[4,5,6], 'c2':[7,8,9], 'c3':[10,11,12], 'c4':[13,14,15]}

df=pd.DataFrame(dict_data)

print('\n')
print(type(df))
print('\n')
print(df)


#######################################################################################
#######################################################################################
###    Example.1-5
#######################################################################################
import pandas as pd

df=pd.DataFrame([[15,'남','덕영중'],[17,'여','수리중']],
                index=['준서', '예은'],
                columns=['나이','성별','학교'])

print('\n')
print(df)
print('\n')
print(df.index)
print('\n')
print(df.columns)


df.index=['학생1','학생2']
df.columns=['연령','남녀','소속']

print('\n')
print(df)
print('\n')
print(df.index)
print('\n')
print(df.columns)

#######################################################################################
#######################################################################################
###    Example.1-6
#######################################################################################
import pandas as pd

df=pd.DataFrame([[15,'남','덕영중'],[17,'여','수리중']],
                index=['준서', '예은'],
                columns=['나이','성별','학교'])

print('\n')
print(df)

df.rename(columns={'나이':'연령', '성별':'남녀', '학교':'소속'}, inplace=True)
df.rename(index={'준서':'학생1', '예은':'학생2'}, inplace=True)

print('\n')
print(df)

#######################################################################################
#######################################################################################
###    Example.1-7
#######################################################################################
import pandas as pd

exam_data={'수학':[90,80,70], '영어':[98,89,95],
           '음악':[85,95,100], '체육':[100,90,90]}

df=pd.DataFrame(exam_data,
                index=['서준', '우현','인아'])

print('\n')
print(df)

df2=df[:]
df2.drop('우현', inplace=True)

print('\n')
print(df2)

df3=df[:]
df3.drop(['우현','인아'], axis=0, inplace=True)

print('\n')
print(df3)

#######################################################################################
#######################################################################################
###    Example.1-8.열 삭제
#######################################################################################
import pandas as pd

exam_data={'수학':[90,80,70], '영어':[98,89,95],
           '음악':[85,95,100], '체육':[100,90,90]}

df=pd.DataFrame(exam_data,
                index=['서준', '우현','인아'])

print('\n')
print(df)

df4=df[:]
df4.drop('수학', axis=1, inplace=True)

print('\n')
print(df4)

df5=df[:]
df5.drop(['영어','음악'], axis=1, inplace=True)

print('\n')
print(df5)

#######################################################################################
#######################################################################################
###    Example.1-9.행 선택  loc, iloc
### 범위 지정 :
###  loc['a':'c'] -> 'a','b','c'
### iloc[  3:7  ] -> 3,4,5,6  (*7 제외)
#######################################################################################
import pandas as pd

exam_data={'수학':[90,80,70], '영어':[98,89,95],
           '음악':[85,95,100], '체육':[100,90,90]}

df=pd.DataFrame(exam_data,
                index=['서준', '우현','인아'])

print('\n')
print(df)

label1=df.loc['서준']
position1 = df.iloc[0]
print('\n')
print(label1)

print('\n')
print(position1)

label3=df.loc['서준':'우현']
position3 = df.iloc[0:1]

print('\n')
print(label3)

print('\n')
print(position3)

#######################################################################################
#######################################################################################
###    Example.1-10.열 선택
###    1. 대괄호([ ])안에 열 이름을 넣음.
###    2. df.영어 와 같이 도트(.) 다음에 열 이름을 입력.
#######################################################################################
import pandas as pd

exam_data={'이름':['서준','우현','인아'],
           '수학':[90,80,70],
           '영어':[98,89,95],
           '음악':[85,95,100],
           '체육':[100,90,90]}

df=pd.DataFrame(exam_data)

print('\n')
print(df)
print(type(df))

math1=df['수학']
print('\n')
print(math1)
print(type(math1))

print('\n')
english=df.영어
print(english)
print(type(english))

music_gym=df[['음악','체육']]
print('\n')
print(music_gym)
print(type(music_gym))

math2=df[['수학']]
print(math2)
print(type(math2))

df.iloc[::2]
df.iloc[0:3:2]


#######################################################################################
#######################################################################################
###    Example.1-11.원소 선택
### 범위 지정 :
###  loc['서준','음악']
### iloc[  0,2  ]
#######################################################################################
import pandas as pd

exam_data={'이름':['서준','우현','인아'],
           '수학':[90,80,70],
           '영어':[98,89,95],
           '음악':[85,95,100],
           '체육':[100,90,90]}

df=pd.DataFrame(exam_data)

df.set_index('이름', inplace=True)

print('\n')
print(df)

a=df.loc['서준', '음악']

print('\n')
print(a)

b=df.iloc[0,2]
print('\n')
print(b)

c=df.loc['서준', ['음악','체육']]
print('\n')
print(c)

d=df.iloc[0, [2,3]]
print('\n')
print(d)

e=df.loc['서준','음악':'체육']
print('\n')
print(e)

f=df.iloc[0,2:]
print('\n')
print(f)

g=df.loc[['서준','우현'], ['음악','체육']]
print('\n')
print(g)

h=df.iloc[[0,1], [2,3]]
print('\n')
print(h)

i=df.loc['서준':'우현', '음악':'체육']
print('\n')
print(i)

j=df.iloc[0:2, 2:]
print('\n')
print(j)



#######################################################################################
#######################################################################################
###    Example.1-12.열 추가
#######################################################################################
import pandas as pd

exam_data={'이름':['서준','우현','인아'],
           '수학':[90,80,70],
           '영어':[98,89,95],
           '음악':[85,95,100],
           '체육':[100,90,90]}

df=pd.DataFrame(exam_data)

print('\n')
print(df)

df['국어']=80
print('\n')
print(df)


#######################################################################################
#######################################################################################
###    Example.1-13.행 추가
#######################################################################################
import pandas as pd

exam_data={'이름':['서준','우현','인아'],
           '수학':[90,80,70],
           '영어':[98,89,95],
           '음악':[85,95,100],
           '체육':[100,90,90]}

df=pd.DataFrame(exam_data)

print('\n')
print(df)

df.loc[3]=0
print('\n')
print(df)

df.loc[4]=['동규', 90, 80, 70, 60]
print('\n')
print(df)

df.loc['행5']=df.loc[3]
print('\n')
print(df)



#######################################################################################
#######################################################################################
###    Example.1-14.원소 값 변경
#######################################################################################
import pandas as pd

exam_data={'이름':['서준','우현','인아'],
           '수학':[90,80,70],
           '영어':[98,89,95],
           '음악':[85,95,100],
           '체육':[100,90,90]}

df=pd.DataFrame(exam_data)

df.set_index('이름', inplace=True)

print('\n')
print(df)

df.iloc[0][3]=80
print('\n')
print(df)

df.loc['서준']['체육']= 90
print('\n')
print(df)

df.loc['서준','체육']=100
print('\n')
print(df)

df.loc['서준',['음악','체육']]= 50
print('\n')
print(df)

df.loc['서준',['음악','체육']]= 100,50
print('\n')
print(df)



#######################################################################################
#######################################################################################
print("\nExample.1-15.행, 열 바꾸기")
#######################################################################################
import pandas as pd

exam_data={'이름':['서준','우현','인아'],
           '수학':[90,80,70],
           '영어':[98,89,95],
           '음악':[85,95,100],
           '체육':[100,90,90]}

df=pd.DataFrame(exam_data)

print('\n')
print(df)

df=df.transpose()
print('\n')
print(df)

df=df.T
print('\n')
print(df)


#######################################################################################
#######################################################################################
print("\n#########################################")
print("\nExample.1-16.특정열을 행 인덱스로 설정하기")
print("\n#########################################\n")
import pandas as pd

exam_data={'이름':['서준','우현','인아'],
           '수학':[90,80,70],
           '영어':[98,89,95],
           '음악':[85,95,100],
           '체육':[100,90,90]}

df=pd.DataFrame(exam_data)

print(df)
print('\n')

ndf=df.set_index(['이름'])
print(ndf)
print('\n')

ndf2=ndf.set_index('음악')
print(ndf2)
print('\n')

ndf3=ndf.set_index(['수학','음악'])
print(ndf3)
print('\n')


#######################################################################################
#######################################################################################
print("\n#########################################")
print("\nExample.1-17.새로운 배열로 행 인덱스 재지정하기")
print("\n#########################################\n")
import pandas as pd

dict_data={'c0':[1,2,3], 'c1':[4,5,6], 'c2':[7,8,9], 'c3':[10,11,12], 'c4':[13,14,15]}

df=pd.DataFrame(dict_data, index=['r0','r1','r2'])

print(df)
print('\n')

new_index=['r0','r1','r2','r3','r4']
ndf=df.reindex(new_index)
print(ndf)
print('\n')

new_index=['r0','r1','r2','r3','r4']
ndf2=df.reindex(new_index, fill_value=0)
print(ndf2)
print('\n')



#######################################################################################
#######################################################################################
print("\n#########################################")
print("\nExample.1-18.정수형 위치 인덱스로 초기화하기")
print("\n#########################################\n")
import pandas as pd

dict_data={'c0':[1,2,3], 'c1':[4,5,6], 'c2':[7,8,9], 'c3':[10,11,12], 'c4':[13,14,15]}

df=pd.DataFrame(dict_data, index=['r0','r1','r2'])

print(df)
print('\n')

ndf=df.reset_index()
print(ndf)
print('\n')



#######################################################################################
#######################################################################################
print("\n#########################################")
print("\nExample.1-19.데이터프레임을 행 인덱스 기준으로 정렬하기")
print("\n#########################################\n")
import pandas as pd

dict_data={'c0':[1,2,3], 'c1':[4,5,6], 'c2':[7,8,9], 'c3':[10,11,12], 'c4':[13,14,15]}

df=pd.DataFrame(dict_data, index=['r0','r1','r2'])

print(df)
print('\n')

ndf=df.sort_index(ascending=False)
print(ndf)
print('\n')



#######################################################################################
#######################################################################################
print("\n#########################################")
print("\nExample.1-20.열기준 정렬하기")
print("\n#########################################\n")
import pandas as pd

dict_data={'c0':[1,2,3], 'c1':[4,5,6], 'c2':[7,8,9], 'c3':[10,11,12], 'c4':[13,14,15]}

df=pd.DataFrame(dict_data, index=['r0','r1','r2'])

print(df)
print('\n')

ndf=df.sort_values(by='c1', ascending=False)
print(ndf)
print('\n')


#######################################################################################
#######################################################################################
print("\n#########################################")
print("\nExample.1-21.시리즈를 숫자로 나누기")
print("\n#########################################\n")
import pandas as pd

student1=pd.Series({'국어':100, '영어':80, '수학':90})

print(student1)
print('\n')

percentage=student1/200

print(percentage)
print('\n')
print(type(percentage))




#######################################################################################
#######################################################################################
print("\n#########################################")
print("\nExample.1-22.시리즈 사칙연산")
print("\n#########################################\n")
import pandas as pd

student1=pd.Series({'국어':100, '영어':80, '수학':90})
student2=pd.Series({'수학':80, '국어':90, '영어':80})

print(student1)
print('\n')
print(student2)
print('\n')

addition=student1 + student2
subtraction=student1 - student2
multiplication=student1 * student2
division=student1 / student2

print(type(division))
print('\n')

result=pd.DataFrame([addition, subtraction, multiplication, division],
                    index=['덧셈', '뺄셈', '곱셈', '나눗셈'])

print(result)
print('\n')


#######################################################################################
print("\n#########################################")
print("\nExample.1-23.NaN 값이 있는 시리즈 연산")
print("\n#########################################\n")
#######################################################################################
import pandas as pd
import numpy as np

student1=pd.Series({'국어':np.nan, '영어':80, '수학':90})
student2=pd.Series({'수학':80, '국어':90})

print(student1)
print('\n')
print(student2)
print('\n')

addition=student1 + student2
subtraction=student1 - student2
multiplication=student1 * student2
division=student1 / student2

print(type(division))
print('\n')

result=pd.DataFrame([addition, subtraction, multiplication, division],
                    index=['덧셈', '뺄셈', '곱셈', '나눗셈'])

print(result)
print('\n')


#######################################################################################
print("\n############################################")
print("\nExample.1-24.연산 메소드 사용하여 시리즈 연산하기")
print("\n############################################\n")
#######################################################################################
import pandas as pd
import numpy as np

student1=pd.Series({'국어':np.nan, '영어':80, '수학':90})
student2=pd.Series({'수학':80, '국어':90})

print(student1)
print('\n')
print(student2)
print('\n')

sr_add = student1.add(student2, fill_value=0)
sr_sub = student1.sub(student2, fill_value=0)
sr_mul = student1.mul(student2, fill_value=0)
sr_div = student1.div(student2, fill_value=0)

result=pd.DataFrame([sr_add, sr_sub, sr_mul, sr_div],
                    index=['덧셈', '뺄셈', '곱셈', '나눗셈'])

print(result)
print('\n')


#######################################################################################
print("############################################")
print("Example.1-25.데이터프레임에 숫자 더하기")
print("############################################\n")
#######################################################################################
import pandas as pd
import seaborn as sns

titanic=sns.load_dataset('titanic')
df=titanic.loc[:, ['age','fare']]

print(df.head())
print('\n')
print(type(df))
print('\n')

addition=df+10
print(addition.head())
print('\n')
print(type(addition))


#######################################################################################
print("############################################")
print("Example.1-26.데이터프레임끼리 더하기")
print("############################################\n")
#######################################################################################
import pandas as pd
import seaborn as sns

titanic=sns.load_dataset('titanic')
df=titanic.loc[:, ['age','fare']]

print(df.tail())
print('\n')
print(type(df))
print('\n')

addition=df+10
print(addition.tail())
print('\n')
print(type(addition))
print('\n')

subtraction=addition - df
print(subtraction.tail())
print('\n')
print(type(subtraction))











