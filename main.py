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