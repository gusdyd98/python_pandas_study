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
