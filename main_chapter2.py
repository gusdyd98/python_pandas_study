####################################
print("####################################################")
print("###    2-1.  CSV 파일 읽기                            ")
print("####################################################")
####################################
import pandas as pd

file_path= './part2/read_csv_sample.csv'

df1=pd.read_csv(file_path)

print(df1)
print('\n')


df2=pd.read_csv(file_path, header=None)
print(df2)
print('\n')

df3=pd.read_csv(file_path, index_col=None)
print(df3)
print('\n')

df4=pd.read_csv(file_path, index_col='c0')
print(df4)
print('\n')


####################################
print("####################################################")
print("###    2-2.  Excel 파일 읽기                         ")
print("####################################################")
####################################
import pandas as pd


file_path= './part2/남북한발전전력량.xlsx'

df1=pd.read_excel('part2/남북한발전전력량.xlsx')
df1=pd.read_excel('./part2/남북한발전전력량.xlsx')
df2=pd.read_excel('./part2/남북한발전전력량.xlsx', header=None)

print(df1)
print('\n')
print(df2)
print('\n')

