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


####################################
print("####################################################")
print("###    2-3.  JSON 파일 읽기                         ")
print("####################################################")
####################################
import pandas as pd

df=pd.read_json('part2/read_json_sample.json')

print(df)
print('\n')



####################################
print("####################################################")
print("###    2-4.  웹에서 표 정보 읽기                      ")
print("####################################################")
####################################
import pandas as pd

url='./part2/sample.html'

tables=pd.read_html(url)

print(len(tables))
print('\n')

for i in range(len(tables)):
    print("tables[%s]" % i)
    print(tables[i])
    print('\n')

df=tables[1]

df.set_index(['name'], inplace=True)
print(df)



####################################
print("####################################################")
print("###    2-5.  웹스크래핑, 미국 ETF 리스트 가져오기      ")
print("####################################################")
####################################
import pandas as pd
from bs4 import BeautifulSoup
import requests
import re

url='https://en.wikipedia.org/wiki/List_of_American_exchange-traded_funds'
resp=requests.get(url)
soup=BeautifulSoup(resp.text, 'lxml')
rows=soup.select('div > ul > li')

etfs={}
for row in rows:
    try:
        etf_name=re.findall('^(.*) \(NYSE', row.text)
        etf_market=re.findall('\((.*)\|',row.text)
        etf_ticker=re.findall('NYSE Arca\|(.*)\)', row.text)

        print(row.text)
        print(etf_name)
        print(etf_market)
        print(etf_ticker)

        if( (len(etf_name) > 0) & (len(etf_market) > 0) & (len(etf_ticker) > 0) ):
            print([etf_market[0], etf_name[0]])
            etfs[etf_ticker[0]]= [etf_market[0], etf_name[0]]
            print(etfs[etf_ticker[0]])


    except AttributeError as err:
        pass
print(etfs)
print('\n')

df=pd.DataFrame(etfs)
print(df)


####################################
print("####################################################")
print("###    2-6.  구글 지오코딩 위치 정보 가져오기          ")
print("####################################################")
####################################
import pandas as pd
import googlemaps

maps=googlemaps.Client(key='AIzaSyBUFrKDpobNTCSTxrj0hP0RjAM5U6d6HVw')
url='https://en.wikipedia.org/wiki/List_of_American_exchange-traded_funds'

lat=[]
lng=[]

places=["서울시청", "국립국악원", "해운대해수욕장"]

i=0
for place in places:
    i=i+1
    try:
        print(i, place)

        geo_location=maps.geocode(place)[0].get('geometry')
        lat.append(geo_location['location']['lat'])
        lng.append(geo_location['location']['lng'])

    except:
        lat.append('')
        lng.append('')
        print(i)

df=pd.DataFrame({'위도':lat, '경도':lng}, index=places)
print(df)
print('\n')

####################################
print("####################################################")
print("###    2-7.  csv 파일로 저장하기                      ")
print("####################################################")
####################################
import pandas as pd

data={'name':['Jerry', 'Riah', 'Paul'],
      'algol':['A','A+','B'],
      'basic':['C','B','B+'],
      'c++':['B+','C','C+'],
      }

df=pd.DataFrame(data)
df.set_index('name', inplace=True)
print(df)

df.to_csv('./part2/df_sample_lhy0.csv')
print('\n')


####################################
print("####################################################")
print("###    2-8.  JSON 파일로 저장하기                      ")
print("####################################################")
####################################
import pandas as pd

data={'name':['Jerry', 'Riah', 'Paul'],
      'algol':['A','A+','B'],
      'basic':['C','B','B+'],
      'c++':['B+','C','C+'],
      }

df=pd.DataFrame(data)
df.set_index('name', inplace=True)
print(df)

df.to_json('./part2/df_sample_lhy0.json')
print('\n')


####################################
print("####################################################")
print("###    2-9.  EXCEL 파일로 저장하기                      ")
print("####################################################")
####################################
import pandas as pd

data={'name':['Jerry', 'Riah', 'Paul'],
      'algol':['A','A+','B'],
      'basic':['C','B','B+'],
      'c++':['B+','C','C+'],
      }

df=pd.DataFrame(data)
df.set_index('name', inplace=True)
print(df)

df.to_excel('./part2/df_sample_lhy0.xlsx')
print('\n')


####################################
print("####################################################")
print("###    2-10.  ExcelWriter 활용하여 파일로 저장하기                      ")
print("####################################################")
####################################
import pandas as pd

data1={'name':['Jerry', 'Riah', 'Paul'],
      'algol':['A','A+','B'],
      'basic':['C','B','B+'],
      'c++':['B+','C','C+'],
      }

data2={'c0':[1,2,3],
       'c1':[4,5,6],
       'c2':[7,8,9],
       'c3':[10,11,12],
       'c4':[13,14,15]}

df1=pd.DataFrame(data1)
df1.set_index('name', inplace=True)
print(df1)
print('\n')
df2=pd.DataFrame(data2)
df2.set_index('c0', inplace=True)
print(df2)

writer=pd.ExcelWriter('./part2/df_excelwriter_lhy0.xlsx')
df1.to_excel(writer, sheet_name="sheet1")
df2.to_excel(writer, sheet_name="sheet2")
writer.save()

