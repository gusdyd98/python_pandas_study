import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_excel('./part4/시도별 전출입 인구수.xlsx', na_values=0, header=0)

print(df.head())
