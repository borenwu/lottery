import pandas as pd
import numpy as np
from fbprophet import Prophet

df = pd.read_csv('./ssq.csv')

# print(df.loc[:,['date','y1']])
y1 = df.loc[:, ['date', 'y1']]
y1.rename(columns={'date':'ds','y1':'y'},inplace = True)

y2 = df.loc[:, ['date', 'y2']]
y2.rename(columns={'date':'ds','y2':'y'},inplace = True)

y3 = df.loc[:, ['date', 'y3']]
y3.rename(columns={'date':'ds','y3':'y'},inplace = True)

y4 = df.loc[:, ['date', 'y4']]
y4.rename(columns={'date':'ds','y4':'y'},inplace = True)

y5 = df.loc[:, ['date', 'y5']]
y5.rename(columns={'date':'ds','y5':'y'},inplace = True)

y6 = df.loc[:, ['date', 'y6']]
y6.rename(columns={'date':'ds','y6':'y'},inplace = True)

y7 = df.loc[:, ['date', 'y7']]
y7.rename(columns={'date':'ds','y7':'y'},inplace = True)



m = Prophet()
m.fit(y1)
future1 = m.make_future_dataframe(periods=1)
forecast1 = m.predict(future1)
print(forecast1[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
print('===========end of y1===========')

m = Prophet()
m.fit(y2)
future2 = m.make_future_dataframe(periods=1)
forecast2 = m.predict(future2)
print(forecast2[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
print('===========end of y2===========')

m = Prophet()
m = Prophet()
m.fit(y3)
future3 = m.make_future_dataframe(periods=1)
forecast3 = m.predict(future3)
print(forecast3[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
print('===========end of y3===========')

m = Prophet()
m.fit(y4)
future4 = m.make_future_dataframe(periods=1)
forecast4 = m.predict(future4)
print(forecast4[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
print('===========end of y4===========')

m = Prophet()
m.fit(y5)
future5 = m.make_future_dataframe(periods=1)
forecast5 = m.predict(future5)
print(forecast5[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
print('===========end of y5===========')

m = Prophet()
m.fit(y6)
future6 = m.make_future_dataframe(periods=1)
forecast6 = m.predict(future6)
print(forecast6[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
print('===========end of y6===========')

m = Prophet()
m.fit(y7)
future7 = m.make_future_dataframe(periods=1)
forecast7 = m.predict(future2)
print(forecast7[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
print('===========end of y7===========')