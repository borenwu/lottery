import pandas as pd
import numpy as np
from fbprophet import Prophet
import matplotlib.pyplot as plt


df = pd.read_csv('./example_retail_sales.csv')
m = Prophet().fit(df)
future = m.make_future_dataframe(periods=120, freq='M')
fcst = m.predict(future)
print(fcst.head())
m.plot(fcst)
plt.show()