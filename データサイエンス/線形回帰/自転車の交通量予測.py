# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 10:28:12 2019

@author: ShimaLab
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from pandas.tseries.holiday import USFederalHolidayCalendar
from sklearn.utils import resample

counts = pd.read_csv('Fremont.csv',index_col='Date',parse_dates=True)
weather = pd.read_csv('BicycleWeather.csv',index_col='DATE',parse_dates=True)

daily=counts.resample('d',how='sum')
daily['Total']=daily.sum(axis=1)
daily=daily[['Total']]

days=['Mon','Tue','Wed','Thu','Fri','Sat','Sun']

for i in range(7):
    daily[days[i]] = (daily.index.dayofweek == i).astype(float)
    
cal = USFederalHolidayCalendar()
holidays = cal.holidays('2012','2016')
daily = daily.join(pd.Series(1,index = holidays,name = 'holiday'))
daily['holiday'].fillna(0,inplace = True)

def hours_of_daylight(date,axis=23.44,latitude=47.61):
    days=(date-pd.datetime(2000,12,21)).days
    m=(1.-np.tan(np.radians(latitude))*np.tan(np.radians(axis)*np.cos(days*2*np.pi/365.25)))
    return 24.*np.degrees(np.arccos(1-np.clip(m,0,2)))/180.

daily['daylight_hrs']=list(map(hours_of_daylight,daily.index))
daily[['daylight_hrs']].plot()
plt.ylim(8,17)

weather['TMIN']/=10
weather['TMAX']/=10
weather['Temp (C)']=0.5*(weather['TMIN']+weather['TMAX'])
weather['PRCP']/=254
weather['dry day'] = (weather['PRCP']==0).astype(int)


daily = daily.join(weather[['PRCP','Temp (C)','dry day']])

daily['annual']=(daily.index-daily.index[0]).days/365.


daily.dropna(axis=0,how='any',inplace=True)

column_names = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun','holiday','daylight_hrs','PRCP','dry day','Temp (C)','annual']

X = daily[column_names]
y = daily['Total']

model = LinearRegression(fit_intercept=False)
model.fit(X,y)
daily['predicted'] = model.predict(X)

daily[['Total','predicted']].plot(alpha = 0.5)

params = pd.Series(model.coef_,index=X.columns)


np.random.seed(1)
err = np.std([model.fit(*resample(X,y)).coef_ for i in range(100)],0)

print(pd.DataFrame({'effect':params.round(0),'error':err.round(0)}))










