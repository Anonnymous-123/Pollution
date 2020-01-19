# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 21:21:21 2020

@author: jaspr
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as pyplot

my_data=pd.read_csv("data.csv");


my_data=pd.read_csv("data.csv");
my_data['date']=pd.to_datetime(my_data['date'])

my_data.head();

my_data.shape

my_data.head

import seaborn as sns
from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()
sns.set(style='whitegrid', context='notebook')
features_plot = ['so2', 'no2']

data_to_plot = my_data[features_plot]
data_to_plot = scalar.fit_transform(data_to_plot)
data_to_plot = pd.DataFrame(data_to_plot)

sns.pairplot(data_to_plot, size=2.0);
pyplot.tight_layout()
pyplot.show()

my_data.dropna(axis=0,how='all')

features=my_data

features = features.drop('so2', axis=1)
features = features.drop('no2', axis=1)

labels = my_data['so2'].values

features = features.values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3)

print("X_trian shape --> {}".format(X_train.shape))
print("y_train shape --> {}".format(y_train.shape))
print("X_test shape --> {}".format(X_test.shape))
print("y_test shape --> {}".format(y_test.shape))

from sklearn.ensemble import ExtraTreesRegressor

etr = ExtraTreesRegressor(n_estimators=300)
etr.fit(X_train, y_train)
