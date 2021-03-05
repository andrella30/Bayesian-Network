import pandas as pd
import numpy as np
import pgmpy as pgm

from pgmpy.estimators import MaximumLikelihoodEstimator 
from pgmpy.models import BayesianModel
from pgmpy.inference import VariableElimination

data = pd.read_csv('train.csv')
data =  data.replace('N/A', np.nan)

#Bateria
data.loc[data["battery_power"] <= 875, "battery_power"] = 0
data.loc[( data["battery_power"] > 875) & (data["battery_power"] <= 1249), "battery_power"] = 1
data.loc[( data["battery_power"] > 1249) & (data["battery_power"] <= 1623), "battery_power"] = 2
data.loc[( data["battery_power"] > 1623) & (data["battery_power"] <= 1998), "battery_power"] = 3

#Ram
data.loc[data["ram"] <= 1191, "ram"] = 0
data.loc[( data["ram"] > 1191) & (data["ram"] <= 2126), "ram" ] = 1
data.loc[( data["ram"] > 2126) & (data["ram"] <= 3061), "ram"] = 2
data.loc[data["ram"] > 3061, "ram"] = 3

#Price_range
data.loc[data["price_range"] <= 1, "price_range"] = 0
data.loc[( data["price_range"] > 1) & (data["price_range"] <= 2), "price_range" ] = 1
data.loc[data["price_range"] > 2,  "price_range"] = 2

#Clock Speed
data.loc[data["clock_speed"] <= 1.33, "clock_speed"] = 0
data.loc[( data["clock_speed"] > 1.33) & (data["clock_speed"] <= 2.16), "clock_speed" ] = 1
data.loc[data["clock_speed"] > 2.16, "clock_speed"] = 2

model = BayesianModel([('n_cores', 'ram'), 
                    ('ram', 'clock_speed'), 
                    ('clock_speed', 'price_range'),
                    ('four_g', 'battery_power'), 
                    ('int_memory', 'ram'), 
                    ('wifi', 'battery_power'), 
                    ('battery_power', 'price_range'),
                    ('fc','pc'), ('pc','price_range')])

model.fit(data, estimator=MaximumLikelihoodEstimator)

train_infer = VariableElimination(model)

q1 = train_infer.query(variables=['price_range'], 
                       evidence={ 'ram': 0, 'battery_power': 0, 'pc': 3, 'fc': 1, 'int_memory': 2, 
                       'n_cores': 2, 'four_g': 0,'wifi': 0,'clock_speed': 0 })
print(q1)


q2 = train_infer.query(variables=['price_range'], 
                       evidence={ 'ram': 1, 'four_g': 0, 'wifi': 0,'four_g': 1, 'pc': 15,
                       'fc': 10,'n_cores': 4,'clock_speed': 1, 'battery_power': 2})
print(q2)

q3 = train_infer.query(variables=['price_range'], 
                       evidence={ 'fc': 6, 'pc': 19, 'four_g': 1, 'wifi': 1, 'battery_power': 3,
                                  'n_cores': 2, 'int_memory': 33, 'ram': 3,  'clock_speed': 2, })
print(q3)