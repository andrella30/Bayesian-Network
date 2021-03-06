import pandas as pd
import numpy as np
import pgmpy as pgm

from pgmpy.estimators import MaximumLikelihoodEstimator 
from pgmpy.models import BayesianModel
from pgmpy.inference import VariableElimination

data = pd.read_csv('dataset/mobile.csv')
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

mobile_infer = VariableElimination(model)

# # Testes
# q4 = mobile_infer.query(variables=['price_range'], 
#                        evidence={ 'ram': 0, 
#                                 'battery_power': 0, 
#                                 'pc': 3, 
#                                 'fc': 1, 
#                                 'int_memory': 2, 
#                                 'n_cores': 2, 
#                                 'four_g': 0,
#                                 'wifi': 0,
#                                 'clock_speed': 0 })
# print(q4)

# q5 = mobile_infer.query(variables=['price_range'], 
#                        evidence={ 'ram': 1, 
#                                 'four_g': 0, 
#                                 'wifi': 0,
#                                 'four_g': 1, 
#                                 'pc': 15,
#                                 'fc': 10,
#                                 'n_cores': 4,
#                                 'clock_speed': 1, 
#                                 'battery_power': 2})
# print(q5)


# q6 = mobile_infer.query(variables=['price_range'], 
#                        evidence={ 'fc': 6, 
#                                 'pc': 19, 
#                                 'four_g': 1, 
#                                 'wifi': 1, 
#                                 'battery_power': 3,
#                                 'n_cores': 2, 
#                                 'int_memory': 33, 
#                                 'ram': 3,  
#                                 'clock_speed': 2})
# print(q6)

def menu():

  opcao_wifi = int(input("Informe se o celular terá wifi: 0 - Não | 1 - Sim: "))
  opcao_4g = int(input("Informe se o celular terá 4G:   0 - Não | 1 - Sim: "))
  opcao_bateria = int(input("Informe a bateria em mAh :   0 - (< 875) | 1 - (875 a 1249) | 2 - (1249 a 1623) | 3 - (> 1623):  "))
  opcao_fc = int(input("Informe os megapixels da camera Frontal  (0 a 19):  "))
  opcao_pc = int(input("Informe os megapixels da camera Traseira (0 a 20): "))
  opcao_numcores = int(input("Informe o número de núcleos (1 a 8): "))
  opcao_intmemory = int(input("Informe a memória interna em Gb (2 a 64): "))
  opcao_ram = int(input("Informe a memória ram em Gb 0 - (< 1Gb) | 1 - (1Gb a 2Gb) | 2 - (2Gb a 3Gb) | 3 - (> 4Gb)  : "))
  opcao_clockspeed = int(input("Informe a velocidade do clock Ghz:  0 - (< 1.33) | 1 - (1.33 á 2.16) | 2 - (> 2.16) : "))
  
  q1 = mobile_infer.query(variables=['price_range'], 
                        evidence={ 'ram': opcao_ram, 'battery_power': opcao_bateria, 'pc': opcao_pc, 'fc': opcao_fc, 'int_memory': opcao_intmemory, 
                        'n_cores': opcao_numcores, 'four_g': opcao_4g,'wifi': opcao_wifi,'clock_speed': opcao_clockspeed })
  
 
  print(q1)
  
  continua = int(input("Realizar outra predição?: 0 - Não | 1 - Sim: "))
  if(continua):
    menu()

menu()

