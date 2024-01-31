import pandas as pd
import datetime
from pyomo.environ import *
import time
import clickhouse_connect
import matplotlib.pyplot as plt
import json 

Output_dict = {}

predicted_charging_session_hours = 8 #hours
predicted_charging_energy = 40 #kWh
grid_connection_I = 32 #A
grid_connection_U = 400 #V
grid_connection_P = (grid_connection_I*grid_connection_U*1.73)/1000 #kW
print(grid_connection_P)
grid_connection_P = 0.8*grid_connection_P #kW

Output_dict['grid_connection_I'] = grid_connection_I
Output_dict['grid_connection_U'] = grid_connection_U
Output_dict['grid_connection_P'] = grid_connection_P


conn = clickhouse_connect.get_client(host='52.28.180.90', port=8123) #moet op reactor zitten want poort zit dicht
query = 'SELECT * FROM day_ahead_prices'

df = conn.query_df(query)
unix = 1706454900 #	Sun Jan 28 2024 16:15:00 GMT+0100
now = datetime.datetime.fromtimestamp(unix)
now = now.replace(tzinfo=datetime.timezone.utc)

unix2 = 1706508600 #Mon Jan 29 2024 07:10:00 GMT+0100
end_charging_session = unix2
end_charging_session = datetime.datetime.fromtimestamp(end_charging_session)
end_charging_session = end_charging_session.replace(tzinfo=datetime.timezone.utc)

Output_dict['EV_start_unix'] = 1706454900
Output_dict['EV_stop_unix'] = 1706508600

df = df[df['time'] >= now]
df = df[df['time'] <= end_charging_session]
print(df.head())

df = df.reset_index()

charging_current_list = [0,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]

model = ConcreteModel()
model.time = Set(initialize=df.index)
model.price = Param(model.time, initialize=df['price'].to_dict())
model.voltage = Param(initialize=grid_connection_U)
model.chargepower = Var(model.time, domain=NonNegativeReals)
model.chargecurrent = Var(model.time, within=charging_current_list)

def chargepower_rule(model, t):
    return model.chargepower[t] == model.chargecurrent[t]*model.voltage*1.73/1000
model.chargepower_rule = Constraint(model.time, rule=chargepower_rule)

def energy_rule(model):
    return sum(model.chargepower[t] for t in model.time) >= predicted_charging_energy
model.energy = Constraint(rule=energy_rule)

def obj_rule(model):
    return sum(model.chargepower[t]*model.price[t]/1000 for t in model.time)
model.obj = Objective(rule=obj_rule, sense=minimize)

def max_power_rule(model, t):
    return model.chargepower[t] <= grid_connection_P
model.max_power = Constraint(model.time, rule=max_power_rule)



solver = SolverFactory('glpk')

solver.solve(model)

print('Total cost = ', model.obj())
print('Total energy = ', sum(model.chargepower[t]() for t in model.time))
chargepower = pd.Series([model.chargepower[t]() for t in model.time], index=model.time)
chargecurrent = pd.Series([model.chargecurrent[t]() for t in model.time], index=model.time)
print(chargepower)
Output_dict['chargepower'] = chargepower.to_list()
# print(chargepower)
chargepower.plot()
#add labels
plt.xlabel('Time')
plt.ylabel('Current (A)')
df['price'].plot(secondary_y=True)
plt.ylabel('Price')
plt.show()

#save model
model.write('model.lp', io_options={'symbolic_solver_labels': True})

with open("output_EV.json", "w") as outfile: 
    json.dump(Output_dict, outfile)

