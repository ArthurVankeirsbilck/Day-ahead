from pyomo.environ import *
import clickhouse_connect
import datetime
import json
import pandas as pd
import matplotlib.pyplot as plt

#BELPEX published 15:00 --> dit is uur 0 in ons model

conn = clickhouse_connect.get_client(host='52.28.180.90', port=8123) #moet op reactor zitten want poort zit dicht
query = 'SELECT * FROM day_ahead_prices'

df = conn.query_df(query)
start_day = 1706450400
start_day = datetime.datetime.fromtimestamp(start_day)
start_day = start_day.replace(tzinfo=datetime.timezone.utc)

end_day = 1706536800
end_day = datetime.datetime.fromtimestamp(end_day)
end_day  = end_day.replace(tzinfo=datetime.timezone.utc)
print(start_day)
print(end_day)
df = df[df['time'] >= start_day]
df = df[df['time'] <= end_day]
print(df)
df = df.reset_index()


predicted_charging_energy_peak_period = 5 #kWh
predicted_charging_energy_peak_period_tomorrow = 7 #kWh
predicted_start_of_peak = 3
predicted_stop_of_peak = 7

Pcharge_max = 2.5
Pdischarge_max = 2.5

with open('output_EV.json') as json_file:
    data = json.load(json_file)

print(data)
EV_start_time = datetime.datetime.fromtimestamp(data['EV_start_unix'])
EV_stop_time = datetime.datetime.fromtimestamp(data['EV_stop_unix'])
EV_time = round((EV_stop_time-EV_start_time).seconds/3600)

if datetime.datetime.fromtimestamp(data['EV_start_unix']).hour > 15:
    EV_start = datetime.datetime.fromtimestamp(data['EV_start_unix']).hour - 15
    EV_stop = EV_start + EV_time
else:
    EV_start = (24-15) + datetime.datetime.fromtimestamp(data['EV_start_unix']).hour
    EV_stop = EV_start + EV_time

print(EV_start)
print(EV_stop)

hourly_values_EV = [0] * 24
hourly_values_EV[EV_start:EV_stop] = data['chargepower']

time_list = [*range(0, predicted_start_of_peak, 1)] + [*range(predicted_stop_of_peak+1, 24, 1)]
time_combined = [*range(0, 24, 1)]
print(time_list)
time_list_peak = [*range(predicted_start_of_peak, predicted_stop_of_peak, 1)]
print(time_list_peak)
EV = {}
for t in time_list:
    for time, d in zip(time_list, hourly_values_EV):
        EV[(time)] = d

SPOT = df['price'].values.tolist()
prices = {}
for t in time_list:
    for time, d in zip(time_list, SPOT):
        prices[(time)] = d
print(prices)
E = 10
SOC_min = 0.0
SOC_max = 0.9
model = ConcreteModel()
model.T = Set(initialize=time_list)  # Time periods
model.T_peak = Set(initialize=time_list_peak)
model.T_combined = Set(initialize=time_combined)
model.EV_charge = Param(model.T, initialize=EV)
model.charge = Var(model.T_combined, within=NonNegativeReals, bounds=(0,Pcharge_max))
model.discharge = Var(model.T_peak, within=NonNegativeReals, bounds=(0,Pdischarge_max))
model.Peak_energy = Param(initialize=predicted_charging_energy_peak_period)
model.Peak_energy_tomorrow = Param(initialize=predicted_charging_energy_peak_period_tomorrow)
model.connection_power = Param(initialize=data['grid_connection_P'])
model.price = Param(model.T_combined, initialize=prices)
model.Estart = Param(initialize=predicted_charging_energy_peak_period)
model.S = Var(model.T_combined, within=NonNegativeReals, bounds=((E*SOC_min),(E*SOC_max)))
model.etaS = Param(initialize=1)
model.etain = Param(initialize=1)
model.etaout = Param(initialize=1)
M = 1000
model.x = Var(model.T, within=Binary)
model.y = Var(model.T_peak, within=Binary)

#variables
model.obj = Objective(expr=model.charge[t]*model.price[t]/1000, sense=minimize)

def charging(model, t):
    return sum(model.charge[t] for t in model.T) == model.Peak_energy_tomorrow

model.charging = Constraint(model.T, rule= charging)

def discharging(model, t):
    return sum(model.discharge[t] for t in model.T_peak) == model.Peak_energy

model.discharging = Constraint(model.T_peak, rule=discharging)

def charging_power(model,t):
    if t == 23:
        model.charge[t] = 0
    return model.charge[t] <= model.connection_power - model.EV_charge[t]

model.charging_power = Constraint(model.T, rule= charging_power)

# def storage_balance(model, t, t2, t3):
#     if t == 0:
#         return model.S[t] == model.Estart + model.charge[t2]*model.etain - model.discharge[t3]/model.etaout
#     return model.S[t] == model.etaS*model.S[t-1] + model.charge[t2]*model.etain - model.discharge[t3]/model.etaout

# model.storage_balance = Constraint(model.T_combined, model.T, model.T_peak, rule=storage_balance)   

def storage_balance(model, t):
    if t == 0:
        return model.S[t] == model.Estart + model.charge[t]*model.etain - (model.discharge[t]/model.etaout if t in model.T_peak else 0)
    else:
        return model.S[t] == model.etaS*model.S[t-1] + model.charge[t]*model.etain - (model.discharge[t]/model.etaout if t in model.T_peak else 0)

model.storage_balance = Constraint(model.T_combined, rule=storage_balance)  


# def battery_power(model, t):
#     return (model.charge[t]) * ((model.discharge[t] if t in model.T_peak else 0)) == 0

# model.battery_power = Constraint(model.T, rule=battery_power)

# Constraint to limit charging power when x_t is 1
def charge_limit_rule(model, t):
    return model.charge[t] <= M * model.x[t]
model.charge_limit = Constraint(model.T, rule=charge_limit_rule)

# Constraint to limit discharging power when y_t is 1, for t in T_peak
def discharge_limit_rule(model, t):
    return model.discharge[t] <= M * model.y[t]
model.discharge_limit = Constraint(model.T_peak, rule=discharge_limit_rule)

# Constraint to ensure either charging or discharging, but not both
def xor_rule(model, t):
    if t in model.T_peak:
        return model.x[t] + model.y[t] <= 1
    else:
        return model.x[t] <= 1  # Only charging is possible outside T_peak
model.xor_constraint = Constraint(model.T, rule=xor_rule)

solver = SolverFactory('glpk')

solver.solve(model, tee=True)

for t in model.T_combined:
    print(model.charge[t].value)
    (print(model.discharge[t].value) if t in model.T_peak else 0)
print('Total cost = ', model.obj())

charge = pd.Series([model.charge[t].value for t in model.T_combined], index=model.T_combined)
discharge = pd.Series([model.discharge[t].value if t in model.T_peak else 0 for t in model.T_combined], index=model.T_combined)
SOC = pd.Series([model.S[t].value for t in model.T_combined], index=model.T_combined)

charge.plot()
discharge.plot()
SOC.plot()

plt.xlabel('Time')
# df['price'].plot(secondary_y=True)
plt.ylabel('Price')
plt.show()