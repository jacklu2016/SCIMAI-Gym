import pandas as pd

data_demand_sa = pd.DataFrame([
    ['stationary',"(s,S)库存策略", 44158.49, 285.41, 43866.23],
    ['stationary',"A3C", 45553.43, 352.99, 44094.34],
    ['stationary',"PPO", 45395.08, 375.45, 44397.48],
    ['stationary',"MA-DFPPO", 46609.33, 272.91, 46019.73]
],columns=['DemandType', 'Algo','Max_Profit','Std','Mean_Profit'])
#data_demand_sa.iloc[[0,1,2,3]] =

print(data_demand_sa)

#reward: mean 32614.099997363985, std 0.0, max 32614.099997363985, min 32614.099997363985