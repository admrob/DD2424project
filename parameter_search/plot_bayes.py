import math 

import pandas as pd
import numpy as np

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

from matplotlib import rc

tuning_new = [{'target': 21.12451107638844, 'params': {'latent_dim': 1468.6165375401004, 'intermediate_dim': 892.3418089348907}},
    {'target': 15.154433700298743, 'params': {'latent_dim': 674.4318880004955, 'intermediate_dim': 100.21731215295529}},
    {'target': 18.755971164131562, 'params': {'latent_dim': 275.4433300607158, 'intermediate_dim': 378.8361925525148}},
    {'target': 19.558897388228097, 'params': {'latent_dim': 756.5653813817908, 'intermediate_dim': 453.8944016175747}},
    {'target': 21.108879994145, 'params': {'latent_dim': 1123.751794606378, 'intermediate_dim': 853.8582010382729}},
    {'target': 21.240378816702968, 'params': {'latent_dim': 1401.9170507538431, 'intermediate_dim': 896.4695773662602}},
    {'target': 20.591492550954406, 'params': {'latent_dim': 1768.4231291427964, 'intermediate_dim': 488.4592744898831}},
    {'target': 16.61474707578163, 'params': {'latent_dim': 1373.8882693389642, 'intermediate_dim': 152.0364270760597}},
    {'target': 21.314390519467434, 'params': {'latent_dim': 1161.5106740469282, 'intermediate_dim': 892.8791244975413}},
    {'target': 18.697452734942516, 'params': {'latent_dim': 476.3928292612697, 'intermediate_dim': 366.73518333094415}},
    {'target': 21.1500878253433, 'params': {'latent_dim': 1939.6969938668553, 'intermediate_dim': 1621.4146804835198}},
    {'target': 21.292680277989273, 'params': {'latent_dim': 1415.4129697716967, 'intermediate_dim': 695.5059385025614}},
    {'target': 19.196792348720493, 'params': {'latent_dim': 1799.75266065731, 'intermediate_dim': 1765.1393893624727}},
    {'target': 17.94855417067779, 'params': {'latent_dim': 174.2040881424765, 'intermediate_dim': 261.58400160257804}},
    {'target': 19.142858329826545, 'params': {'latent_dim': 1768.470756515885, 'intermediate_dim': 422.6777971726809}},
    {'target': 17.905319910809453, 'params': {'latent_dim': 900.1044875095992, 'intermediate_dim': 286.8589842827952}},
    {'target': 19.157868216082388, 'params': {'latent_dim': 1113.0140414487323, 'intermediate_dim': 1919.9901072859536}},
    {'target': 21.30677492573666, 'params': {'latent_dim': 699.4796989115196, 'intermediate_dim': 1414.5665165058995}},
    {'target': 21.032205273485854, 'params': {'latent_dim': 1685.7887766050085, 'intermediate_dim': 1404.351762595009}},
    {'target': 16.766980630208252, 'params': {'latent_dim': 1525.2741983954381, 'intermediate_dim': 134.74772695396445}},
    {'target': 20.993613679035818, 'params': {'latent_dim': 1722.0751819441962, 'intermediate_dim': 688.1315642006846}},
    {'target': 21.58157119630179, 'params': {'latent_dim': 1925.0195578064324, 'intermediate_dim': 1420.6891652177983}},
    {'target': 21.954819430710383, 'params': {'latent_dim': 1831.9666262739004, 'intermediate_dim': 1222.459052722929}},
    {'target': 22.531761178595477, 'params': {'latent_dim': 2000.0, 'intermediate_dim': 1188.9819215402276}},
    {'target': 22.51543593462577, 'params': {'latent_dim': 1964.5228154722054, 'intermediate_dim': 1041.4204462168714}},
    {'target': 21.077096612810777, 'params': {'latent_dim': 2000.0, 'intermediate_dim': 850.1066410541006}},
    {'target': 22.301450260178022, 'params': {'latent_dim': 1833.5545847601677, 'intermediate_dim': 1047.6149242202523}},
    {'target': 21.77231249836044, 'params': {'latent_dim': 1924.0780748985758, 'intermediate_dim': 1125.1166623064978}},
    {'target': 22.496454001361794, 'params': {'latent_dim': 1668.4794446286771, 'intermediate_dim': 1109.0992296952302}},
    {'target': 22.881142389983044, 'params': {'latent_dim': 1714.5154697218593, 'intermediate_dim': 976.5918457554901}},
    {'target': 22.156879645700883, 'params': {'latent_dim': 1808.2369489750934, 'intermediate_dim': 924.2259908602886}},
    {'target': 21.940565472143387, 'params': {'latent_dim': 1632.6051582791893, 'intermediate_dim': 1012.449248855676}},
    {'target': 22.830629737533673, 'params': {'latent_dim': 1735.565161867441, 'intermediate_dim': 1045.9959573451401}},
    {'target': 21.878811989466428, 'params': {'latent_dim': 2000.0, 'intermediate_dim': 1297.734214232365}},
    {'target': 21.96078679408564, 'params': {'latent_dim': 1689.0988717456785, 'intermediate_dim': 1219.9128035614117}},
    {'target': 21.812638559688764, 'params': {'latent_dim': 954.0070265280276, 'intermediate_dim': 1196.5087274641282}},
    {'target': 22.04073660294288, 'params': {'latent_dim': 1179.9483607979996, 'intermediate_dim': 1199.5450873071782}},
    {'target': 21.71173233695473, 'params': {'latent_dim': 1071.3335974641689, 'intermediate_dim': 1378.3333360421425}},
    {'target': 21.724807664636188, 'params': {'latent_dim': 1340.3759651538364, 'intermediate_dim': 1323.7852612724328}},
    {'target': 21.985003992129563, 'params': {'latent_dim': 696.2581729052766, 'intermediate_dim': 1102.917025138305}}]


intermediate = []
latent = []
loss = []
for row in tuning_new:
    intermediate.append(math.floor(row['params']['intermediate_dim']))
    latent.append(math.floor(row['params']['latent_dim']))
    loss.append(row['target'])
data = {'intermediate_dim':intermediate,
        'latent_dim':latent,
        'loss':loss}
df = pd.DataFrame.from_dict(data)

df = df.sort_values('loss')
d = [np.random.uniform(0,1) for i in range(0,len(df.loss))]
col = [tuple((i,0,0)) for i in list(np.sort(d)) ]


fig = plt.figure()

plt.scatter(list(df.intermediate_dim.values), list(df.loss.values), c='r')
plt.xlabel('Intermediate dimensions')
plt.ylabel(r'$\frac{1}{Loss}$')
plt.show()

plt.scatter(df.latent_dim.values, df.loss.values, c='g')
plt.xlabel('Latent dimensions')
plt.ylabel(r'$\frac{1}{Loss}$')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(list(df.intermediate_dim.values), list(df.latent_dim.values), 
               list(df.loss.values),c=col)
ax.set_xlabel('Intermediate dimensions')
ax.set_ylabel('Latent dimension')
ax.set_zlabel(r'$\frac{1}{Loss}$')
plt.show()



