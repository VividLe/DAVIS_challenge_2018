import pickle
import numpy as np
import matplotlib.pyplot as plt

pkl_name = 'data_zd.pkl'
pkl_file = open(pkl_name, 'rb')
distance_set, ratio_set = pickle.load(pkl_file)

distance_np = np.asarray(distance_set)
index = distance_np.argsort()
distance_np.sort()
ratio_np = np.asarray(ratio_set)
ratio_np = ratio_np[index]
# ratio_np.sort()
# ratio_np = 1 - ratio_np
ratio_np *= 30
x = range(len(distance_np))
plt.plot(x[::10], distance_np[::10])
plt.plot(x[::10], ratio_np[::10])
plt.show()
# print(len(distance_set))
# print(len(ratio_set))

# the determined distance TH is 20
