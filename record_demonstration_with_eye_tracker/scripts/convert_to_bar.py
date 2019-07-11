#!/usr/bin/env python3

# BAR plot for Attention to Robot Gripper

import matplotlib.pyplot as plt
import numpy as np 

my_dpi = 96
n_groups = 2
novice_mean = (16.05793159,17.77138289)
expert_mean = (14.7443329,9.180728071)

novice_std = (1.321735449,5.454284248)
expert_std = (0.8862112544,2.354132062)

# create plot
fig, ax = plt.subplots(figsize=(500/my_dpi, 600/my_dpi), dpi=my_dpi)
# index = np.arange(n_groups)
# index = [0, 0.5]
index=[0, 1]
print(index)
bar_width = 0.3
opacity = 0.8
# plt.rcParams.update({'font.size': 18})

rects1 = plt.bar(index, novice_mean, bar_width, yerr = novice_std,
alpha=0.5,
ecolor='black',
label='Novice',
capsize = 10)

rects2 = plt.bar([i + bar_width for i in index], expert_mean, bar_width, yerr = expert_std, 
alpha=0.5,
ecolor='black',
label='Expert',
capsize = 10)

#y = (5,10,15,20)
#plt.hlines(y,0,1.5,linestyles='dashed')
ax.yaxis.grid(True)

# plt.xlabel('Tasks')
# plt.ylabel('Time')

# font sizes
SMALL_SIZE = 18
MEDIUM_SIZE = 20
LARGE_SIZE = 20
plt.rc('font', size=MEDIUM_SIZE)
plt.rc('xtick', labelsize=MEDIUM_SIZE)
plt.rc('ytick', labelsize=MEDIUM_SIZE)
plt.rc('legend', fontsize=SMALL_SIZE)
plt.rc('axes', titlesize=SMALL_SIZE)
plt.rc('axes', labelsize=SMALL_SIZE) 
ax.tick_params(axis='both', which='major', labelsize=SMALL_SIZE)
# ax.tick_params(axis='both', which='minor', labelsize=8)

# plt.title('Attention to Robot Gripper', fontsize=LARGE_SIZE)

plt.xticks([i + bar_width/2 for i in index], ('Pouring\nTask','Placement\nTask'))
plt.legend(loc=2)

plt.tight_layout()
plt.show()