#!/usr/bin/env python3

# BAR plot for Reference Frame segmentation

import matplotlib.pyplot as plt
import numpy as np 

my_dpi = 96
n_groups = 2

# Experts
# video_mean = (92.85714286, 87.5, 71.875, 93.75, 6.25, 50)
# kt_mean = (27.08333333,	45.31746032, 0, 43.42261905, 16.66666667, 45.45454545)

# video_std = (6.613000713, 7.654655446, 13.5766775, 5.846339667, 5.846339667, 12.5)
# kt_std = (5.546112143, 4.878630351, 0, 7.901104773, 4.561045943, 10.16394535)

# Novices
video_mean = (87.5,	81.25, 62.5, 93.75,	43.75, 12.5)
kt_mean = (27.08333333,	45.31746032, 0, 43.42261905, 16.66666667, 45.45454545)

video_std = (7.654655446, 12.3031373, 14.65754925, 5.846339667,	16.38763825, 11.69267933)
kt_std = (5.546112143, 4.878630351,	0, 7.901104773, 4.561045943, 10.16394535)


# create plot
fig, ax = plt.subplots(figsize=(800/my_dpi, 600/my_dpi), dpi=my_dpi)
# index = np.arange(n_groups)
# index = [0, 0.5]
index=[0, 1, 2, 3, 4, 5]
print(index)
bar_width = 0.3
opacity = 0.8
# plt.rcParams.update({'font.size': 18})

rects1 = plt.bar(index, video_mean, bar_width, yerr = video_std,
alpha=0.5,
ecolor='black',
label='Video Demo',
capsize = 10)

rects2 = plt.bar([i + bar_width for i in index], kt_mean, bar_width, yerr = kt_std, 
alpha=0.5,
ecolor='black',
label='Kinesthetic Demo',
capsize = 10)

#y = (5,10,15,20)
#plt.hlines(y,0,1.5,linestyles='dashed')
ax.yaxis.grid(True)
plt.ylim([0,118])
# plt.xlabel('Tasks')
# plt.ylabel('Time')

# font sizes
SMALL_SIZE = 16
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

plt.xticks([i + bar_width/2 for i in index], ('Reaching','Grasping','Transport','Pouring','Return','Release'))
plt.legend(loc='top right')

plt.tight_layout()
plt.show()