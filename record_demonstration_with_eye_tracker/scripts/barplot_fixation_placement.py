#!/usr/bin/env python3

# BAR plot for Basic Fixation times

import matplotlib.pyplot as plt
import numpy as np 

my_dpi = 96
n_groups = 2

# Video pouring
relevant_mean = (50.84372122, 57.8359222)
irrelevant_mean = (1.382865013, 1.636616288)
distracting_mean = (8.615726848, 8.029644499)

relevant_std = (4.801264126, 5.119874755)
irrelevant_std = (0.4249314469, 1.021842455)
distracting_std = (2.562025906, 2.882666241)

# KT pouring
# relevant_mean = (43.52789952, 43.02024031)
# irrelevant_mean = (2.920784132, 3.487463666)
# distracting_mean = (6.833694186, 5.651150029)

# relevant_std = (4.594178145, 2.413700268)
# irrelevant_std = (2.279337526, 1.730272784)
# distracting_std = (1.984565412, 1.084465766)
					
						

# create plot
fig, ax = plt.subplots(figsize=(500/my_dpi, 600/my_dpi), dpi=my_dpi)
# index = np.arange(n_groups)
# index = [0, 0.5]
index=[0, 1]
print(index)
bar_width = 0.3
opacity = 0.8
# plt.rcParams.update({'font.size': 18})

rects1 = plt.bar(index, relevant_mean, bar_width, yerr = relevant_std,
alpha=0.5,
ecolor='black',
label='Task Relevant Objects',
capsize = 10)

rects2 = plt.bar([i + bar_width for i in index], distracting_mean, bar_width, yerr = distracting_std, 
alpha=0.5,
ecolor='black',
label='Distracting Objects',
capsize = 10)

rects3 = plt.bar([i + 2*bar_width for i in index], irrelevant_mean, bar_width, yerr = irrelevant_std, 
alpha=0.5,
ecolor='black',
label='Background',
capsize = 10)

#y = (5,10,15,20)
#plt.hlines(y,0,1.5,linestyles='dashed')
ax.yaxis.grid(True)
plt.ylim([0,88]) # video
# plt.ylim([0,70]) # KT

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

# plt.title('Kinesthetic Demonstrations', fontsize=LARGE_SIZE)
plt.title('Video Demonstrations', fontsize=LARGE_SIZE)

plt.xticks([i + bar_width/2 for i in index], ('Novice Users','Expert Users'))
plt.legend(loc=2)

plt.tight_layout()
plt.show()